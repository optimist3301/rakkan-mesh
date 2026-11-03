package dsv41

import (
	"encoding/json"
	"fmt"
	"math"
	"os"

	"github.com/optimist3301/rakkan-mesh/internal/num"
)

// EngramTables are what hashing n-grams needs besides the tokens: the
// compressed token map, and per Engram layer the bucket primes, row
// offsets and hash multipliers (docs/dev/deepseek-v41.md, section 8). They
// depend only on the tokenizer and the config.
//
// TODO(stage 2): build them from the tokenizer in Go. Until then they come
// from a file in the format tools/llmref/dsv41 writes (engram.json).
type EngramTables struct {
	MaxNgram   int     `json:"max_ngram_size"`
	Heads      int     `json:"n_heads"`
	Compressed int     `json:"compressed_vocab_size"`
	Pad        int64   `json:"pad_id"`
	TokenMap   []int64 `json:"token_map"`
	Layers     []struct {
		Layer       int       `json:"layer"`
		Rows        int       `json:"num_embeddings"`
		Primes      [][]int64 `json:"primes"`      // [ngram size - 2][head]
		Offsets     []int64   `json:"offsets"`     // [(ngram size - 2)·heads + head]
		Multipliers []int64   `json:"multipliers"` // [lookback]
	} `json:"layers"`
}

// LoadEngramTables reads tables in the engram.json format.
func LoadEngramTables(path string) (*EngramTables, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var t EngramTables
	if err := json.Unmarshal(b, &t); err != nil {
		return nil, fmt.Errorf("engram tables: %w", err)
	}
	return &t, nil
}

func (t *EngramTables) check(c *Config) error {
	switch {
	case t == nil:
		return fmt.Errorf("dsv41: the model has Engram layers and no hash tables")
	case t.MaxNgram != c.EngramMaxNgram || t.Heads != c.EngramHeads || len(t.TokenMap) != c.Vocab || len(t.Layers) != len(c.EngramLayers):
		return fmt.Errorf("dsv41: Engram tables do not fit the config")
	}
	for i, l := range t.Layers {
		if l.Layer != c.EngramLayers[i] || l.Rows != c.EngramRows[i] || len(l.Primes) != c.EngramMaxNgram-1 ||
			len(l.Offsets) != (c.EngramMaxNgram-1)*c.EngramHeads || len(l.Multipliers) != c.EngramMaxNgram {
			return fmt.Errorf("dsv41: Engram tables of layer %d do not fit the config", l.Layer)
		}
	}
	return nil
}

// hash returns, for each Engram layer, the table rows of the n-grams
// ending at the last token of hist (the compressed ids of the last
// MaxNgram tokens or fewer, oldest first; a short history pads).
func (t *EngramTables) hash(hist []int64) [][]int64 {
	n := t.MaxNgram
	toks := make([]int64, n) // toks[k]: the token k back, or the pad
	for k := range n {
		if k < len(hist) {
			toks[k] = hist[len(hist)-1-k]
		} else {
			toks[k] = t.Pad
		}
	}
	out := make([][]int64, len(t.Layers))
	for li, l := range t.Layers {
		ids := make([]int64, 0, (n-1)*t.Heads)
		rolling := toks[0] * l.Multipliers[0]
		for k := 1; k < n; k++ {
			rolling ^= toks[k] * l.Multipliers[k]
			for h, p := range l.Primes[k-1] {
				ids = append(ids, rolling%p+l.Offsets[(k-1)*t.Heads+h])
			}
		}
		out[li] = ids
	}
	return out
}

// apply adds the layer's n-gram memory to the stream h ([hc][dim]): each
// copy is gated by how well it matches that copy's key.
func (e *engram) apply(h []float32, ids []int64, c *Config, act *num.ActFP8) {
	dim, hc := c.Dim, c.HCMult
	emb := make([]float32, len(ids)*e.dim)
	for i, row := range ids {
		q := e.rows[row*int64(e.dim) : (row+1)*int64(e.dim)]
		sc := e.rowScales[row*int64(e.dim/32) : (row+1)*int64(e.dim/32)]
		for j, v := range q {
			emb[i*e.dim+j] = bf16(num.DecodeE4M3(v) * num.DecodeE8M0(sc[j/32]))
		}
	}
	kv := make([]float32, dim*(hc+1))
	e.wkv.apply(kv, emb, act)
	value := kv[hc*dim:]
	eps := c.NormEps
	scale := float32(1 / math.Sqrt(float64(dim)))
	tmp := make([]float32, dim)
	for k := range hc {
		x, key, w := h[k*dim:(k+1)*dim], kv[k*dim:(k+1)*dim], e.qk[k*dim:(k+1)*dim]
		rstd := float32(rsqrt(meanSquare(x)+eps) * rsqrt(meanSquare(key)+eps))
		for d := range dim {
			tmp[d] = float32(float32(x[d]*w[d]) * key[d])
		}
		dot := float32(float32(sum32(tmp)*rstd) * scale)
		g := sigmoid(float32(math.Copysign(float64(sqrt32(max(float32(math.Abs(float64(dot))), 1e-6))), float64(dot))))
		for d := range dim {
			x[d] = bf16(x[d] + float32(g*value[d]))
		}
	}
}
