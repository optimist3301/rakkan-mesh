// Package llm runs LLaMA-architecture language models, split by layers
// across nodes: pipeline parallelism.
//
// A model is an embedding table, a sequence of transformer layers and a
// head (final normalization and classifier). A Stage runs a contiguous
// range of layers and keeps their key/value cache for one generation; a
// coordinator embeds tokens, passes activations through the stages in
// order, and samples from the head's logits. Running every layer in one
// stage is ordinary local inference.
//
// Every operation uses package num, so a model split over any number of
// nodes, of any architectures, produces exactly the logits of the whole
// model run in one process. Pipelines can therefore be verified and, after
// a node fails, rebuilt mid-generation without changing the output.
//
// Checkpoints use the llama2.c format (Karpathy's tinyllamas, and LLaMA-2
// exports): RMSNorm, rotary position embeddings, grouped-query attention,
// SwiGLU feed-forward, float32 weights.
package llm

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/optimist3301/rakkan-mesh/internal/num"
)

// Config describes a model's shape.
type Config struct {
	Dim     int `json:"dim"`
	Hidden  int `json:"hidden_dim"`
	Layers  int `json:"n_layers"`
	Heads   int `json:"n_heads"`
	KVHeads int `json:"n_kv_heads"`
	Vocab   int `json:"vocab_size"`
	SeqLen  int `json:"seq_len"`
	// SharedClassifier means the classifier is the embedding table.
	SharedClassifier bool `json:"shared_classifier"`
}

func (c Config) HeadSize() int { return c.Dim / c.Heads }
func (c Config) KVDim() int    { return c.Dim * c.KVHeads / c.Heads }

func (c Config) validate() error {
	switch {
	case c.Dim <= 0 || c.Hidden <= 0 || c.Layers <= 0 || c.Heads <= 0 || c.KVHeads <= 0 || c.Vocab <= 0 || c.SeqLen <= 0:
		return errors.New("non-positive dimension")
	case c.Dim%c.Heads != 0 || c.Heads%c.KVHeads != 0 || c.HeadSize()%2 != 0:
		return errors.New("heads do not divide the model dimension evenly")
	case c.Dim > 1<<16 || c.Hidden > 1<<18 || c.Layers > 1024 || c.Vocab > 1<<20 || c.SeqLen > 1<<17:
		return errors.New("dimensions out of range")
	}
	return nil
}

// Layer is one transformer layer's weights. Matrices are row-major
// [out][in]: out[i] = dot(row i, x).
type Layer struct {
	RmsAtt, Wq, Wk, Wv, Wo, RmsFfn, W1, W2, W3 []float32
}

// layerSizes lists a layer's tensors' element counts, in storage order.
func (c Config) layerSizes() []int {
	d, kv, h := c.Dim, c.KVDim(), c.Hidden
	return []int{d, d * d, d * kv, d * kv, d * d, d, d * h, h * d, d * h}
}

func (l *Layer) tensors() []*[]float32 {
	return []*[]float32{&l.RmsAtt, &l.Wq, &l.Wk, &l.Wv, &l.Wo, &l.RmsFfn, &l.W1, &l.W2, &l.W3}
}

// Head is the final normalization and the classifier.
type Head struct {
	RmsFinal []float32
	Wcls     []float32 // [vocab][dim]
}

// Rope holds the rotary embedding's cos and sin for every position and
// pair of a head's dimensions.
type Rope struct {
	cos, sin []float32 // [seq][headSize/2]
	half     int
}

// NewRope computes the rotation table with package num's functions, so it
// is the same everywhere.
func NewRope(c Config) *Rope {
	half := c.HeadSize() / 2
	r := &Rope{cos: make([]float32, c.SeqLen*half), sin: make([]float32, c.SeqLen*half), half: half}
	for p := range half {
		// freq = 10000^(-2p/headSize)
		freq := num.Exp(-math.Log(10000) * float64(2*p) / float64(c.HeadSize()))
		for pos := range c.SeqLen {
			s, co := num.Sincos(float64(float64(pos) * freq))
			r.cos[pos*half+p], r.sin[pos*half+p] = float32(co), float32(s)
		}
	}
	return r
}

// Stage runs layers [first, first+len(layers)) for one generation.
type Stage struct {
	cfg    Config
	layers []*Layer
	rope   *Rope
	keys   [][]float32 // per layer: [seq][kvDim]
	vals   [][]float32
	// Scratch buffers.
	xb, xb2, q, k, v, att, hb, hb2 []float32
}

// NewStage allocates a stage's caches for a generation.
func NewStage(c Config, layers []*Layer, rope *Rope) *Stage {
	s := &Stage{
		cfg: c, layers: layers, rope: rope,
		xb: make([]float32, c.Dim), xb2: make([]float32, c.Dim),
		q: make([]float32, c.Dim), k: make([]float32, c.KVDim()), v: make([]float32, c.KVDim()),
		att: make([]float32, c.SeqLen), hb: make([]float32, c.Hidden), hb2: make([]float32, c.Hidden),
	}
	for range layers {
		s.keys = append(s.keys, make([]float32, c.SeqLen*c.KVDim()))
		s.vals = append(s.vals, make([]float32, c.SeqLen*c.KVDim()))
	}
	return s
}

// CacheBytes is the memory a stage's caches take.
func CacheBytes(c Config, layers int) int64 {
	return int64(layers) * 2 * int64(c.SeqLen) * int64(c.KVDim()) * 4
}

// Forward runs the stage's layers on x, the activation of the token at
// position pos, in place.
func (s *Stage) Forward(x []float32, pos int) {
	c := s.cfg
	hs, kvDim := c.HeadSize(), c.KVDim()
	kvMul := c.Heads / c.KVHeads
	invSqrt := float32(1 / math.Sqrt(float64(hs)))
	for li, l := range s.layers {
		rmsNorm(s.xb, x, l.RmsAtt)
		num.MatVec(s.q, l.Wq, s.xb)
		num.MatVec(s.k, l.Wk, s.xb)
		num.MatVec(s.v, l.Wv, s.xb)

		// Rotary embeddings on queries and keys.
		rc, rs := s.rope.cos[pos*s.rope.half:], s.rope.sin[pos*s.rope.half:]
		for i := 0; i < c.Dim; i += 2 {
			p := (i % hs) / 2
			co, si := rc[p], rs[p]
			rotate(s.q, i, co, si)
			if i < kvDim {
				rotate(s.k, i, co, si)
			}
		}
		copy(s.keys[li][pos*kvDim:], s.k)
		copy(s.vals[li][pos*kvDim:], s.v)

		// Attention, head by head, over positions 0..pos.
		for h := range c.Heads {
			qh := s.q[h*hs : (h+1)*hs]
			off := (h / kvMul) * hs
			att := s.att[:pos+1]
			for t := range att {
				att[t] = float32(num.Dot(qh, s.keys[li][t*kvDim+off:t*kvDim+off+hs]) * invSqrt)
			}
			softmax(att)
			out := s.xb[h*hs : (h+1)*hs]
			clear(out)
			for t, a := range att {
				vt := s.vals[li][t*kvDim+off : t*kvDim+off+hs]
				for i := range out {
					out[i] += float32(a * vt[i])
				}
			}
		}
		num.MatVec(s.xb2, l.Wo, s.xb)
		for i := range x {
			x[i] += s.xb2[i]
		}

		// Feed-forward: w2(silu(w1 x) * w3 x).
		rmsNorm(s.xb, x, l.RmsFfn)
		num.MatVec(s.hb, l.W1, s.xb)
		num.MatVec(s.hb2, l.W3, s.xb)
		for i, v := range s.hb {
			silu := float32(v * float32(1/(1+num.Exp32(-v))))
			s.hb[i] = float32(silu * s.hb2[i])
		}
		num.MatVec(s.xb, l.W2, s.hb)
		for i := range x {
			x[i] = num.Canonical(x[i] + s.xb[i])
		}
	}
}

func rotate(v []float32, i int, co, si float32) {
	v0, v1 := v[i], v[i+1]
	v[i] = float32(v0*co) - float32(v1*si)
	v[i+1] = float32(v0*si) + float32(v1*co)
}

func rmsNorm(out, x, w []float32) {
	ss := num.Dot(x, x)/float32(len(x)) + 1e-5
	inv := 1 / float32(math.Sqrt(float64(ss)))
	for i := range out {
		out[i] = float32(w[i] * float32(inv*x[i]))
	}
}

func softmax(x []float32) {
	mx := x[0]
	for _, v := range x[1:] {
		mx = max(mx, v)
	}
	var sum float32
	for i, v := range x {
		x[i] = num.Exp32(v - mx)
		sum += x[i]
	}
	for i := range x {
		x[i] /= sum
	}
}

// Embed returns the activation for a token.
func Embed(c Config, table []float32, token int) []float32 {
	return append([]float32(nil), table[token*c.Dim:(token+1)*c.Dim]...)
}

// Logits applies the head to the last stage's output.
func (h *Head) Logits(c Config, x []float32) []float32 {
	xn := make([]float32, c.Dim)
	rmsNorm(xn, x, h.RmsFinal)
	logits := make([]float32, c.Vocab)
	num.MatVec(logits, h.Wcls, xn)
	return logits
}

// --- llama2.c checkpoints ---

// Checkpoint is a whole model in memory.
type Checkpoint struct {
	Config Config
	Embed  []float32
	Layers []*Layer
	Head   *Head
}

func floats(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

// ReadCheckpoint parses a llama2.c checkpoint: seven int32s (dim,
// hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len; a
// negative vocab_size means a separate classifier), then the weights,
// grouped by tensor across layers.
func ReadCheckpoint(data []byte) (*Checkpoint, error) {
	if len(data) < 28 {
		return nil, errors.New("checkpoint too short")
	}
	h := make([]int, 7)
	for i := range h {
		h[i] = int(int32(binary.LittleEndian.Uint32(data[4*i:])))
	}
	c := Config{Dim: h[0], Hidden: h[1], Layers: h[2], Heads: h[3], KVHeads: h[4], Vocab: h[5], SeqLen: h[6], SharedClassifier: h[5] > 0}
	if c.Vocab < 0 {
		c.Vocab = -c.Vocab
	}
	if err := c.validate(); err != nil {
		return nil, fmt.Errorf("checkpoint: %w", err)
	}
	off := 28
	take := func(n int) ([]float32, error) {
		if n < 0 || off+4*n > len(data) {
			return nil, errors.New("checkpoint truncated")
		}
		v := floats(data[off : off+4*n])
		off += 4 * n
		return v, nil
	}
	ck := &Checkpoint{Config: c, Head: &Head{}}
	var err error
	if ck.Embed, err = take(c.Vocab * c.Dim); err != nil {
		return nil, err
	}
	for range c.Layers {
		ck.Layers = append(ck.Layers, &Layer{})
	}
	// Each tensor kind is stored for all layers before the next kind.
	for kind, size := range c.layerSizes() {
		for _, l := range ck.Layers {
			if *l.tensors()[kind], err = take(size); err != nil {
				return nil, err
			}
		}
	}
	if ck.Head.RmsFinal, err = take(c.Dim); err != nil {
		return nil, err
	}
	// Legacy rotary tables, unused: computed by NewRope.
	if _, err = take(c.SeqLen * c.HeadSize()); err != nil {
		return nil, err
	}
	if c.SharedClassifier {
		ck.Head.Wcls = ck.Embed
	} else if ck.Head.Wcls, err = take(c.Vocab * c.Dim); err != nil {
		return nil, err
	}
	return ck, nil
}

// Encode writes a checkpoint in llama2.c's format, as ReadCheckpoint reads
// it (the legacy rotary tables are written as zeros).
func (ck *Checkpoint) Encode() []byte {
	c := ck.Config
	vocab := c.Vocab
	if !c.SharedClassifier {
		vocab = -vocab
	}
	var b []byte
	for _, v := range []int{c.Dim, c.Hidden, c.Layers, c.Heads, c.KVHeads, vocab, c.SeqLen} {
		b = binary.LittleEndian.AppendUint32(b, uint32(int32(v)))
	}
	b = append(b, f32Bytes(ck.Embed)...)
	for kind := range c.layerSizes() {
		for _, l := range ck.Layers {
			b = append(b, f32Bytes(*l.tensors()[kind])...)
		}
	}
	b = append(b, f32Bytes(ck.Head.RmsFinal)...)
	b = append(b, make([]byte, 4*c.SeqLen*c.HeadSize())...)
	if !c.SharedClassifier {
		b = append(b, f32Bytes(ck.Head.Wcls)...)
	}
	return b
}
