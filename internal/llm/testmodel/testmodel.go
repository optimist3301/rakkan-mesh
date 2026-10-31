// Package testmodel builds a tiny random language model and tokenizer in
// llama2.c's format, for tests and examples that need a model file without
// a download. It generates gibberish, deterministically.
package testmodel

import (
	"encoding/binary"
	"math"
	"math/rand/v2"

	"github.com/optimist3301/rakkan-mesh/internal/llm"
)

// Checkpoint returns the model's checkpoint bytes.
func Checkpoint(seed uint64) []byte {
	c := llm.Config{Dim: 64, Hidden: 160, Layers: 4, Heads: 4, KVHeads: 2, Vocab: 50, SeqLen: 64}
	rng := rand.New(rand.NewPCG(seed, seed))
	r := func(n int, scale float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64() * scale)
		}
		return v
	}
	m := func(n int, scale float64) llm.Matrix { return llm.Matrix{F32: r(n, scale)} }
	d, kv, h := c.Dim, c.KVDim(), c.Hidden
	ck := &llm.Checkpoint{Config: c, Embed: m(c.Vocab*d, 1), Head: &llm.Head{RmsFinal: r(d, 1), Wcls: m(c.Vocab*d, 0.2)}}
	for range c.Layers {
		ck.Layers = append(ck.Layers, &llm.Layer{
			RmsAtt: r(d, 1), Wq: m(d*d, 0.2), Wk: m(d*kv, 0.2), Wv: m(d*kv, 0.2), Wo: m(d*d, 0.2),
			RmsFfn: r(d, 1), W1: m(d*h, 0.2), W2: m(h*d, 0.2), W3: m(d*h, 0.2),
		})
	}
	return ck.Encode()
}

// Tokenizer returns a tokenizer for the model: specials, then characters.
func Tokenizer() []byte {
	vocab := []string{"<unk>", "<s>", "</s>"}
	for _, r := range " abcdefghijklmnopqrstuvwxyz.,!?ABCDEFGHIJKLMNO" {
		vocab = append(vocab, string(r))
	}
	b := binary.LittleEndian.AppendUint32(nil, 16)
	for i, s := range vocab[:50] {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(float32(-i)))
		b = binary.LittleEndian.AppendUint32(b, uint32(len(s)))
		b = append(b, s...)
	}
	return b
}
