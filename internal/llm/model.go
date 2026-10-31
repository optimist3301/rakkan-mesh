// Package llm runs decoder-only transformer language models, split by
// layers across nodes: pipeline parallelism.
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
// The architecture is the LLaMA family's: RMSNorm, rotary position
// embeddings, grouped-query attention, SwiGLU feed-forward; with the
// variations of Qwen2 (biases on the query, key and value projections) and
// Qwen3 (RMSNorm of each query and key head, head width independent of the
// model width). Weights come from llama2.c checkpoints (float32) or
// Hugging Face safetensors (see ImportHF); matrices are kept in float32 or
// bfloat16 and multiplied in float32.
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
	// HeadDim is a head's width; 0 means Dim/Heads.
	HeadDim int `json:"head_dim,omitempty"`
	// RopeTheta is the rotary embedding's base; 0 means 10000.
	RopeTheta float64 `json:"rope_theta,omitempty"`
	// NormEps is RMSNorm's epsilon; 0 means 1e-5.
	NormEps float64 `json:"norm_eps,omitempty"`
	// QKVBias adds biases to the query, key and value projections (Qwen2).
	QKVBias bool `json:"qkv_bias,omitempty"`
	// QKNorm normalizes each query and key head before the rotary
	// embedding (Qwen3).
	QKNorm bool `json:"qk_norm,omitempty"`
	// DType stores weight matrices as "f32" (also "") or "bf16".
	DType string `json:"dtype,omitempty"`
}

func (c Config) HeadSize() int {
	if c.HeadDim > 0 {
		return c.HeadDim
	}
	return c.Dim / c.Heads
}

// QDim is the width of all query heads together.
func (c Config) QDim() int { return c.Heads * c.HeadSize() }

// KVDim is the width of all key (or value) heads together.
func (c Config) KVDim() int { return c.KVHeads * c.HeadSize() }

func (c Config) theta() float64 {
	if c.RopeTheta > 0 {
		return c.RopeTheta
	}
	return 10000
}

func (c Config) eps() float32 {
	if c.NormEps > 0 {
		return float32(c.NormEps)
	}
	return 1e-5
}

func (c Config) validate() error {
	switch {
	case c.Dim <= 0 || c.Hidden <= 0 || c.Layers <= 0 || c.Heads <= 0 || c.KVHeads <= 0 || c.Vocab <= 0 || c.SeqLen <= 0 || c.HeadDim < 0:
		return errors.New("non-positive dimension")
	case c.HeadDim == 0 && c.Dim%c.Heads != 0 || c.Heads%c.KVHeads != 0 || c.HeadSize()%2 != 0:
		return errors.New("heads do not divide the model dimension evenly")
	case c.Dim > 1<<16 || c.Hidden > 1<<18 || c.Layers > 1024 || c.Vocab > 1<<20 || c.SeqLen > 1<<20 || c.HeadSize() > 1024 || c.Heads > 1024:
		return errors.New("dimensions out of range")
	case c.DType != "" && c.DType != "f32" && c.DType != "bf16":
		return fmt.Errorf("unknown weight type %q", c.DType)
	case c.RopeTheta < 0 || c.NormEps < 0 || c.NormEps > 1:
		return errors.New("rope base or norm epsilon out of range")
	}
	return nil
}

// Matrix is a weight matrix, row-major [out][in], in float32 or bfloat16.
type Matrix struct {
	F32  []float32
	BF16 []uint16
}

// Len is the number of elements.
func (m Matrix) Len() int { return max(len(m.F32), len(m.BF16)) }

// MulVec sets out = m·x.
func (m Matrix) MulVec(out, x []float32) {
	if m.BF16 != nil {
		num.MatVecBF16(out, m.BF16, x)
	} else {
		num.MatVec32(out, m.F32, x)
	}
}

// Row copies row i, of n elements, into dst.
func (m Matrix) Row(dst []float32, i int) {
	n := len(dst)
	if m.BF16 != nil {
		for j, v := range m.BF16[i*n : (i+1)*n] {
			dst[j] = num.BF16(v)
		}
		return
	}
	copy(dst, m.F32[i*n:(i+1)*n])
}

// newMatrix allocates a zero matrix of n elements in the config's type.
func newMatrix(c Config, n int) Matrix {
	if c.DType == "bf16" {
		return Matrix{BF16: make([]uint16, n)}
	}
	return Matrix{F32: make([]float32, n)}
}

// Layer is one transformer layer's weights.
type Layer struct {
	RmsAtt         []float32
	Wq, Wk, Wv, Wo Matrix
	Bq, Bk, Bv     []float32 // with QKVBias
	QNorm, KNorm   []float32 // with QKNorm: one head's width
	RmsFfn         []float32
	W1, W2, W3     Matrix // gate, down, up
}

// tensor is one of a layer's weights: a vector (float32) or a matrix.
type tensor struct {
	vec  *[]float32
	mat  *Matrix
	size int
}

// tensors lists a layer's weights in storage order: llama2.c's nine, then
// the biases and head norms the config calls for.
func (l *Layer) tensors(c Config) []tensor {
	d, q, kv, h, hs := c.Dim, c.QDim(), c.KVDim(), c.Hidden, c.HeadSize()
	t := []tensor{
		{vec: &l.RmsAtt, size: d}, {mat: &l.Wq, size: q * d}, {mat: &l.Wk, size: kv * d}, {mat: &l.Wv, size: kv * d},
		{mat: &l.Wo, size: d * q}, {vec: &l.RmsFfn, size: d}, {mat: &l.W1, size: h * d}, {mat: &l.W2, size: d * h},
		{mat: &l.W3, size: h * d},
	}
	if c.QKVBias {
		t = append(t, tensor{vec: &l.Bq, size: q}, tensor{vec: &l.Bk, size: kv}, tensor{vec: &l.Bv, size: kv})
	}
	if c.QKNorm {
		t = append(t, tensor{vec: &l.QNorm, size: hs}, tensor{vec: &l.KNorm, size: hs})
	}
	return t
}

// LayerBytes is a layer's size in memory and in storage.
func LayerBytes(c Config) int64 {
	var n int64
	for _, t := range (&Layer{}).tensors(c) {
		n += int64(t.size) * elemBytes(c, t.mat != nil)
	}
	return n
}

func elemBytes(c Config, matrix bool) int64 {
	if matrix && c.DType == "bf16" {
		return 2
	}
	return 4
}

// StageGas prices a forward through layers for tokens in gas units, for
// credit accounting: eight per weight per token, as tensor kernels charge
// per multiply-add (package sandbox).
func StageGas(c Config, layers, tokens int) int64 {
	var weights int64
	for _, t := range (&Layer{}).tensors(c) {
		weights += int64(t.size)
	}
	return 8 * weights * int64(layers) * int64(tokens)
}

// Head is the final normalization and the classifier.
type Head struct {
	RmsFinal []float32
	Wcls     Matrix // [vocab][dim]
}

// Rope holds the rotary embedding's cos and sin for every position and
// pair of a head's dimensions.
type Rope struct {
	cos, sin []float32 // [seq][headSize/2]
	half     int
}

// NewRope computes the rotation table for positions below n with package
// num's functions, so it is the same everywhere.
func NewRope(c Config, n int) *Rope {
	hs := c.HeadSize()
	half := hs / 2
	r := &Rope{cos: make([]float32, n*half), sin: make([]float32, n*half), half: half}
	for p := range half {
		// freq = theta^(-2p/headSize)
		freq := num.Exp(-math.Log(c.theta()) * float64(2*p) / float64(hs))
		for pos := range n {
			s, co := num.Sincos(float64(float64(pos) * freq))
			r.cos[pos*half+p], r.sin[pos*half+p] = float32(co), float32(s)
		}
	}
	return r
}

// Positions is how many positions the table covers.
func (r *Rope) Positions() int { return len(r.cos) / r.half }

// Stage runs layers [first, first+len(layers)) for one generation.
type Stage struct {
	cfg    Config
	layers []*Layer
	rope   *Rope
	ctx    int
	keys   [][]float32 // per layer: [ctx][kvDim]
	vals   [][]float32
	// Scratch buffers.
	xb, xb2, q, k, v, att, attOut, hb, hb2 []float32
}

// NewStage allocates a stage's caches for a generation of up to ctx
// positions, at most the rope table's.
func NewStage(c Config, layers []*Layer, rope *Rope, ctx int) *Stage {
	ctx = min(ctx, rope.Positions())
	s := &Stage{
		cfg: c, layers: layers, rope: rope, ctx: ctx,
		xb: make([]float32, c.Dim), xb2: make([]float32, c.Dim),
		q: make([]float32, c.QDim()), k: make([]float32, c.KVDim()), v: make([]float32, c.KVDim()),
		att: make([]float32, ctx), attOut: make([]float32, c.QDim()), hb: make([]float32, c.Hidden), hb2: make([]float32, c.Hidden),
	}
	for range layers {
		s.keys = append(s.keys, make([]float32, ctx*c.KVDim()))
		s.vals = append(s.vals, make([]float32, ctx*c.KVDim()))
	}
	return s
}

// Positions is the stage's context length.
func (s *Stage) Positions() int { return s.ctx }

// CacheBytes is the memory a stage's caches take for ctx positions.
func CacheBytes(c Config, layers, ctx int) int64 {
	return int64(layers) * 2 * int64(ctx) * int64(c.KVDim()) * 4
}

// Forward runs the stage's layers on x, the activation of the token at
// position pos (below Positions), in place.
func (s *Stage) Forward(x []float32, pos int) {
	c := s.cfg
	hs, qDim, kvDim := c.HeadSize(), c.QDim(), c.KVDim()
	kvMul := c.Heads / c.KVHeads
	invSqrt := float32(1 / math.Sqrt(float64(hs)))
	eps := c.eps()
	for li, l := range s.layers {
		rmsNorm(s.xb, x, l.RmsAtt, eps)
		l.Wq.MulVec(s.q, s.xb)
		l.Wk.MulVec(s.k, s.xb)
		l.Wv.MulVec(s.v, s.xb)
		if c.QKVBias {
			addTo(s.q, l.Bq)
			addTo(s.k, l.Bk)
			addTo(s.v, l.Bv)
		}
		if c.QKNorm {
			for h := 0; h < qDim; h += hs {
				rmsNorm(s.q[h:h+hs], s.q[h:h+hs], l.QNorm, eps)
			}
			for h := 0; h < kvDim; h += hs {
				rmsNorm(s.k[h:h+hs], s.k[h:h+hs], l.KNorm, eps)
			}
		}

		// Rotary embeddings on queries and keys, on pairs of adjacent
		// dimensions within each head.
		rc, rs := s.rope.cos[pos*s.rope.half:], s.rope.sin[pos*s.rope.half:]
		for i := 0; i < qDim; i += 2 {
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
				att[t] = float32(num.Dot32(qh, s.keys[li][t*kvDim+off:t*kvDim+off+hs]) * invSqrt)
			}
			softmax(att)
			out := s.attOut[h*hs : (h+1)*hs]
			clear(out)
			for t, a := range att {
				vt := s.vals[li][t*kvDim+off : t*kvDim+off+hs]
				for i := range out {
					out[i] += float32(a * vt[i])
				}
			}
		}
		l.Wo.MulVec(s.xb2, s.attOut)
		for i := range x {
			x[i] += s.xb2[i]
		}

		// Feed-forward: w2(silu(w1 x) * w3 x).
		rmsNorm(s.xb, x, l.RmsFfn, eps)
		l.W1.MulVec(s.hb, s.xb)
		l.W3.MulVec(s.hb2, s.xb)
		for i, v := range s.hb {
			silu := float32(v * float32(1/(1+num.Exp32(-v))))
			s.hb[i] = float32(silu * s.hb2[i])
		}
		l.W2.MulVec(s.xb, s.hb)
		for i := range x {
			x[i] = num.Canonical(x[i] + s.xb[i])
		}
	}
}

func addTo(v, b []float32) {
	for i := range v {
		v[i] += b[i]
	}
}

func rotate(v []float32, i int, co, si float32) {
	v0, v1 := v[i], v[i+1]
	v[i] = float32(v0*co) - float32(v1*si)
	v[i+1] = float32(v0*si) + float32(v1*co)
}

// rmsNorm normalizes x into out, which may be x.
func rmsNorm(out, x, w []float32, eps float32) {
	ss := num.Dot32(x, x)/float32(len(x)) + eps
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
func Embed(c Config, table Matrix, token int) []float32 {
	x := make([]float32, c.Dim)
	table.Row(x, token)
	return x
}

// Logits applies the head to the last stage's output.
func (h *Head) Logits(c Config, x []float32) []float32 {
	xn := make([]float32, c.Dim)
	rmsNorm(xn, x, h.RmsFinal, c.eps())
	logits := make([]float32, c.Vocab)
	h.Wcls.MulVec(logits, xn)
	return logits
}

// --- llama2.c checkpoints ---

// Checkpoint is a whole model in memory.
type Checkpoint struct {
	Config Config
	Embed  Matrix
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
	if ck.Embed.F32, err = take(c.Vocab * c.Dim); err != nil {
		return nil, err
	}
	for range c.Layers {
		ck.Layers = append(ck.Layers, &Layer{})
	}
	// Each tensor kind is stored for all layers before the next kind.
	for kind := range (&Layer{}).tensors(c) {
		for _, l := range ck.Layers {
			t := l.tensors(c)[kind]
			v, err := take(t.size)
			if err != nil {
				return nil, err
			}
			if t.mat != nil {
				t.mat.F32 = v
			} else {
				*t.vec = v
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
	} else if ck.Head.Wcls.F32, err = take(c.Vocab * c.Dim); err != nil {
		return nil, err
	}
	return ck, nil
}

// Encode writes a checkpoint in llama2.c's format, as ReadCheckpoint reads
// it (the legacy rotary tables are written as zeros). The format holds
// only float32 LLaMA models.
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
	b = append(b, f32Bytes(ck.Embed.F32)...)
	for kind := range (&Layer{}).tensors(c) {
		for _, l := range ck.Layers {
			t := l.tensors(c)[kind]
			if t.mat != nil {
				b = append(b, f32Bytes(t.mat.F32)...)
			} else {
				b = append(b, f32Bytes(*t.vec)...)
			}
		}
	}
	b = append(b, f32Bytes(ck.Head.RmsFinal)...)
	b = append(b, make([]byte, 4*c.SeqLen*c.HeadSize())...)
	if !c.SharedClassifier {
		b = append(b, f32Bytes(ck.Head.Wcls.F32)...)
	}
	return b
}
