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
	"slices"

	"github.com/optimist3301/rakkan-mesh/internal/gpu"
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
	// DType stores weight matrices as "f32" (also ""), "bf16", or
	// quantized: "q8_0", "q4_0" (package num), or "q4_m", which is q4_0
	// with the value and down projections in q8_0, as llama.cpp's Q4_K_M
	// keeps them more precise. With q4_0 and q4_m the embedding table and
	// classifier, the most sensitive to precision, are q8_0.
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

func (c Config) quantized() bool { return c.DType == "q8_0" || c.DType == "q4_0" || c.DType == "q4_m" }

// embedType is the type of the embedding table and the classifier.
func (c Config) embedType() string {
	if c.DType == "q4_0" || c.DType == "q4_m" {
		return "q8_0"
	}
	return c.DType
}

// matType is the type of a layer's matrices; q4_m keeps the sensitive ones,
// the value and down projections, in q8_0.
func (c Config) matType(sensitive bool) string {
	switch {
	case c.DType != "q4_m":
		return c.DType
	case sensitive:
		return "q8_0"
	}
	return "q4_0"
}

func (c Config) validate() error {
	switch {
	case c.Dim <= 0 || c.Hidden <= 0 || c.Layers <= 0 || c.Heads <= 0 || c.KVHeads <= 0 || c.Vocab <= 0 || c.SeqLen <= 0 || c.HeadDim < 0:
		return errors.New("non-positive dimension")
	case c.HeadDim == 0 && c.Dim%c.Heads != 0 || c.Heads%c.KVHeads != 0 || c.HeadSize()%2 != 0:
		return errors.New("heads do not divide the model dimension evenly")
	case c.Dim > 1<<16 || c.Hidden > 1<<18 || c.Layers > 1024 || c.Vocab > 1<<20 || c.SeqLen > 1<<20 || c.HeadSize() > 1024 || c.Heads > 1024:
		return errors.New("dimensions out of range")
	case !slices.Contains([]string{"", "f32", "bf16", "q8_0", "q4_0", "q4_m"}, c.DType):
		return fmt.Errorf("unknown weight type %q", c.DType)
	case c.quantized() && (c.Dim%num.QBlock != 0 || c.QDim()%num.QBlock != 0 || c.Hidden%num.QBlock != 0):
		return fmt.Errorf("%s needs widths that are multiples of %d", c.DType, num.QBlock)
	case c.RopeTheta < 0 || c.NormEps < 0 || c.NormEps > 1:
		return errors.New("rope base or norm epsilon out of range")
	}
	return nil
}

// Matrix is a weight matrix, row-major [out][in]: float32, bfloat16, or
// quantized in blocks with one scale each (Q8, Q4 and Scales; see package
// num). Exactly one of F32, BF16, Q8 and Q4 is set.
type Matrix struct {
	F32    []float32
	BF16   []uint16
	Q8     []int8
	Q4     []byte
	Scales []float32

	// dev is the matrix on a GPU, which computes the same bits as the
	// host code; the host copy may have been dropped (see toDevice).
	dev    *gpu.Matrix
	devLen int
}

// Len is the number of elements.
func (m Matrix) Len() int { return max(len(m.F32), len(m.BF16), len(m.Q8), 2*len(m.Q4), m.devLen) }

func (m Matrix) onHost() bool { return m.F32 != nil || m.BF16 != nil || m.Q8 != nil || m.Q4 != nil }

// toDevice copies the matrix, of rows×cols, to a GPU; without keepHost the
// host copy is dropped, so the weights take device memory only.
func (m *Matrix) toDevice(d *gpu.Device, rows, cols int, keepHost bool) error {
	dm, err := d.Upload(rows, cols, gpu.Format{F32: m.F32, BF16: m.BF16, Q8: m.Q8, Q4: m.Q4, Scales: m.Scales})
	if err != nil {
		return err
	}
	m.dev, m.devLen = dm, rows*cols
	if !keepHost {
		m.F32, m.BF16, m.Q8, m.Q4, m.Scales = nil, nil, nil, nil, nil
	}
	return nil
}

// MulVec sets out = m·x. A GPU failure falls back to the host copy, when
// there is one.
func (m Matrix) MulVec(out, x []float32) error {
	if m.dev != nil {
		err := m.dev.MulVec(out, x)
		if err == nil || !m.onHost() {
			return err
		}
	}
	switch {
	case m.Q8 != nil:
		num.MatVecQ8(out, m.Q8, m.Scales, x)
	case m.Q4 != nil:
		num.MatVecQ4(out, m.Q4, m.Scales, x)
	case m.BF16 != nil:
		num.MatVecBF16(out, m.BF16, x)
	default:
		num.MatVec32(out, m.F32, x)
	}
	return nil
}

// mulVecs sets outs[i] = ms[i]·x: on the GPU in one submission when all
// the matrices are on the same device, else one by one.
func mulVecs(ms []Matrix, outs [][]float32, x []float32) error {
	devs := make([]*gpu.Matrix, len(ms))
	for i, m := range ms {
		devs[i] = m.dev
		if m.dev == nil || !gpu.SameDevice(m.dev, ms[0].dev) {
			devs = nil
			break
		}
	}
	if devs != nil {
		err := gpu.MulVecs(devs, outs, x)
		if err == nil || !ms[0].onHost() {
			return err
		}
	}
	for i, m := range ms {
		if err := m.MulVec(outs[i], x); err != nil {
			return err
		}
	}
	return nil
}

// Row copies row i, of n elements, into dst.
func (m Matrix) Row(dst []float32, i int) {
	n := len(dst)
	nb := n / num.QBlock
	switch {
	case m.Q8 != nil:
		num.DequantQ8(dst, m.Q8[i*n:(i+1)*n], m.Scales[i*nb:(i+1)*nb])
	case m.Q4 != nil:
		num.DequantQ4(dst, m.Q4[i*n/2:(i+1)*n/2], m.Scales[i*nb:(i+1)*nb])
	case m.BF16 != nil:
		for j, v := range m.BF16[i*n : (i+1)*n] {
			dst[j] = num.BF16(v)
		}
	default:
		copy(dst, m.F32[i*n:(i+1)*n])
	}
}

// floats returns the matrix's values in float32; quantized matrices are
// not expanded.
func (m Matrix) floats() []float32 {
	if m.BF16 == nil {
		return m.F32
	}
	out := make([]float32, len(m.BF16))
	for i, v := range m.BF16 {
		out[i] = num.BF16(v)
	}
	return out
}

// convert stores a float32 or bfloat16 matrix as dtype.
func (m Matrix) convert(dtype string) Matrix {
	switch dtype {
	case "q8_0":
		q, s := num.QuantizeQ8(m.floats())
		return Matrix{Q8: q, Scales: s}
	case "q4_0":
		q, s := num.QuantizeQ4(m.floats())
		return Matrix{Q4: q, Scales: s}
	case "bf16":
		if m.BF16 != nil {
			return m
		}
		w := make([]uint16, len(m.F32))
		for i, f := range m.F32 {
			w[i] = num.ToBF16(f)
		}
		return Matrix{BF16: w}
	}
	return Matrix{F32: m.floats()}
}

// matBytes is the storage of n elements of a matrix type.
func matBytes(dtype string, n int) int64 {
	switch dtype {
	case "bf16":
		return 2 * int64(n)
	case "q8_0":
		return int64(n) + int64(n/num.QBlock)*4
	case "q4_0":
		return int64(n/2) + int64(n/num.QBlock)*4
	}
	return 4 * int64(n)
}

// Layer is one transformer layer's weights.
type Layer struct {
	RmsAtt         []float32
	Wq, Wk, Wv, Wo Matrix
	Bq, Bk, Bv     []float32 // with QKVBias
	QNorm, KNorm   []float32 // with QKNorm: one head's width
	RmsFfn         []float32
	W1, W2, W3     Matrix // gate, down, up

	// vecs holds the vectors on the GPU holding the matrices (see
	// vecLayout), so the whole layer runs there.
	vecs *gpu.Vectors
}

// vecOffsets are the positions of a layer's vectors in its device buffer.
type vecOffsets struct{ rmsAtt, rmsFfn, bq, bk, bv, qNorm, kNorm int }

func (l *Layer) vecLayout(c Config) ([]float32, vecOffsets) {
	var all []float32
	var o vecOffsets
	put := func(v []float32) int {
		at := len(all)
		all = append(all, v...)
		return at
	}
	o.rmsAtt, o.rmsFfn = put(l.RmsAtt), put(l.RmsFfn)
	if c.QKVBias {
		o.bq, o.bk, o.bv = put(l.Bq), put(l.Bk), put(l.Bv)
	}
	if c.QKNorm {
		o.qNorm, o.kNorm = put(l.QNorm), put(l.KNorm)
	}
	return all, o
}

// tensor is one of a layer's weights: a vector (float32) or a matrix,
// stored as dtype.
type tensor struct {
	vec   *[]float32
	mat   *Matrix
	size  int
	cols  int // a matrix's input width
	dtype string
}

// tensors lists a layer's weights in storage order: llama2.c's nine, then
// the biases and head norms the config calls for.
func (l *Layer) tensors(c Config) []tensor {
	d, q, kv, h, hs := c.Dim, c.QDim(), c.KVDim(), c.Hidden, c.HeadSize()
	m, sensitive := c.matType(false), c.matType(true)
	t := []tensor{
		{vec: &l.RmsAtt, size: d}, {mat: &l.Wq, size: q * d, cols: d, dtype: m}, {mat: &l.Wk, size: kv * d, cols: d, dtype: m},
		{mat: &l.Wv, size: kv * d, cols: d, dtype: sensitive}, {mat: &l.Wo, size: d * q, cols: q, dtype: m}, {vec: &l.RmsFfn, size: d},
		{mat: &l.W1, size: h * d, cols: d, dtype: m}, {mat: &l.W2, size: d * h, cols: h, dtype: sensitive}, {mat: &l.W3, size: h * d, cols: d, dtype: m},
	}
	if c.QKVBias {
		t = append(t, tensor{vec: &l.Bq, size: q}, tensor{vec: &l.Bk, size: kv}, tensor{vec: &l.Bv, size: kv})
	}
	if c.QKNorm {
		t = append(t, tensor{vec: &l.QNorm, size: hs}, tensor{vec: &l.KNorm, size: hs})
	}
	return t
}

// toDevice moves the layer's matrices to a GPU, returning the device
// memory they take. On failure the layer stays on the host.
func (l *Layer) toDevice(c Config, d *gpu.Device) (int64, error) {
	var moved []*Matrix
	var bytes int64
	for _, t := range l.tensors(c) {
		if t.mat == nil {
			continue
		}
		host := *t.mat
		if err := t.mat.toDevice(d, t.size/t.cols, t.cols, false); err != nil {
			*t.mat = host
			for _, m := range moved {
				m.dev.Free()
			}
			return 0, err
		}
		moved = append(moved, t.mat)
		bytes += t.mat.dev.Bytes()
	}
	all, _ := l.vecLayout(c)
	vecs, err := d.UploadVectors(all)
	if err != nil {
		l.release(c)
		return 0, err
	}
	l.vecs = vecs
	return bytes + int64(4*len(all)), nil
}

// release frees the layer's device memory, if any.
func (l *Layer) release(c Config) {
	if l.vecs != nil {
		l.vecs.Free()
		l.vecs = nil
	}
	for _, t := range l.tensors(c) {
		if t.mat != nil && t.mat.dev != nil {
			t.mat.dev.Free()
			t.mat.dev = nil
		}
	}
}

// LayerBytes is a layer's size in memory and in storage.
func LayerBytes(c Config) int64 {
	var n int64
	for _, t := range (&Layer{}).tensors(c) {
		if t.mat != nil {
			n += matBytes(t.dtype, t.size)
		} else {
			n += 4 * int64(t.size)
		}
	}
	return n
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

	// gpu runs the stage on the GPU holding its layers, if they are on one;
	// gpuErr is why that GPU could not take the stage.
	gpu    *gpuStage
	gpuErr error
}

// NewStage allocates a stage's caches for a generation of up to ctx
// positions, at most the rope table's. Layers on a GPU run there, caches
// included.
func NewStage(c Config, layers []*Layer, rope *Rope, ctx int) *Stage {
	ctx = min(ctx, rope.Positions())
	if d := layersDevice(c, layers); d != nil {
		s := &Stage{cfg: c, layers: layers, rope: rope, ctx: ctx}
		s.gpu, s.gpuErr = newGPUStage(d, c, layers, rope, ctx)
		return s
	}
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

// Close frees the stage's GPU memory, if any.
func (s *Stage) Close() {
	if s.gpu != nil {
		s.gpu.close()
		s.gpu = nil
	}
}

// ForwardMany runs tokens activations, each c.Dim wide, at positions pos,
// pos+1, ..., through the stage in place: on a GPU, in one submission.
func (s *Stage) ForwardMany(xs []float32, pos, tokens int) error {
	if s.gpu != nil || s.gpuErr != nil {
		if s.gpuErr != nil {
			return s.gpuErr
		}
		return s.gpu.forward(xs, pos, tokens)
	}
	d := s.cfg.Dim
	for t := range tokens {
		if err := s.Forward(xs[t*d:(t+1)*d], pos+t); err != nil {
			return err
		}
	}
	return nil
}

// Forward runs the stage's layers on x, the activation of the token at
// position pos (below Positions), in place. It fails only if a GPU does.
func (s *Stage) Forward(x []float32, pos int) error {
	if s.gpu != nil || s.gpuErr != nil {
		return s.ForwardMany(x, pos, 1)
	}
	c := s.cfg
	hs, qDim, kvDim := c.HeadSize(), c.QDim(), c.KVDim()
	kvMul := c.Heads / c.KVHeads
	invSqrt := float32(1 / math.Sqrt(float64(hs)))
	eps := c.eps()
	var err error
	mul := func(m Matrix, out, in []float32) {
		if err == nil {
			err = m.MulVec(out, in)
		}
	}
	for li, l := range s.layers {
		rmsNorm(s.xb, x, l.RmsAtt, eps)
		if err := mulVecs([]Matrix{l.Wq, l.Wk, l.Wv}, [][]float32{s.q, s.k, s.v}, s.xb); err != nil {
			return err
		}
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
		mul(l.Wo, s.xb2, s.attOut)
		if err != nil {
			return err
		}
		for i := range x {
			x[i] += s.xb2[i]
		}

		// Feed-forward: w2(silu(w1 x) * w3 x).
		rmsNorm(s.xb, x, l.RmsFfn, eps)
		if err := mulVecs([]Matrix{l.W1, l.W3}, [][]float32{s.hb, s.hb2}, s.xb); err != nil {
			return err
		}
		for i, v := range s.hb {
			s.hb[i] = float32(silu(v) * s.hb2[i])
		}
		mul(l.W2, s.xb, s.hb)
		if err != nil {
			return err
		}
		for i := range x {
			x[i] = num.Canonical(x[i] + s.xb[i])
		}
	}
	return nil
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

// Divisions and square roots go through num.Recip32 and num.Rsqrt32,
// which GPUs compute identically (see package num), so a layer runs the
// same on either.

// rmsNorm normalizes x into out, which may be x.
func rmsNorm(out, x, w []float32, eps float32) {
	ss := float32(num.Dot32(x, x)*num.Recip32(float32(len(x)))) + eps
	inv := num.Rsqrt32(ss)
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
	inv := num.Recip32(sum)
	for i := range x {
		x[i] = float32(x[i] * inv)
	}
}

// silu is x·sigmoid(x).
func silu(v float32) float32 { return float32(v * num.Recip32(1+num.Exp32(-v))) }

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
	h.Wcls.MulVec(logits, xn) // the head keeps its host copy: never fails
	return logits
}

// Quantize converts the checkpoint's matrices to dtype ("bf16", "q8_0" or
// "q4_0"). The llama2.c format holds only float32, so a quantized
// checkpoint is imported (Import), not encoded.
func (ck *Checkpoint) Quantize(dtype string) error {
	c := ck.Config
	c.DType = dtype
	if err := c.validate(); err != nil {
		return err
	}
	for _, l := range ck.Layers {
		for _, t := range l.tensors(c) {
			if t.mat != nil {
				*t.mat = t.mat.convert(t.dtype)
			}
		}
	}
	shared := c.SharedClassifier
	ck.Embed = ck.Embed.convert(c.embedType())
	if shared {
		ck.Head.Wcls = ck.Embed
	} else {
		ck.Head.Wcls = ck.Head.Wcls.convert(c.embedType())
	}
	ck.Config = c
	return nil
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
