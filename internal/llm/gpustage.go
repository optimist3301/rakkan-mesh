package llm

import (
	"math"

	"github.com/optimist3301/rakkan-mesh/internal/gpu"
	"github.com/optimist3301/rakkan-mesh/internal/num"
)

// A stage whose layers are on a GPU runs there entirely: each kernel does
// what Stage.Forward does on the CPU, in the same order of operations, so
// the outputs are bit for bit the CPU's. Activations, the key/value caches
// and the rotary tables live in an arena; the host sends a batch of token
// activations and gets them back after one submission.

// gpuBatch bounds the tokens of one submission, as staging holds them.
const gpuBatch = 64

type gpuStage struct {
	c      Config
	layers []*Layer
	ctx    int
	arena  *gpu.Arena
	// Word offsets in the arena.
	x, xb, xb2, q, k, v, attOut, hb, hb2, att int
	cos, sin                                  int
	keys, vals                                []int
	offs                                      vecOffsets
}

// layersDevice returns the GPU holding all of the layers, or nil.
func layersDevice(c Config, layers []*Layer) *gpu.Device {
	var d *gpu.Device
	for _, l := range layers {
		if l.vecs == nil {
			return nil
		}
		for _, t := range l.tensors(c) {
			if t.mat == nil {
				continue
			}
			if t.mat.dev == nil || d != nil && t.mat.dev.Device() != d {
				return nil
			}
			d = t.mat.dev.Device()
		}
	}
	return d
}

func newGPUStage(d *gpu.Device, c Config, layers []*Layer, rope *Rope, ctx int) (*gpuStage, error) {
	g := &gpuStage{c: c, layers: layers, ctx: ctx}
	at := 0
	alloc := func(n int) int { o := at; at += n; return o }
	g.x, g.xb, g.xb2 = alloc(c.Dim), alloc(c.Dim), alloc(c.Dim)
	g.q, g.k, g.v, g.attOut = alloc(c.QDim()), alloc(c.KVDim()), alloc(c.KVDim()), alloc(c.QDim())
	g.hb, g.hb2 = alloc(c.Hidden), alloc(c.Hidden)
	g.att = alloc(c.Heads * ctx)
	widest := max(c.Dim, c.QDim(), c.Hidden)
	qx, xs := alloc(widest/4), alloc(widest/num.QBlock)
	half := c.HeadSize() / 2
	g.cos, g.sin = alloc(ctx*half), alloc(ctx*half)
	for range layers {
		g.keys = append(g.keys, alloc(ctx*c.KVDim()))
		g.vals = append(g.vals, alloc(ctx*c.KVDim()))
	}
	// Each token of a batch is written and read back.
	staging := max(2*gpuBatch*c.Dim, 2*ctx*half)
	var err error
	if g.arena, err = d.NewArena(at, staging); err != nil {
		return nil, err
	}
	g.arena.SetQuantScratch(qx, xs)
	_, g.offs = layers[0].vecLayout(c)
	b := g.arena.Begin()
	b.Write(g.cos, rope.cos[:ctx*half])
	b.Write(g.sin, rope.sin[:ctx*half])
	if err := b.Submit(); err != nil {
		g.close()
		return nil, err
	}
	return g, nil
}

func (g *gpuStage) close() {
	if g.arena != nil {
		g.arena.Free()
		g.arena = nil
	}
}

// ArenaBytes is the GPU memory a stage of layers takes beyond its weights,
// for ctx positions: caches, tables and activations.
func ArenaBytes(c Config, layers, ctx int) int64 {
	widest := max(c.Dim, c.QDim(), c.Hidden)
	words := 3*c.Dim + 2*c.QDim() + 2*c.KVDim() + 2*c.Hidden + c.Heads*ctx + widest/4 + widest/num.QBlock + ctx*c.HeadSize()
	return 4*int64(words) + CacheBytes(c, layers, ctx)
}

func (g *gpuStage) forward(xs []float32, pos, tokens int) error {
	c := g.c
	for start := 0; start < tokens; start += gpuBatch {
		n := min(gpuBatch, tokens-start)
		b := g.arena.Begin()
		for t := range n {
			x := xs[(start+t)*c.Dim : (start+t+1)*c.Dim]
			b.Write(g.x, x)
			g.token(b, pos+start+t)
			b.Read(g.x, x)
		}
		if err := b.Submit(); err != nil {
			return err
		}
	}
	return nil
}

// token records one position's pass through the layers, as Stage.Forward.
func (g *gpuStage) token(b *gpu.Batch, pos int) {
	c := g.c
	hs, qDim, kvDim := c.HeadSize(), c.QDim(), c.KVDim()
	eps := c.eps()
	recipDim := num.Recip32(float32(c.Dim))
	recipHead := num.Recip32(float32(hs))
	invSqrt := float32(1 / math.Sqrt(float64(hs)))
	half := hs / 2
	o := g.offs
	for li, l := range g.layers {
		b.RMSNorm(l.vecs, g.x, g.xb, o.rmsAtt, c.Dim, 1, 0, recipDim, eps)
		b.MatVecs([]*gpu.Matrix{l.Wq.dev, l.Wk.dev, l.Wv.dev}, []int{g.q, g.k, g.v}, g.xb)
		if c.QKVBias {
			b.AddVec(l.vecs, g.q, o.bq, qDim)
			b.AddVec(l.vecs, g.k, o.bk, kvDim)
			b.AddVec(l.vecs, g.v, o.bv, kvDim)
		}
		if c.QKNorm {
			b.RMSNorm(l.vecs, g.q, g.q, o.qNorm, hs, c.Heads, hs, recipHead, eps)
			b.RMSNorm(l.vecs, g.k, g.k, o.kNorm, hs, c.KVHeads, hs, recipHead, eps)
		}
		b.Rope(g.q, g.k, qDim, kvDim, hs, g.cos+pos*half, g.sin+pos*half)
		b.Copy(g.k, g.keys[li]+pos*kvDim, kvDim)
		b.Copy(g.v, g.vals[li]+pos*kvDim, kvDim)
		b.Attention(g.q, g.keys[li], g.vals[li], g.att, g.ctx, g.attOut, pos+1, c.Heads, c.KVHeads, hs, kvDim, invSqrt)
		b.MatVecs([]*gpu.Matrix{l.Wo.dev}, []int{g.xb2}, g.attOut)
		b.Residual(g.x, g.xb2, c.Dim, false)
		b.RMSNorm(l.vecs, g.x, g.xb, o.rmsFfn, c.Dim, 1, 0, recipDim, eps)
		b.MatVecs([]*gpu.Matrix{l.W1.dev, l.W3.dev}, []int{g.hb, g.hb2}, g.xb)
		b.SiluMul(g.hb, g.hb2, c.Hidden)
		b.MatVecs([]*gpu.Matrix{l.W2.dev}, []int{g.xb}, g.hb)
		b.Residual(g.x, g.xb, c.Dim, true)
	}
}
