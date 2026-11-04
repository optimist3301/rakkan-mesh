package dsv41

import (
	"errors"
	"fmt"
	"math"

	"github.com/optimist3301/rakkan-mesh/internal/gpu"
)

// GPUVision is the image encoder on a GPU (package gpu's vision kernels):
// the same bits as Vision's on the CPU. Its matrices stay on the device;
// each image gets an arena of its size. The aligner's GELU (erf in
// float64) runs on the CPU, between its two products on the GPU.
type GPUVision struct {
	vis    *Vision
	d      *gpu.Device
	patch  *gpu.Matrix
	blocks []gpuVisionBlock
	align1 *gpu.Matrix
	align2 *gpu.Matrix
	vecs   []float32 // norms and biases, written at the start of each arena
	off    visionVecs
	bytes  int64
}

type gpuVisionBlock struct {
	qkv, o, w1, w2 *gpu.Matrix
	norm1, norm2   int // offsets in vecs
	qkvB, oB       int
}

type visionVecs struct {
	patchB, norm, align1B, align2B int
}

// ToGPU uploads the encoder's matrices to d.
func (vis *Vision) ToGPU(d *gpu.Device) (*GPUVision, error) {
	v := &vis.cfg.Vision
	if hd := v.Dim / v.Heads; hd != 16 && hd != 32 && hd != 64 {
		return nil, fmt.Errorf("dsv41: vision heads of %d on a GPU", hd)
	}
	g := &GPUVision{vis: vis, d: d}
	vec := func(x []float32) int {
		off := len(g.vecs)
		g.vecs = append(g.vecs, x...)
		return off
	}
	var err error
	up := func(l *bf16Linear) *gpu.Matrix {
		if err != nil {
			return nil
		}
		var m *gpu.Matrix
		if m, err = d.Upload(l.out, l.in, gpu.Format{BF16: l.w}); err == nil {
			g.bytes += m.Bytes()
		}
		return m
	}
	g.patch = up(vis.patch)
	g.off.patchB = vec(vis.patch.b)
	for _, b := range vis.blocks {
		gb := gpuVisionBlock{qkv: up(b.qkv), o: up(b.o), w1: up(b.w1), w2: up(b.w2)}
		gb.norm1, gb.norm2 = vec(b.norm1), vec(b.norm2)
		gb.qkvB, gb.oB = vec(b.qkv.b), vec(b.o.b)
		g.blocks = append(g.blocks, gb)
	}
	g.off.norm = vec(vis.norm)
	g.align1, g.align2 = up(vis.align1), up(vis.align2)
	g.off.align1B, g.off.align2B = vec(vis.align1.b), vec(vis.align2.b)
	if err != nil {
		g.Free()
		return nil, err
	}
	return g, nil
}

// Bytes is the device memory the matrices hold.
func (g *GPUVision) Bytes() int64 { return g.bytes }

// Free releases the matrices.
func (g *GPUVision) Free() {
	for _, m := range []*gpu.Matrix{g.patch, g.align1, g.align2} {
		if m != nil {
			m.Free()
		}
	}
	for _, b := range g.blocks {
		for _, m := range []*gpu.Matrix{b.qkv, b.o, b.w1, b.w2} {
			if m != nil {
				m.Free()
			}
		}
	}
	g.blocks = nil
}

// ArenaBytes is the device memory encoding an image of n patches takes
// besides the matrices.
func (g *GPUVision) ArenaBytes(n int) int64 {
	words, _ := g.layout(n, 0)
	return 4 * int64(words)
}

// arena regions, word offsets.
type visionArena struct {
	vecs, cos, sin, patches int
	x, h, qkv, y, gu, a     int
	feat, h1, out           int
	words, staging          int
}

func (g *GPUVision) layout(n, L int) (int, visionArena) {
	v := &g.vis.cfg.Vision
	d, p, dim := v.Dim, v.Patch, g.vis.cfg.Dim
	half := d / v.Heads / 2
	if L == 0 { // the most an image of n patches has
		L = n
	}
	var a visionArena
	next := 0
	take := func(words int) int {
		off := next
		next += words
		return off
	}
	a.vecs = take(len(g.vecs))
	a.cos, a.sin = take(n*half), take(n*half)
	a.patches = take(n * 3 * p * p)
	a.x, a.h, a.y = take(n*d), take(n*d), take(n*d)
	a.qkv = take(n * 3 * d)
	a.gu, a.a = take(n*2*v.Inter), take(n*v.Inter)
	// The aligner's in the encoder's, which it no longer needs.
	r := v.Downsample
	a.feat, a.h1, a.out = a.qkv, a.qkv+L*d*r*r, a.qkv+L*d*r*r+L*dim
	next = max(next, a.out+L*dim)
	a.words = next
	// A small image goes in one submission: its writes and the read back.
	a.staging = max(len(g.vecs)+2*n*half+n*3*p*p+n*d, L*d*r*r+L*dim, 2*L*dim)
	return a.words, a
}

// Encode is Vision.Encode on the GPU.
func (g *GPUVision) Encode(img *Image) ([]float32, error) {
	vis := g.vis
	x, a, err := g.vit(img, true)
	if err != nil {
		return nil, err
	}
	defer a.Free()
	feat, L := vis.unfold(x, img.VitH, img.VitW)
	_, lay := g.layout(img.VitH*img.VitW, L)
	dim := vis.cfg.Dim
	in := len(feat) / L
	q := &visionQueue{a: a}
	q.b = a.Begin()
	q.b.Write(lay.feat, feat)
	q.matmul(g.align1, dim, in, L, lay.h1, dim, lay.feat, in, lay.vecs+g.off.align1B)
	h := make([]float32, L*dim)
	q.add(0, func(b *gpu.Batch) { b.Read(lay.h1, h) })
	if err := q.submit(); err != nil {
		q.b.Submit()
		return nil, err
	}
	for i, t := range h {
		h[i] = bf16c(gelu(t))
	}
	q.b.Write(lay.h1, h)
	q.matmul(g.align2, dim, dim, L, lay.out, dim, lay.h1, dim, lay.vecs+g.off.align2B)
	out := make([]float32, L*dim)
	q.add(0, func(b *gpu.Batch) { b.Read(lay.out, out) })
	err = q.submit()
	q.b.Submit()
	if err != nil {
		return nil, err
	}
	return out, nil
}

// submitWork bounds the multiply-adds of one submission: a long one makes
// drivers reset the device (a few seconds on desktops' GPUs).
const submitWork = 1 << 31

// vAttnQueries is the attention kernel's queries per workgroup.
const vAttnQueries = 64

// visionQueue records operations into batches of at most submitWork.
type visionQueue struct {
	a    *gpu.Arena
	b    *gpu.Batch
	work int64
	err  error
}

// matmul is VMatMul of n inputs by an m of rows×cols, in parts of inputs
// that fit the budget.
func (q *visionQueue) matmul(m *gpu.Matrix, rows, cols, n, out, outStride, in, inStride, bias int) {
	per := max(1, int(submitWork/int64(rows*cols)))
	for r0 := 0; r0 < n; r0 += per {
		nr := min(per, n-r0)
		q.add(int64(nr*rows*cols), func(b *gpu.Batch) {
			b.VMatMul(m, out+r0*outStride, outStride, in+r0*inStride, inStride, nr, bias)
		})
	}
}

// submit sends what is queued.
func (q *visionQueue) submit() error {
	err := q.b.Submit()
	q.b, q.work = q.a.Begin(), 0
	return errors.Join(q.err, err)
}

func (q *visionQueue) add(work int64, op func(b *gpu.Batch)) {
	if q.err != nil {
		return
	}
	if q.work > 0 && q.work+work > submitWork {
		if q.err = q.b.Submit(); q.err != nil {
			q.b = q.a.Begin()
			return
		}
		q.b, q.work = q.a.Begin(), 0
	}
	op(q.b)
	q.work += work
}

// ViT is Vision.ViT on the GPU.
func (g *GPUVision) ViT(img *Image) ([]float32, error) {
	x, a, err := g.vit(img, false)
	if a != nil {
		a.Free()
	}
	return x, err
}

// vit runs the encoder and returns its output, and the arena if keep (for
// the aligner to go on in).
func (g *GPUVision) vit(img *Image, keep bool) ([]float32, *gpu.Arena, error) {
	vis := g.vis
	v := &vis.cfg.Vision
	n, d, p := img.VitH*img.VitW, v.Dim, v.Patch
	if n == 0 || len(img.Patches) != n*3*p*p {
		return nil, nil, fmt.Errorf("dsv41: %d patch values for a %d×%d grid", len(img.Patches), img.VitH, img.VitW)
	}
	if len(g.blocks) != len(vis.blocks) {
		return nil, nil, errors.New("dsv41: the GPU encoder was freed")
	}
	L := ceilDiv(img.VitH, v.Downsample) * ceilDiv(img.VitW, v.Downsample)
	words, lay := g.layout(n, L)
	a, err := g.d.NewArena(words, lay.staging)
	if err != nil {
		return nil, nil, err
	}
	fail := func(err error) ([]float32, *gpu.Arena, error) {
		a.Free()
		return nil, nil, err
	}
	cos, sin := vis.angles(img.VitH, img.VitW)
	q := &visionQueue{a: a}
	q.b = a.Begin()
	q.b.Write(lay.vecs, g.vecs)
	q.b.Write(lay.cos, cos)
	q.b.Write(lay.sin, sin)
	q.b.Write(lay.patches, img.Patches)
	q.matmul(g.patch, d, 3*p*p, n, lay.x, d, lay.patches, 3*p*p, lay.vecs+g.off.patchB)
	hd := d / v.Heads
	scale := float32(1 / math.Sqrt(float64(hd)))
	vec := func(off int) int { return lay.vecs + off }
	mm := func(m *gpu.Matrix, rows, cols, out, outStride, in, inStride, bias int) {
		q.matmul(m, rows, cols, n, out, outStride, in, inStride, bias)
	}
	for _, gb := range g.blocks {
		q.add(int64(n*d), func(b *gpu.Batch) { b.DSNorm(lay.x, lay.h, vec(gb.norm1), d, n, d, 1e-6) })
		mm(gb.qkv, 3*d, d, lay.qkv, 3*d, lay.h, d, vec(gb.qkvB))
		q.add(int64(n*d), func(b *gpu.Batch) { b.VRope(n, v.Heads, hd, lay.qkv, d, lay.cos, lay.sin) })
		// The attention's queries in parts, every part over all keys.
		per := max(vAttnQueries, int(submitWork/int64(3*n*d))/vAttnQueries*vAttnQueries)
		for q0 := 0; q0 < n; q0 += per {
			q1 := min(n, q0+per)
			q.add(int64(3*(q1-q0)*n*d), func(b *gpu.Batch) { b.VAttention(n, v.Heads, hd, lay.qkv, d, lay.h, d, q0, q1, scale) })
		}
		mm(gb.o, d, d, lay.y, d, lay.h, d, vec(gb.oB))
		q.add(int64(n*d), func(b *gpu.Batch) { b.VAdd(lay.x, lay.y, n*d) })
		q.add(int64(n*d), func(b *gpu.Batch) { b.DSNorm(lay.x, lay.h, vec(gb.norm2), d, n, d, 1e-6) })
		mm(gb.w1, 2*v.Inter, d, lay.gu, 2*v.Inter, lay.h, d, -1)
		q.add(int64(n*v.Inter), func(b *gpu.Batch) { b.VSiluMul(n, v.Inter, lay.gu, lay.a) })
		mm(gb.w2, d, v.Inter, lay.y, d, lay.a, v.Inter, -1)
		q.add(int64(n*d), func(b *gpu.Batch) { b.VAdd(lay.x, lay.y, n*d) })
	}
	x := make([]float32, n*d)
	q.add(int64(n*d), func(b *gpu.Batch) {
		b.DSNorm(lay.x, lay.x, vec(g.off.norm), d, n, d, 1e-6)
		b.Read(lay.x, x)
	})
	if err := q.submit(); err != nil {
		q.b.Submit()
		return fail(err)
	}
	q.b.Submit()
	if !keep {
		a.Free()
		return x, nil, nil
	}
	return x, a, nil
}
