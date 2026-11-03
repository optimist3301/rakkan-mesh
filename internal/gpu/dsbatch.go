package gpu

import (
	"errors"
	"math"
)

// Batch operations of a DeepSeek-V4.1 block (dsops.go). Offsets are words
// of the arena.

func b2w(v bool) uint32 {
	if v {
		return 1
	}
	return 0
}

// WriteWords copies words (indices, flags) into the arena at off.
func (b *Batch) WriteWords(off int, data []uint32) {
	f := make([]float32, len(data))
	for i, v := range data {
		f[i] = math.Float32frombits(v)
	}
	b.Write(off, f)
}

// ReadWords copies len(dst) words at off into dst when the batch is
// submitted, through dst's own float view.
func (b *Batch) ReadWords(off int, dst []float32) { b.Read(off, dst) }

// DSNorm is dsv41's rmsNorm of segs segments of n floats (stride apart)
// at in into out, with weights at w.
func (b *Batch) DSNorm(in, out, w, n, segs, stride int, eps float32) {
	b.dispatch(kindDSNorm, 0, segs, 1, uint32(n), uint32(in), uint32(out), uint32(w), uint32(segs), uint32(stride), f32bits(eps))
}

// HCSplit turns the 24 mixing logits at mix into pre, post and comb, the
// stream h of n floats giving their scale (hc = 4).
func (b *Batch) HCSplit(mix, h, n, base, scale, iters int, eps, normEps float32, pre, post, comb int) {
	b.dispatch(kindHCSplit, 0, 1, 1, uint32(mix), uint32(h), uint32(n), uint32(base), uint32(scale), uint32(iters),
		f32bits(eps), f32bits(normEps), uint32(pre), uint32(post), uint32(comb))
}

// HCPre collapses the stream h (4 copies of dim) by the mix at p into out.
func (b *Batch) HCPre(dim, h, p, out int) {
	b.dispatch(kindHCPre, 0, groups(dim, flatGroup), 1, uint32(dim), uint32(h), uint32(p), uint32(out))
}

// HCPost sets the stream h from y and the residual res.
func (b *Batch) HCPost(dim, h, y, res, post, comb int) {
	b.dispatch(kindHCPost, 0, groups(4*dim, flatGroup), 1, uint32(dim), uint32(h), uint32(y), uint32(res), uint32(post), uint32(comb))
}

// RopeDS rotates segs segments of 2·pairs floats at x (stride apart) by
// the cos and sin at cs, inversely if inverse.
func (b *Batch) RopeDS(x, pairs, segs, stride, cs int, inverse bool) {
	b.dispatch(kindRopeDS, 0, groups(pairs*segs, flatGroup), 1, uint32(pairs), uint32(segs), uint32(x), uint32(stride), uint32(cs), b2w(inverse))
}

// ActQuantInPlace is num.ActQuantInPlace of n floats at off.
func (b *Batch) ActQuantInPlace(off, n int) {
	b.dispatch(kindActQuant, 0, groups(n/32, flatGroup), 1, uint32(n), uint32(off))
}

// FP4QuantInPlace is num.FP4QuantInPlace of n floats at off, with E4M3
// scales per 16 if e4m3, else E8M0 per 32.
func (b *Batch) FP4QuantInPlace(off, n int, e4m3 bool) {
	blk := 32
	if e4m3 {
		blk = 16
	}
	b.dispatch(kindFP4Quant, 0, groups(n/blk, flatGroup), 1, uint32(n), uint32(off), b2w(e4m3))
}

// RoundScale sets n floats at off to bf16(x·scale).
func (b *Batch) RoundScale(off, n int, scale float32) {
	b.dispatch(kindRoundScale, 0, groups(n, flatGroup), 1, uint32(n), uint32(off), f32bits(scale))
}

// Compress pools a group of ratio slots of KV and scores (hd apart) into out.
func (b *Batch) Compress(hd, ratio, kv, score, out int) {
	b.dispatch(kindCompress, 0, groups(hd, flatGroup), 1, uint32(hd), uint32(ratio), uint32(kv), uint32(score), uint32(out))
}

// IndexScores scores width keys (dim apart at keys) for the heads' queries
// at q weighted by w; cand and candLen are the candidate pool's flags, used
// if mask.
func (b *Batch) IndexScores(width, visible, q, keys, w, heads, dim, out, cand, candLen int, mask bool) {
	b.dispatch(kindIdxScores, 0, groups(width, flatGroup), 1, uint32(width), uint32(visible), uint32(q), uint32(keys), uint32(w),
		uint32(heads), uint32(dim), uint32(out), uint32(cand), uint32(candLen), b2w(mask))
}

// CandidatePool flags at keep the positions of the best k blocks of size
// among width scores; bs and sel are scratch for a score and a flag per
// block.
func (b *Batch) CandidatePool(width, visible, k, size, scores, bs, sel, keep int) {
	n := (width + size - 1) / size
	phase := func(p uint32, count int) {
		b.dispatch(kindCandPool, 0, groups(count, flatGroup), 1, uint32(width), uint32(visible), uint32(k), uint32(size),
			uint32(scores), uint32(bs), uint32(sel), uint32(keep), p)
	}
	phase(0, n)
	b.Select(n, min(k, n), bs, sel)
	phase(2, width)
}

// TopK writes the indices of the k largest of n values at vals, by rank,
// at out.
func (b *Batch) TopK(n, k, vals, out int) {
	b.dispatch(kindTopK, 0, groups(n, flatGroup), 1, uint32(n), uint32(k), uint32(vals), uint32(out))
}

// Select flags at flags the k largest of n values at vals (dsv41.topK's
// choice) in 32·n steps; SelectSorted also writes their indices ascending
// at out, -1 for those from visible on.
func (b *Batch) Select(n, k, vals, flags int) {
	b.dispatch(kindSelect, 0, 1, 1, uint32(n), uint32(k), uint32(vals), uint32(flags))
}

func (b *Batch) SelectSorted(n, k, vals, out, flags, visible int) {
	b.Select(n, k, vals, flags)
	b.dispatch(kindCompact, 0, 1, 1, uint32(n), uint32(flags), uint32(out), uint32(visible))
}

// SparseAttention attends the heads' queries at q to nWin rows listed at
// rows and nTop entries of the cache at comp selected at topk.
func (b *Batch) SparseAttention(heads, hd, q, rows, nWin, topk, nTop, comp, sink int, scale float32, acc, out int) {
	b.dispatch(kindSparseAttn, 0, heads, 1, uint32(heads), uint32(hd), uint32(q), uint32(rows), uint32(nWin), uint32(topk),
		uint32(nTop), uint32(comp), uint32(sink), f32bits(scale), uint32(acc), uint32(out))
}

// MoEScores computes the gate's scores of n experts from its logits at z.
func (b *Batch) MoEScores(n, z, bias int, sigmoid bool, score, biased int) {
	b.dispatch(kindMoEScore, 0, groups(n, flatGroup), 1, uint32(n), uint32(z), uint32(bias), b2w(sigmoid), uint32(score), uint32(biased))
}

// MoEWeights orders the k selected experts (by rank at sel) ascending at
// order, with their weights at w.
func (b *Batch) MoEWeights(k, score, sel int, norm bool, scale float32, order, w int) {
	b.dispatch(kindMoEWeights, 0, 1, 1, uint32(k), uint32(score), uint32(sel), b2w(norm), f32bits(scale), uint32(order), uint32(w))
}

// ExpertMatVecs multiplies the experts selected at ids (slots of them) of
// m, E2M1 experts of perExpert rows each, by inputs at in (inStride
// apart), slot s taking input s/div (div 0: one input for all), into out
// (outStride apart). qx and xs are scratch for the quantized inputs.
func (b *Batch) ExpertMatVecs(m *Matrix, perExpert, ids, slots, in, inStride, div, out, outStride, qx, xs int) {
	if m.kind != kindFP4 || m.rows%perExpert != 0 {
		b.err = errors.New("gpu: not a matrix of E2M1 experts")
		return
	}
	n := m.cols
	inputs := 1
	if div > 0 {
		inputs = (slots + div - 1) / div
	} else {
		div = slots
	}
	for s := range inputs {
		b.dispatch(kindActFP8, 0, groups(n/32, flatGroup), 1, uint32(n), uint32(in+s*inStride), uint32(qx+s*n), uint32(xs+s*n/32))
	}
	for _, pt := range m.parts {
		if pt.row0%perExpert != 0 || pt.rows%perExpert != 0 {
			b.err = errors.New("gpu: experts split across parts")
			return
		}
		e0 := pt.row0 / perExpert
		b.dispatch(kindFP4Experts, pt.set, groups(perExpert, 2*mxRows), slots, uint32(perExpert), uint32(n), 0, uint32(out),
			uint32(qx), uint32(xs), 0, uint32(ids), uint32(e0), uint32(e0+pt.rows/perExpert), uint32(n), uint32(n/32), uint32(outStride), uint32(div))
	}
}

// MatVecsBatch multiplies an E4M3 matrix by n inputs at in (inStride
// apart) into out (outStride apart), reading the weights once for up to
// MatBatchMax of them; qx and xs are scratch for the quantized inputs
// (n·cols and n·cols/32 words). The results are MatVecs' each.
func (b *Batch) MatVecsBatch(m *Matrix, out, outStride, in, inStride, n, qx, xs int) {
	if m.kind != kindFP8 {
		for j := range n {
			b.MatVecs([]*Matrix{m}, []int{out + j*outStride}, in+j*inStride)
		}
		return
	}
	c := m.cols
	for j := range n {
		b.dispatch(kindActFP8, 0, groups(c/32, flatGroup), 1, uint32(c), uint32(in+j*inStride), uint32(qx+j*c), uint32(xs+j*c/32))
	}
	raw := b2w(m.unrounded)
	for j0 := 0; j0 < n; j0 += mxBatchMax {
		nj := min(mxBatchMax, n-j0)
		for _, pt := range m.parts {
			g := groups(pt.rows, 2*mxRows)
			gx := min(g, 65535)
			gy := groups(g, gx)
			b.dispatch(kindFP8Batch, pt.set, gx, gy, uint32(pt.rows), uint32(c), uint32(gx), uint32(out+j0*outStride+pt.row0),
				uint32(qx+j0*c), uint32(xs+j0*c/32), raw, uint32(nj), uint32(c), uint32(c/32), uint32(outStride))
		}
	}
}

// MatBatchMax is how many inputs MatVecsBatch takes in one reading of the
// weights.
const MatBatchMax = mxBatchMax

// SiluMulDS is dsv41's expert activation on slots of n floats.
func (b *Batch) SiluMulDS(n, g, u int, limit float32, weighted bool, w, slots, stride int) {
	b.dispatch(kindSiluMulDS, 0, groups(n*slots, flatGroup), 1, uint32(n), uint32(g), uint32(u), f32bits(limit), b2w(weighted), uint32(w), uint32(slots), uint32(stride))
}

// MoESum adds k experts' outputs (stride apart) and the shared one's into y.
func (b *Batch) MoESum(dim, outs, k, stride, shared, y int) {
	b.dispatch(kindMoESum, 0, groups(dim, flatGroup), 1, uint32(dim), uint32(outs), uint32(k), uint32(stride), uint32(shared), uint32(y))
}

// EngramGate adds the Engram value to the stream h (4 copies of dim),
// gated per copy, from the projection at kv and the weights at qk.
func (b *Batch) EngramGate(dim, h, kv, qk int, eps, scale float32) {
	b.dispatch(kindEngramGate, 0, 4, 1, uint32(dim), uint32(h), uint32(kv), uint32(qk), f32bits(eps), f32bits(scale))
}
