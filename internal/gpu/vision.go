package gpu

import "fmt"

// Kernels of DeepSeek-V4.1's image encoder (package dsv41, vision.go):
// products of many rows by bf16 matrices, full attention over an image's
// patches, a rotary embedding of head halves, and bf16 element-wise steps.
// Each is the CPU's arithmetic: the same sums in the same order, every
// product rounded, NaNs canonical before every rounding to bf16 (devices
// make NaNs of different bits).

// Tiles of the product kernel: a workgroup of 32 lanes × vmLy rows of
// threads computes vmX rows of the input by vmW rows of the matrix; the
// threads of row ly take inputs vmX/vmLy·ly.. of it. Each 32 columns of
// both tiles are read into shared memory once.
const (
	vmX  = 32
	vmW  = 8
	vmLy = 8
	vmR  = vmX / vmLy // inputs per thread
	vmK  = 64         // columns per tile
)

// vAttnTile is how many keys and values the attention kernel holds in
// shared memory at once; vAttnGroup its queries per workgroup.
const (
	vAttnTile  = 16
	vAttnGroup = 64
)

// vision kernels' kinds, after the DeepSeek block's.
func visionKernel(k kernelKind, modes floatModes) []uint32 {
	switch k {
	case kindVMatMul:
		return vMatMulKernel(modes)
	case kindVAttn16:
		return vAttnKernel(modes, 16)
	case kindVAttn32:
		return vAttnKernel(modes, 32)
	case kindVAttn64:
		return vAttnKernel(modes, 64)
	case kindVRope:
		return vRopeKernel(modes)
	case kindVAdd:
		return vAddKernel(modes)
	case kindVSiluMul:
		return vSiluMulKernel(modes)
	}
	panic("gpu: unknown kernel")
}

// bf16c rounds to bf16 after making a NaN canonical.
func (b *builder) bf16c(v id) id { return b.roundBF16(b.canonical(v)) }

// vMatMulKernel: out[i·outStride + row0 + j] = bf16(DotBF16(row j, x_i)
// (+ bias[row0 + j])) for the part's rows j and the n inputs x_i (inStride
// apart). Lane L sums the elements L, L+32, ... of a row in order, as
// num.DotBF16. Push rows, cols, -, outOff (row0 added), inOff, n, biasOff
// (0xffffffff: none, else row0 added), outStride, inStride.
func vMatMulKernel(modes floatModes) []uint32 {
	k := newKernel(modes, laneCount, vmLy)
	w := k.storageBuffer(0, true)
	xt := k.workgroupFloats(vmX * vmK)
	wt := k.workgroupFloats(vmW * vmK)
	const round = 8 // outputs per thread reduced at once
	// The reduction reuses the tiles' memory.
	sums32, sums8 := xt, wt
	lid, wid := k.ids()
	lane, ly := k.component(lid, 0), k.component(lid, 1)
	tid := k.iadd(k.imul(ly, k.u32(laneCount)), lane)
	rows, cols, outOff, inOff, n := k.p(0), k.p(1), k.p(3), k.p(4), k.p(5)
	biasOff, outStride, inStride := k.p(6), k.p(7), k.p(8)
	w0 := k.imul(k.component(wid, 0), k.u32(vmW))
	x0 := k.imul(k.component(wid, 1), k.u32(vmX))
	var acc [vmR][vmW]id
	for r := range acc {
		for c := range acc[r] {
			acc[r][c] = k.local(k.tF32)
			k.store(acc[r][c], k.f32(0))
		}
	}
	k.forRange(k.u32(0), cols, k.u32(vmK), func(k0 id) {
		k.barrier()
		// The tiles' vmK columns from k0, by all threads; past the input or
		// the matrix their last row is read (and not used).
		k.forRange(tid, k.u32(vmX*vmK), k.u32(laneCount*vmLy), func(i id) {
			r, c := k.floorDiv(i, k.u32(vmK)), k.umod(i, k.u32(vmK))
			j := k.umin(k.iadd(k0, c), k.isubU(cols, 1))
			xr := k.umin(k.iadd(x0, r), k.isubU(n, 1))
			k.store(k.shared(xt, i), k.get(k.iadd(k.iadd(inOff, k.imul(xr, inStride)), j)))
		})
		k.forRange(tid, k.u32(vmW*vmK), k.u32(laneCount*vmLy), func(i id) {
			r, c := k.floorDiv(i, k.u32(vmK)), k.umod(i, k.u32(vmK))
			j := k.umin(k.iadd(k0, c), k.isubU(cols, 1))
			wr := k.umin(k.iadd(w0, r), k.isubU(rows, 1))
			e := k.iadd(k.imul(wr, cols), j)
			word := k.word(w, k.shr(e, k.u32(1)))
			odd := k.ieq(k.and(e, k.u32(1)), k.u32(1))
			k.store(k.shared(wt, i), k.toF(k.sel(odd, k.and(word, k.u32(0xffff0000)), k.shl(word, k.u32(16)))))
		})
		k.barrier()
		// Lane L's columns, if the row has them: absent columns add nothing.
		for step := range uint32(vmK / laneCount) {
			col := k.iadd(k.u32(step*laneCount), lane)
			k.ifThen(k.ult(k.iadd(k0, col), cols), func() {
				var wv [vmW]id
				for c := range wv {
					wv[c] = k.load(k.tF32, k.shared(wt, k.iadd(k.u32(uint32(c*vmK)), col)))
				}
				for r := range acc {
					xi := k.iadd(k.imul(k.iadd(k.imul(ly, k.u32(vmR)), k.u32(uint32(r))), k.u32(vmK)), col)
					xv := k.load(k.tF32, k.shared(xt, xi))
					for c := range wv {
						k.store(acc[r][c], k.fadd(k.load(k.tF32, acc[r][c]), k.fmul(xv, wv[c])))
					}
				}
			})
		}
	})
	// The lanes' sums of each output combine in the fixed tree, round
	// outputs per thread at a time.
	outs := make([]id, 0, vmR*vmW)
	for r := range acc {
		for c := range acc[r] {
			outs = append(outs, acc[r][c])
		}
	}
	for o0 := 0; o0 < len(outs); o0 += round {
		k.barrier()
		for o := range round {
			k.store(k.shared(sums32, k.iadd(k.imul(k.iadd(k.imul(ly, k.u32(round)), k.u32(uint32(o))), k.u32(laneCount)), lane)), k.load(k.tF32, outs[o0+o]))
		}
		k.barrier()
		k.ifThen(k.ult(lane, k.u32(8)), func() {
			for o := range uint32(round) {
				base := k.imul(k.iadd(k.imul(ly, k.u32(round)), k.u32(o)), k.u32(laneCount))
				at := func(off uint32) id { return k.load(k.tF32, k.shared(sums32, k.iadd(base, k.iadd(lane, k.u32(off))))) }
				e := k.fadd(k.fadd(at(0), at(8)), k.fadd(at(16), at(24)))
				k.store(k.shared(sums8, k.iadd(k.imul(k.iadd(k.imul(ly, k.u32(round)), k.u32(o)), k.u32(8)), lane)), e)
			}
		})
		k.barrier()
		k.ifThen(k.ult(lane, k.u32(round)), func() {
			o := k.iadd(k.u32(uint32(o0)), lane) // the thread's output r·vmW + c
			xr := k.iadd(x0, k.iadd(k.imul(ly, k.u32(vmR)), k.floorDiv(o, k.u32(vmW))))
			wr := k.iadd(w0, k.umod(o, k.u32(vmW)))
			k.ifThen(k.land(k.ult(xr, n), k.ult(wr, rows)), func() {
				var e [8]id
				base := k.imul(k.iadd(k.imul(ly, k.u32(round)), lane), k.u32(8))
				for i := range e {
					e[i] = k.load(k.tF32, k.shared(sums8, k.iadd(base, k.u32(uint32(i)))))
				}
				v := k.local(k.tF32)
				k.store(v, k.combine8(e))
				k.ifThen(k.ine(biasOff, k.u32(0xffffffff)), func() {
					k.store(v, k.fadd(k.load(k.tF32, v), k.get(k.iadd(biasOff, wr))))
				})
				k.put(k.iadd(k.iadd(outOff, k.imul(xr, outStride)), wr), k.bf16c(k.load(k.tF32, v)))
			})
		})
	}
	return k.finish()
}

// umin is the smaller of two unsigned integers.
func (b *builder) umin(x, y id) id { return b.sel(b.ult(x, y), x, y) }

// isubU is x − c for a constant c.
func (b *builder) isubU(x id, c uint32) id { return b.op(opISub, b.tU32, x, b.u32(c)) }

// dot32Const is num.Dot32 of hd (a constant) values a(i)·b(i): lane l sums
// the products l, l+32, ... from +0 in order, the lanes combine in the
// fixed tree.
func (b *builder) dot32Const(hd int, a, c func(i int) id) id {
	var s [32]id
	for l := range s {
		s[l] = b.f32(0)
		for i := l; i < hd; i += 32 {
			s[l] = b.fadd(s[l], b.fmul(a(i), c(i)))
		}
	}
	var e [8]id
	for j := range e {
		e[j] = b.fadd(b.fadd(s[j], s[j+8]), b.fadd(s[j+16], s[j+24]))
	}
	return b.combine8(e)
}

// vAttnKernel is Vision.attention for heads of hd values: a thread per
// query and head, over all n keys twice (their scores' maximum, then the
// softmax and the values' sum, in order of keys). Push n, qkvOff,
// rowStride, d, outOff, outStride, scale (float bits), q0, q1: the
// queries q0..q1, in workgroups of vAttnGroup from q0.
func vAttnKernel(modes floatModes, hd int) []uint32 {
	k := newKernel(modes, vAttnGroup, 1)
	ks := k.workgroupFloats(uint32(vAttnTile * hd))
	vs := k.workgroupFloats(uint32(vAttnTile * hd))
	lid, wid := k.ids()
	t := k.component(lid, 0)
	q := k.iadd(k.p(7), k.iadd(k.imul(k.component(wid, 0), k.u32(vAttnGroup)), t))
	h := k.component(wid, 1)
	n, qkvOff, rowStride, d, outOff, outStride, scale := k.p(0), k.p(1), k.p(2), k.p(3), k.p(4), k.p(5), k.pf(6)
	qi := k.umin(q, k.isubU(n, 1))
	head := k.imul(h, k.u32(uint32(hd)))
	qv := make([]id, hd)
	for e := range qv {
		qv[e] = k.get(k.iadd(k.iadd(qkvOff, k.imul(qi, rowStride)), k.iadd(head, k.u32(uint32(e)))))
	}
	// tiles runs body over the keys in tiles, held in ks (and vs if values).
	tiles := func(values bool, body func(j, kj id)) {
		k.forRange(k.u32(0), n, k.u32(vAttnTile), func(t0 id) {
			k.barrier()
			k.forRange(t, k.u32(uint32(vAttnTile*hd)), k.u32(vAttnGroup), func(i id) {
				j := k.umin(k.iadd(t0, k.floorDiv(i, k.u32(uint32(hd)))), k.isubU(n, 1))
				c := k.umod(i, k.u32(uint32(hd)))
				row := k.iadd(k.iadd(qkvOff, k.imul(j, rowStride)), k.iadd(head, c))
				k.store(k.shared(ks, i), k.get(k.iadd(row, d)))
				if values {
					k.store(k.shared(vs, i), k.get(k.iadd(row, k.imul(d, k.u32(2)))))
				}
			})
			k.barrier()
			k.forRange(k.u32(0), k.umin(k.u32(vAttnTile), k.op(opISub, k.tU32, n, t0)), k.u32(1), func(kj id) {
				body(k.iadd(t0, kj), kj)
			})
		})
	}
	score := func(kj id) id {
		base := k.imul(kj, k.u32(uint32(hd)))
		return k.fmul(k.dot32Const(hd, func(i int) id { return qv[i] }, func(i int) id {
			return k.load(k.tF32, k.shared(ks, k.iadd(base, k.u32(uint32(i)))))
		}), scale)
	}
	m := k.local(k.tF32)
	k.store(m, k.toF(k.u32(0xff800000)))
	tiles(false, func(_, kj id) { k.store(m, k.gomax(k.load(k.tF32, m), score(kj))) })
	mx := k.load(k.tF32, m)
	sum := k.local(k.tF32)
	k.store(sum, k.f32(0))
	acc := make([]id, hd)
	for c := range acc {
		acc[c] = k.local(k.tF32)
		k.store(acc[c], k.f32(0))
	}
	tiles(true, func(_, kj id) {
		e := k.exp32(k.fsub(score(kj), mx))
		k.store(sum, k.fadd(k.load(k.tF32, sum), e))
		pe := k.bf16c(e) // multiplied with the values in bf16
		base := k.imul(kj, k.u32(uint32(hd)))
		for c := range acc {
			v := k.load(k.tF32, k.shared(vs, k.iadd(base, k.u32(uint32(c)))))
			k.store(acc[c], k.fadd(k.load(k.tF32, acc[c]), k.fmul(pe, v)))
		}
	})
	inv := k.fdiv(k.f32(1), k.load(k.tF32, sum))
	k.ifThen(k.ult(q, k.umin(n, k.p(8))), func() {
		o := k.iadd(k.iadd(outOff, k.imul(q, outStride)), head)
		for c := range acc {
			k.put(k.iadd(o, k.u32(uint32(c))), k.bf16c(k.fmul(k.load(k.tF32, acc[c]), inv)))
		}
	})
	return k.finish()
}

// vRopeKernel is Vision.rotate: the queries and keys of n rows of qkv
// (3·d apart), per head the halves x1, x2 to bf16(x1·cos − x2·sin),
// bf16(x2·cos + x1·sin), cos and sin per row and pair. Push n, heads, hd,
// qkvOff, d, cosOff, sinOff.
func vRopeKernel(modes floatModes) []uint32 {
	k := newKernel(modes, flatGroup, 1)
	idx := k.flat()
	n, heads, hd, qkv, d, cosOff, sinOff := k.p(0), k.p(1), k.p(2), k.p(3), k.p(4), k.p(5), k.p(6)
	half := k.shr(hd, k.u32(1))
	k.ifThen(k.ult(idx, k.imul(k.imul(n, heads), k.imul(k.u32(2), half))), func() {
		j := k.umod(idx, half)
		rest := k.floorDiv(idx, half)
		h := k.umod(rest, heads)
		rest = k.floorDiv(rest, heads)
		part, i := k.umod(rest, k.u32(2)), k.floorDiv(rest, k.u32(2))
		base := k.iadd(k.iadd(qkv, k.imul(i, k.imul(d, k.u32(3)))), k.iadd(k.imul(part, d), k.imul(h, hd)))
		x1, x2 := k.get(k.iadd(base, j)), k.get(k.iadd(base, k.iadd(half, j)))
		cs := k.iadd(k.imul(i, half), j)
		c, s := k.get(k.iadd(cosOff, cs)), k.get(k.iadd(sinOff, cs))
		k.put(k.iadd(base, j), k.bf16c(k.fsub(k.fmul(x1, c), k.fmul(x2, s))))
		k.put(k.iadd(base, k.iadd(half, j)), k.bf16c(k.fadd(k.fmul(x2, c), k.fmul(x1, s))))
	})
	return k.finish()
}

// vAddKernel: x = bf16(x + y), n values. Push n, x, y.
func vAddKernel(modes floatModes) []uint32 {
	k := newKernel(modes, flatGroup, 1)
	i := k.flat()
	n, x, y := k.p(0), k.p(1), k.p(2)
	k.ifThen(k.ult(i, n), func() {
		k.put(k.iadd(x, i), k.bf16c(k.fadd(k.get(k.iadd(x, i)), k.get(k.iadd(y, i)))))
	})
	return k.finish()
}

// vSiluMulKernel: a[r·inter + j] = bf16(bf16(silu(g)) · u), g and u the
// halves of row r of gu (2·inter wide), rows rows. Push rows, inter, gu, a.
func vSiluMulKernel(modes floatModes) []uint32 {
	k := newKernel(modes, flatGroup, 1)
	i := k.flat()
	rows, inter, gu, a := k.p(0), k.p(1), k.p(2), k.p(3)
	k.ifThen(k.ult(i, k.imul(rows, inter)), func() {
		r, j := k.floorDiv(i, inter), k.umod(i, inter)
		g := k.iadd(gu, k.iadd(k.imul(r, k.imul(inter, k.u32(2))), j))
		v := k.fmul(k.bf16c(k.silu(k.get(g))), k.get(k.iadd(g, inter)))
		k.put(k.iadd(a, i), k.bf16c(v))
	})
	return k.finish()
}

// VMatMul sets n rows of out (outStride apart) to bf16(x_i·mᵀ + bias) for
// the rows x_i at in (inStride apart); bias is an offset, -1 for none.
func (b *Batch) VMatMul(m *Matrix, out, outStride, in, inStride, n, bias int) {
	if m.kind != kindBF16 {
		b.err = fmt.Errorf("gpu: VMatMul of a %d matrix", m.kind)
		return
	}
	for _, pt := range m.parts {
		bo := uint32(0xffffffff)
		if bias >= 0 {
			bo = uint32(bias + pt.row0)
		}
		for x0 := 0; x0 < n; x0 += 65535 * vmX {
			nx := min(n-x0, 65535*vmX)
			b.dispatch(kindVMatMul, pt.set, groups(pt.rows, vmW), groups(nx, vmX), uint32(pt.rows), uint32(m.cols), 0,
				uint32(out+x0*outStride+pt.row0), uint32(in+x0*inStride), uint32(nx), bo, uint32(outStride), uint32(inStride))
		}
	}
}

// VAttention is full attention of n rows of qkv (q, k, v of d each, heads
// of hd ∈ {16, 32, 64}) into out (outStride apart), scores scaled by
// scale. Queries from q0 to q1 only: a long attention goes in parts.
func (b *Batch) VAttention(n, heads, hd, qkv, d, out, outStride, q0, q1 int, scale float32) {
	kinds := map[int]kernelKind{16: kindVAttn16, 32: kindVAttn32, 64: kindVAttn64}
	kind, ok := kinds[hd]
	if !ok {
		b.err = fmt.Errorf("gpu: attention heads of %d", hd)
		return
	}
	// The kernel's queries are rows q0.. of a view starting there; its
	// keys must be all n rows: the view's query offset moves the output
	// only, so queries are run in parts over the same keys.
	for s := q0; s < q1; s += 65535 * vAttnGroup {
		e := min(q1, s+65535*vAttnGroup)
		b.dispatch(kind, 0, groups(e-s, vAttnGroup), heads, uint32(n), uint32(qkv), uint32(3*d), uint32(d), uint32(out), uint32(outStride), f32bits(scale), uint32(s), uint32(e))
	}
}

// VRope rotates the queries and keys of n rows of qkv.
func (b *Batch) VRope(n, heads, hd, qkv, d, cos, sin int) {
	b.dispatch(kindVRope, 0, groups(n*heads*hd, flatGroup), 1, uint32(n), uint32(heads), uint32(hd), uint32(qkv), uint32(d), uint32(cos), uint32(sin))
}

// VAdd sets n values at x to bf16(x + y).
func (b *Batch) VAdd(x, y, n int) {
	b.dispatch(kindVAdd, 0, groups(n, flatGroup), 1, uint32(n), uint32(x), uint32(y))
}

// VSiluMul sets rows·inter values at a from the rows of gu.
func (b *Batch) VSiluMul(rows, inter, gu, a int) {
	b.dispatch(kindVSiluMul, 0, groups(rows*inter, flatGroup), 1, uint32(rows), uint32(inter), uint32(gu), uint32(a))
}
