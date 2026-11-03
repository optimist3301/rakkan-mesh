package gpu

// Kernels of DeepSeek-V4.1's block-scaled products (package num, mx.go):
// the input quantized to E4M3 with a power-of-two scale per 32 values
// (num.QuantizeActFP8), and products with E4M3 weights scaled per 32×32
// block or packed E2M1 weights scaled per 32 of a row (num.MatVecFP8,
// num.MatVecFP4).
//
// The CPU sums a block's 32 products exactly, in float64 for E4M3 weights.
// Many GPUs have no float64, nor 64-bit integers (Apple's through
// MoltenVK, Intel's Arc, mobile ones), so the sum is made of float32 sums
// that are exact by construction. The product of two E4M3 values (4
// significant bits each) is exact in float32, a multiple of 2^-18 below
// 2^18. Products below 2^-6, those from 2^-6 to 2^6 and those above go to
// three sums: each is a multiple of its smallest product's unit (2^-18,
// 2^-13, 2^-1, as a product of mantissas is below 2^8) and, over 32
// products, below 2^24 such units, so every partial sum is exact. The
// three are then integers in their units, combined into a 42-bit integer
// and rounded to float32 once, ties to even, with integer operations: the
// CPU's float32(sum). E2M1 weights work the same with two sums (below 4,
// in units of 2^-10, and above, in units of 2^-3). As every intermediate
// value is exact and normal, the sum does not depend on the device's
// rounding nor on whether it keeps denormals. What follows is the CPU's
// float32 arithmetic in its order: times the input's scale, times the
// weight's, blocks added in order, the sum rounded to bfloat16. Those
// terms are denormal only with weight scales below about 2^-100, which
// models do not have; there a device must keep denormals to agree.

const (
	opShiftRightArithmetic = 195
	opINotEqual            = 171
)

func (b *builder) sar(x, n id) id { return b.op(opShiftRightArithmetic, b.tI32, x, n) }
func (b *builder) ine(x, y id) id { return b.op(opINotEqual, b.tBool, x, y) }
func (b *builder) ugt(x, y id) id { return b.ult(y, x) }
func (b *builder) ineg(x id) id   { return b.isubS(b.i32(0), x) }

// msb is the index of the highest set bit of x > 0.
func (b *builder) msb(x id) id {
	r := b.u32(0)
	for _, s := range []uint32{16, 8, 4, 2, 1} {
		big := b.ugt(x, b.u32(1<<s-1))
		x = b.sel(big, b.shr(x, b.u32(s)), x)
		r = b.sel(big, b.iadd(r, b.u32(s)), r)
	}
	return r
}

// intToF rounds hi·2^16 + lo (hi, lo signed, the sum below 2^42 in
// magnitude) to float32, ties to even, times 2^-k: exact, for the
// results stay normal.
func (b *builder) intToF(hi, lo id, k int32) id {
	h := b.iaddS(hi, b.sar(lo, b.u32(16)))
	l := b.and(b.toU(lo), b.u32(0xffff))
	neg := b.slt(h, b.i32(0))
	lz := b.ieq(l, b.u32(0))
	// The magnitude, A·2^16 + B with B < 2^16.
	negA := b.sel(lz, b.toU(b.ineg(h)), b.toU(b.isubS(b.ineg(h), b.i32(1))))
	a := b.sel(neg, negA, b.toU(h))
	bb := b.sel(neg, b.sel(lz, b.u32(0), b.op(opISub, b.tU32, b.u32(1<<16), l)), l)
	whi := b.shr(a, b.u32(16))
	wlo := b.or(b.shl(a, b.u32(16)), bb)
	hasHi := b.ine(whi, b.u32(0))
	n := b.sel(hasHi, b.iadd(b.msb(b.sel(hasHi, whi, b.u32(1))), b.u32(32)), b.msb(b.sel(b.ieq(wlo, b.u32(0)), b.u32(1), wlo)))
	// Exact when n < 24: the value shifted up to 24 bits.
	small := b.ult(n, b.u32(24))
	up := b.shl(wlo, b.op(opISub, b.tU32, b.u32(23), b.sel(small, n, b.u32(23))))
	// Else d = n-23 bits are dropped, rounding to even.
	d := b.op(opISub, b.tU32, b.sel(small, b.u32(24), n), b.u32(23))
	mant := b.or(b.shl(whi, b.op(opISub, b.tU32, b.u32(32), d)), b.shr(wlo, d))
	rem := b.and(wlo, b.op(opISub, b.tU32, b.shl(b.u32(1), d), b.u32(1)))
	half := b.shl(b.u32(1), b.op(opISub, b.tU32, d, b.u32(1)))
	odd := b.ieq(b.and(mant, b.u32(1)), b.u32(1))
	roundUp := b.lor(b.ugt(rem, half), b.land(b.ieq(rem, half), odd))
	mant = b.iadd(mant, b.sel(roundUp, b.u32(1), b.u32(0)))
	carry := b.ieq(mant, b.u32(1<<24))
	n = b.iadd(n, b.sel(b.land(carry, b.ugt(n, b.u32(23))), b.u32(1), b.u32(0)))
	mant = b.sel(small, up, b.sel(carry, b.u32(1<<23), mant))
	bits := b.or(b.shl(b.op(opISub, b.tU32, b.iadd(n, b.u32(127)), b.u32(uint32(k))), b.u32(23)), b.and(mant, b.u32(0x7fffff)))
	bits = b.or(bits, b.sel(neg, b.u32(0x80000000), b.u32(0)))
	zero := b.land(b.ieq(whi, b.u32(0)), b.ieq(wlo, b.u32(0)))
	return b.toF(b.sel(zero, b.u32(0), bits))
}

// roundBF16 is num.RoundBF16.
func (b *builder) roundBF16(v id) id {
	bits := b.toU(v)
	r := b.and(b.iadd(b.iadd(bits, b.u32(0x7fff)), b.and(b.shr(bits, b.u32(16)), b.u32(1))), b.u32(0xffff0000))
	nan := b.and(b.or(bits, b.u32(0x00400000)), b.u32(0xffff0000))
	return b.toF(b.sel(b.isNaN(v), nan, r))
}

// e8m0 is num.DecodeE8M0 of a scale byte.
func (b *builder) e8m0(c id) id {
	bits := b.sel(b.ieq(c, b.u32(0)), b.u32(1<<22), b.shl(c, b.u32(23)))
	return b.toF(b.sel(b.ieq(c, b.u32(255)), b.u32(0x7fc00000), bits))
}

// e4m3F is the value of an E4M3 code (not NaN), from its bits: a
// subnormal one m·2^-9 as (8+m)·2^-9 - 2^-6, exactly, so no denormal arises.
func (b *builder) e4m3F(c id) id {
	sub := b.ieq(b.and(c, b.u32(0x78)), b.u32(0))
	mag := b.or(b.and(c, b.u32(0x7f)), b.sel(sub, b.u32(8), b.u32(0)))
	v := b.toF(b.iadd(b.shl(mag, b.u32(20)), b.u32(120<<23)))
	v = b.fsub(v, b.selF(sub, b.f32(1.0/64), b.f32(0)))
	return b.toF(b.or(b.toU(v), b.shl(b.and(c, b.u32(0x80)), b.u32(24))))
}

// e2m1F is the value of an E2M1 code: 0 and 0.5 as 1 and 1.5 less 1.
func (b *builder) e2m1F(c id) id {
	sub := b.ult(b.and(c, b.u32(6)), b.u32(2))
	mag := b.or(b.and(c, b.u32(7)), b.sel(sub, b.u32(2), b.u32(0)))
	v := b.toF(b.iadd(b.shl(mag, b.u32(22)), b.u32(126<<23)))
	v = b.fsub(v, b.selF(sub, b.f32(1), b.f32(0)))
	return b.toF(b.or(b.toU(v), b.shl(b.and(c, b.u32(8)), b.u32(28))))
}

// fToI converts an integer-valued float32 below 2^31 exactly.
func (b *builder) fToI(v id) id { return b.op(opConvertFToS, b.tI32, v) }

// encodeE4M3 is num.EncodeE4M3 of v already within ±448, or NaN.
func (b *builder) encodeE4M3(v id) id {
	bits := b.toU(v)
	sign := b.shr(b.and(bits, b.u32(0x80000000)), b.u32(24))
	a := b.toF(b.and(bits, b.u32(0x7fffffff)))
	// Below 2^-6: multiples of 2^-9 (8 is the smallest normal's code).
	subCode := b.op(opConvertFToS, b.tI32, b.roundEven(b.fmul(a, b.f32(512))))
	// Else the exponent, and the mantissa scaled to [8, 16), rounded.
	be := b.and(b.shr(b.toU(a), b.u32(23)), b.u32(0xff))
	scaled := b.toF(b.or(b.and(b.toU(a), b.u32(0x7fffff)), b.u32(130<<23)))
	n := b.toU(b.op(opConvertFToS, b.tI32, b.roundEven(scaled)))
	carry := b.ieq(n, b.u32(16))
	be = b.iadd(be, b.sel(carry, b.u32(1), b.u32(0)))
	n = b.sel(carry, b.u32(8), n)
	// (e+7)<<3 | (n-8), with e = be-127.
	code := b.op(opISub, b.tU32, b.iadd(b.shl(b.op(opISub, b.tU32, be, b.u32(120)), b.u32(3)), n), b.u32(8))
	code = b.sel(b.ugt(code, b.u32(0x7e)), b.u32(0x7e), code)
	code = b.sel(b.flt(a, b.f32(1.0/64)), b.toU(subCode), code)
	code = b.or(code, sign)
	return b.sel(b.isNaN(v), b.u32(0x7f), code)
}

// actFP8Kernel quantizes n floats as num.QuantizeActFP8, one block per
// thread: the E4M3 values, unscaled, n words at qxOff, the block scales at
// xsOff. Push n, inOff, qxOff, xsOff.
func actFP8Kernel(modes floatModes) []uint32 {
	k := newKernel(modes, flatGroup, 1)
	blk := k.flat()
	n, inOff, qxOff, xsOff := k.p(0), k.p(1), k.p(2), k.p(3)
	k.ifThen(k.ult(blk, k.shr(n, k.u32(5))), func() {
		base := k.iadd(inOff, k.imul(blk, k.u32(32)))
		vals := make([]id, 32)
		amax := k.f32(0)
		for j := range vals {
			vals[j] = k.get(k.iadd(base, k.u32(uint32(j))))
			amax = k.gomax(amax, k.toF(k.and(k.toU(vals[j]), k.u32(0x7fffffff))))
		}
		t := k.fmul(k.gomax(amax, k.f32(float32(1e-4))), k.f32(float32(1.0/448)))
		// pow2Ceil: the exponent, plus one if any mantissa bit is set.
		tb := k.toU(t)
		e := k.iadd(k.and(k.shr(tb, k.u32(23)), k.u32(0xff)), k.sel(k.ine(k.and(tb, k.u32(0x7fffff)), k.u32(0)), k.u32(1), k.u32(0)))
		s := k.toF(k.shl(e, k.u32(23)))
		inv := k.toF(k.shl(k.op(opISub, k.tU32, k.u32(254), e), k.u32(23))) // 1/s, exact
		k.put(k.at(xsOff, blk), s)
		for j := range uint32(32) {
			x := k.fmul(vals[j], inv) // v/s: exact scaling, rounded once
			x = k.selF(k.fgt(x, k.f32(448)), k.f32(448), k.selF(k.flt(x, k.f32(-448)), k.f32(-448), x))
			k.put(k.iadd(qxOff, k.iadd(k.imul(blk, k.u32(32)), k.u32(j))), k.e4m3F(k.encodeE4M3(x)))
		}
	})
	return k.finish()
}

// mxRows is how many rows a thread of mxKernel computes: the input's
// values it loads serve them all.
const mxRows = 4

// mxKernel multiplies E4M3 (kindFP8) or E2M1 (kindFP4) weights by an
// input quantized by actFP8Kernel. A workgroup has two groups of 32 lanes,
// each lane computing a block for mxRows rows: blocks L, L+32, ... of
// rows 4g to 4g+3; per chunk of 32 blocks lane q of the group adds row
// 4g+q's terms in order. Push rows, cols, gridX, outOff, qxOff, xsOff,
// and raw: if set, the sums are not rounded to bfloat16 (for tests).
func mxKernel(kind kernelKind, modes floatModes) []uint32 { return mxKernelGen(kind, modes, false) }

// mxKernelIndexed is mxKernel on E2M1 experts stacked in one matrix part,
// rows apart, for slots selected experts: workgroup row y is a slot, whose
// expert is at word ids+y; experts outside the part's [e0, e1) are left to
// another part's dispatch. Each slot has its own input (qx, xs strides)
// and output (out stride). Push rows (per expert), cols, gridX, out, qx,
// xs, raw, ids, e0, e1, qxStride, xsStride, outStride.
func mxKernelIndexed(modes floatModes) []uint32 { return mxKernelGen(kindFP4, modes, true) }

func mxKernelGen(kind kernelKind, modes floatModes, indexed bool) []uint32 {
	k := newKernel(modes, laneCount, 2)
	w, ws := k.storageBuffer(0, true), k.storageBuffer(1, true)
	terms := k.workgroupFloats(laneCount * 2 * mxRows)
	acc, chunk := k.local(k.tF32), k.local(k.tU32)
	var term [mxRows]id
	for q := range term {
		term[q] = k.local(k.tF32)
	}
	lid, wid := k.ids()
	lane, ly := k.component(lid, 0), k.component(lid, 1)
	rows, cols, outOff, qxOff, xsOff, raw := k.p(0), k.p(1), k.p(3), k.p(4), k.p(5), k.p(6)
	group := k.iadd(k.imul(k.component(wid, 1), k.p(2)), k.component(wid, 0))
	inPart := k.ieq(k.u32(0), k.u32(0))
	rowOff := k.u32(0) // of the expert's rows in the part
	if indexed {
		slot := k.component(wid, 1)
		group = k.component(wid, 0)
		e := k.iget(k.at(k.p(7), slot))
		inPart = k.land(k.uge(e, k.p(8)), k.ult(e, k.p(9)))
		rowOff = k.imul(k.op(opISub, k.tU32, e, k.p(8)), rows)
		qxOff = k.iadd(qxOff, k.imul(slot, k.p(10)))
		xsOff = k.iadd(xsOff, k.imul(slot, k.p(11)))
		outOff = k.iadd(outOff, k.imul(slot, k.p(12)))
	}
	base := k.iadd(k.imul(group, k.u32(2*mxRows)), k.imul(ly, k.u32(mxRows)))
	nb := k.shr(cols, k.u32(5))
	chunks := k.shr(k.iadd(nb, k.u32(laneCount-1)), k.u32(5))
	// The row lane q < mxRows adds up.
	myRow := k.iadd(base, lane)
	k.store(acc, k.f32(0))
	k.store(chunk, k.u32(0))
	// The whole workgroup takes the same branch: its barriers stay uniform.
	k.ifThen(inPart, func() {
		k.loop(func() id { return k.ult(k.load(k.tU32, chunk), chunks) }, func() {
			c := k.load(k.tU32, chunk)
			blk := k.iadd(k.imul(c, k.u32(laneCount)), lane)
			for q := range term {
				k.store(term[q], k.f32(0))
			}
			k.ifThen(k.land(k.ult(base, rows), k.ult(blk, nb)), func() {
				xbase := k.iadd(qxOff, k.imul(blk, k.u32(32)))
				xs := make([]id, 32)
				for e := range xs {
					xs[e] = k.get(k.iadd(xbase, k.u32(uint32(e))))
				}
				xsc := k.get(k.at(xsOff, blk))
				for q := range uint32(mxRows) {
					row := k.iadd(base, k.u32(q))
					k.ifThen(k.ult(row, rows), func() {
						k.store(term[q], k.mxBlock(kind, w, ws, k.iadd(rowOff, row), cols, nb, blk, xs, xsc))
					})
				}
			})
			for q := range uint32(mxRows) {
				slot := k.iadd(k.imul(k.iadd(k.imul(ly, k.u32(mxRows)), k.u32(q)), k.u32(laneCount)), lane)
				k.store(k.shared(terms, slot), k.load(k.tF32, term[q]))
			}
			k.barrier()
			// Lane q adds its row's terms of the chunk's blocks that exist, in order.
			k.ifThen(k.ult(lane, k.u32(mxRows)), func() {
				rowBase := k.imul(k.iadd(k.imul(ly, k.u32(mxRows)), lane), k.u32(laneCount))
				first := k.imul(c, k.u32(laneCount))
				for j := range uint32(laneCount) {
					k.ifThen(k.ult(k.iadd(first, k.u32(j)), nb), func() {
						t := k.load(k.tF32, k.shared(terms, k.iadd(rowBase, k.u32(j))))
						k.store(acc, k.fadd(k.load(k.tF32, acc), t))
					})
				}
			})
			k.barrier()
		}, func() { k.store(chunk, k.iadd(k.load(k.tU32, chunk), k.u32(1))) })
		k.ifThen(k.land(k.ult(lane, k.u32(mxRows)), k.ult(myRow, rows)), func() {
			v := k.load(k.tF32, acc)
			k.put(k.at(outOff, myRow), k.selF(k.ieq(raw, k.u32(0)), k.roundBF16(v), v))
		})
	})
	return k.finish()
}

// mxBlock is block blk's term of a row: its exact sum rounded to
// float32, times the input's scale xsc, times the weight's.
func (k *kbuilder) mxBlock(kind kernelKind, w, ws, row, cols, nb, blk id, xv []id, xsc id) id {
	var sum id
	if kind == kindFP8 {
		b0, b1, b2 := k.f32(0), k.f32(0), k.f32(0)
		wbase := k.shr(k.iadd(k.imul(row, cols), k.imul(blk, k.u32(32))), k.u32(2))
		for i := range uint32(8) {
			ww := k.word(w, k.iadd(wbase, k.u32(i)))
			for j := range uint32(4) {
				p := k.fmul(k.e4m3F(k.and(k.shr(ww, k.u32(8*j)), k.u32(0xff))), xv[4*i+j])
				a := k.toF(k.and(k.toU(p), k.u32(0x7fffffff)))
				small, large := k.flt(a, k.f32(1.0/64)), k.lor(k.flt(k.f32(64), a), k.feq(a, k.f32(64)))
				b0 = k.fadd(b0, k.selF(small, p, k.f32(0)))
				b2 = k.fadd(b2, k.selF(large, p, k.f32(0)))
				b1 = k.fadd(b1, k.selF(k.lor(small, large), k.f32(0), p))
			}
		}
		// In units of 2^-18: i0 + i1·2^5 + i2·2^17.
		i0 := k.fToI(k.fmul(b0, k.f32(1<<18)))
		i1 := k.fToI(k.fmul(b1, k.f32(1<<13)))
		i2 := k.fToI(k.fmul(b2, k.f32(2)))
		hi := k.iaddS(k.sar(i1, k.u32(11)), k.imulS(i2, k.i32(2)))
		lo := k.iaddS(i0, k.toI(k.shl(k.and(k.toU(i1), k.u32(0x7ff)), k.u32(5))))
		sum = k.intToF(hi, lo, 18)
	} else {
		b0, b1 := k.f32(0), k.f32(0)
		wbase := k.shr(k.iadd(k.imul(row, k.shr(cols, k.u32(1))), k.imul(blk, k.u32(16))), k.u32(2))
		for i := range uint32(4) {
			ww := k.word(w, k.iadd(wbase, k.u32(i)))
			for j := range uint32(4) {
				e := 2 * (4*i + j) // the low nibble's element; the high one's is e+1
				for h := range uint32(2) {
					p := k.fmul(k.e2m1F(k.and(k.shr(ww, k.u32(8*j+4*h)), k.u32(15))), xv[e+h])
					a := k.toF(k.and(k.toU(p), k.u32(0x7fffffff)))
					large := k.lor(k.flt(k.f32(4), a), k.feq(a, k.f32(4)))
					b0 = k.fadd(b0, k.selF(large, k.f32(0), p))
					b1 = k.fadd(b1, k.selF(large, p, k.f32(0)))
				}
			}
		}
		// In units of 2^-10: i0 + i1·2^7.
		i0 := k.fToI(k.fmul(b0, k.f32(1<<10)))
		i1 := k.fToI(k.fmul(b1, k.f32(8)))
		hi := k.sar(i1, k.u32(9))
		lo := k.iaddS(i0, k.toI(k.shl(k.and(k.toU(i1), k.u32(0x1ff)), k.u32(7))))
		sum = k.intToF(hi, lo, 10)
	}
	var sidx id
	if kind == kindFP8 {
		sidx = k.iadd(k.imul(k.shr(row, k.u32(5)), nb), blk)
	} else {
		sidx = k.iadd(k.imul(row, nb), blk)
	}
	sbyte := k.and(k.shr(k.word(ws, k.shr(sidx, k.u32(2))), k.imul(k.and(sidx, k.u32(3)), k.u32(8))), k.u32(0xff))
	return k.fmul(k.fmul(sum, xsc), k.e8m0(sbyte))
}
