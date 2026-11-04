package gpu

// Kernels compute exactly what the CPU code computes, bit for bit, so a
// model runs the same on any mix of CPUs and GPUs.
//
// Matrix-vector products follow package num:
//
//   - float32 and bfloat16 weights (num.MatVec32, num.MatVecBF16): lane j
//     of 32 sums elements j, j+32, j+64, ... of a row in order, each
//     product rounded before it is added; the lanes combine as
//     e[j] = (s[j]+s[j+8]) + (s[j+16]+s[j+24]), then
//     ((e0+e1)+(e2+e3)) + ((e4+e5)+(e6+e7)).
//   - Q8_0 and Q4_0 weights (num.MatVecQ8, num.MatVecQ4), with the input
//     quantized to Q8_0 by the quantize kernel: each block's dot product is
//     an exact integer; block b contributes float32(sum)·float32(dw·dx) to
//     lane b mod 8, in order of b, and the eight lanes combine as above.
//
// The other kernels follow a transformer layer of package llm (Stage.Forward):
// RMS normalization, bias additions, rotary embeddings, attention (scores,
// softmax, weighted values), the residual additions, SiLU gating.
//
// Data: descriptor set 1 binds an arena, one buffer of 32-bit words holding
// a computation's activations, caches and tables, addressed by word offsets
// in push constants. Set 0 binds weights: a matrix part (binding 0) and its
// scales (binding 1), or a layer's vectors (binding 0).

const (
	laneCount    = 32
	rowsPerGroup = 4
	flatGroup    = 64 // threads per workgroup of the element-wise kernels
	pushWords    = 16
)

// kernelKind names a kernel.
type kernelKind int

const (
	kindF32 kernelKind = iota
	kindBF16
	kindQ8
	kindQ4
	kindQuantize
	kindRMSNorm
	kindAddVec
	kindRope
	kindCopy
	kindScores
	kindSoftmax
	kindWSum
	kindResidual
	kindSiluMul
	kindFP8
	kindFP4
	kindActFP8
	kindMathTest
	kindCount
)

// kernel builds a kernel's SPIR-V for a device's float modes.
func kernel(k kernelKind, modes floatModes) []uint32 {
	switch k {
	case kindF32:
		return floatKernel(modes)
	case kindBF16:
		return bf16Kernel(modes)
	case kindQ8, kindQ4:
		return quantKernel(k, modes)
	case kindQuantize:
		return quantizeKernel(modes)
	case kindRMSNorm:
		return rmsNormKernel(modes)
	case kindAddVec:
		return addVecKernel(modes)
	case kindRope:
		return ropeKernel(modes)
	case kindCopy:
		return copyKernel(modes)
	case kindScores:
		return scoresKernel(modes)
	case kindSoftmax:
		return softmaxKernel(modes)
	case kindWSum:
		return wsumKernel(modes)
	case kindResidual:
		return residualKernel(modes)
	case kindSiluMul:
		return siluMulKernel(modes)
	case kindFP8, kindFP4:
		return mxKernel(k, modes)
	case kindActFP8:
		return actFP8Kernel(modes)
	case kindMathTest:
		return mathTestKernel(modes)
	}
	panic("gpu: unknown kernel")
}

// groupRows is how many rows a matrix kernel's workgroup computes.
func groupRows(kind kernelKind) int {
	switch kind {
	case kindBF16:
		return bf16Rows
	case kindFP8, kindFP4:
		return 2 * mxRows
	}
	return rowsPerGroup
}

// bfloat16 rows take 16 threads, each summing two adjacent lanes from one
// word: lane order is unchanged, and each word is read once.
const bf16Rows = 8

// kbuilder adds the arena and push constants every kernel has.
type kbuilder struct {
	*builder
	arena, pc id
}

func newKernel(modes floatModes, x, y uint32) *kbuilder {
	b := newBuilder(modes, x, y)
	k := &kbuilder{builder: b}
	k.arena = b.storageBufferIn(1, 0, false)
	k.pc = b.pushConstants(pushWords)
	return k
}

func (k *kbuilder) p(i uint32) id    { return k.push(k.pc, i) }
func (k *kbuilder) get(off id) id    { return k.toF(k.word(k.arena, off)) }
func (k *kbuilder) put(off, v id)    { k.setWord(k.arena, off, k.toU(v)) }
func (k *kbuilder) at(base, i id) id { return k.iadd(base, i) }
func (k *kbuilder) ids() (lid, wid id) {
	return k.load(k.tV3, k.builtin(builtInLocalID)), k.load(k.tV3, k.builtin(builtInWorkgroupID))
}

// flat returns the global index of an element-wise kernel's thread.
func (k *kbuilder) flat() id {
	lid, wid := k.ids()
	return k.iadd(k.imul(k.component(wid, 0), k.u32(flatGroup)), k.component(lid, 0))
}

// combine8 adds eight lane sums in the fixed tree.
func (b *builder) combine8(e [8]id) id {
	return b.fadd(b.fadd(b.fadd(e[0], e[1]), b.fadd(e[2], e[3])), b.fadd(b.fadd(e[4], e[5]), b.fadd(e[6], e[7])))
}

// reduce32 combines 32 lane sums of group row ly, held in sums32, in the
// fixed tree, and calls done with the result in lane 0 of the row.
func (k *kbuilder) reduce32(lane, ly, sums32, sums8 id, done func(v id)) {
	k.barrier()
	k.ifThen(k.ult(lane, k.u32(8)), func() {
		base := k.imul(ly, k.u32(laneCount))
		at := func(off uint32) id {
			return k.load(k.tF32, k.shared(sums32, k.iadd(base, k.iadd(lane, k.u32(off)))))
		}
		e := k.fadd(k.fadd(at(0), at(8)), k.fadd(at(16), at(24)))
		k.store(k.shared(sums8, k.iadd(k.imul(ly, k.u32(8)), lane)), e)
	})
	k.barrier()
	k.ifThen(k.ieq(lane, k.u32(0)), func() {
		var e [8]id
		for i := range e {
			e[i] = k.load(k.tF32, k.shared(sums8, k.iadd(k.imul(ly, k.u32(8)), k.u32(uint32(i)))))
		}
		done(k.combine8(e))
	})
}

// matrixRow computes the workgroup's row for this thread of a matrix
// kernel, and its thread index within the row.
func (k *kbuilder) matrixRow(rows uint32) (row, lane, ly id) {
	lid, wid := k.ids()
	lane, ly = k.component(lid, 0), k.component(lid, 1)
	group := k.iadd(k.imul(k.component(wid, 1), k.p(2)), k.component(wid, 0))
	return k.iadd(k.imul(group, k.u32(rows)), ly), lane, ly
}

// floatKernel: push rows, cols, gridX, outOff, inOff.
func floatKernel(modes floatModes) []uint32 {
	k := newKernel(modes, laneCount, rowsPerGroup)
	w := k.storageBuffer(0, true)
	sums32 := k.workgroupFloats(laneCount * rowsPerGroup)
	sums8 := k.workgroupFloats(8 * rowsPerGroup)
	acc, j := k.local(k.tF32), k.local(k.tU32)
	row, lane, ly := k.matrixRow(rowsPerGroup)
	rows, cols, outOff, inOff := k.p(0), k.p(1), k.p(3), k.p(4)
	inRow := k.ult(row, rows)
	k.store(acc, k.f32(0))
	k.ifThen(inRow, func() {
		base := k.imul(row, cols)
		k.store(j, lane)
		k.loop(func() id { return k.ult(k.load(k.tU32, j), cols) }, func() {
			jj := k.load(k.tU32, j)
			p := k.fmul(k.toF(k.word(w, k.iadd(base, jj))), k.get(k.at(inOff, jj)))
			k.store(acc, k.fadd(k.load(k.tF32, acc), p))
		}, func() { k.store(j, k.iadd(k.load(k.tU32, j), k.u32(laneCount))) })
	})
	k.store(k.shared(sums32, k.iadd(k.imul(ly, k.u32(laneCount)), lane)), k.load(k.tF32, acc))
	k.reduce32(lane, ly, sums32, sums8, func(v id) {
		k.ifThen(inRow, func() { k.put(k.at(outOff, row), v) })
	})
	return k.finish()
}

// bf16Kernel: thread t of a row's 16 sums lanes 2t and 2t+1, reading the
// word holding elements 2t+32k and 2t+1+32k (rows have an even width, so
// words never straddle rows). Push as floatKernel.
func bf16Kernel(modes floatModes) []uint32 {
	k := newKernel(modes, laneCount/2, bf16Rows)
	w := k.storageBuffer(0, true)
	sums32 := k.workgroupFloats(laneCount * bf16Rows)
	sums8 := k.workgroupFloats(8 * bf16Rows)
	acc0, acc1, j := k.local(k.tF32), k.local(k.tF32), k.local(k.tU32)
	row, t, ly := k.matrixRow(bf16Rows)
	rows, cols, outOff, inOff := k.p(0), k.p(1), k.p(3), k.p(4)
	inRow := k.ult(row, rows)
	even := k.imul(t, k.u32(2))
	k.store(acc0, k.f32(0))
	k.store(acc1, k.f32(0))
	k.ifThen(inRow, func() {
		base := k.imul(row, cols)
		k.store(j, even)
		k.loop(func() id { return k.ult(k.load(k.tU32, j), cols) }, func() {
			jj := k.load(k.tU32, j)
			word := k.word(w, k.shr(k.iadd(base, jj), k.u32(1)))
			p0 := k.fmul(k.toF(k.shl(word, k.u32(16))), k.get(k.at(inOff, jj)))
			k.store(acc0, k.fadd(k.load(k.tF32, acc0), p0))
			p1 := k.fmul(k.toF(k.and(word, k.u32(0xffff0000))), k.get(k.at(inOff, k.iadd(jj, k.u32(1)))))
			k.store(acc1, k.fadd(k.load(k.tF32, acc1), p1))
		}, func() { k.store(j, k.iadd(k.load(k.tU32, j), k.u32(laneCount))) })
	})
	slot := k.iadd(k.imul(ly, k.u32(laneCount)), even)
	k.store(k.shared(sums32, slot), k.load(k.tF32, acc0))
	k.store(k.shared(sums32, k.iadd(slot, k.u32(1))), k.load(k.tF32, acc1))
	k.reduce32(t, ly, sums32, sums8, func(v id) {
		k.ifThen(inRow, func() { k.put(k.at(outOff, row), v) })
	})
	return k.finish()
}

// quantKernel: push rows, cols, gridX, outOff, qxOff (words of packed
// bytes), xsOff.
func quantKernel(kind kernelKind, modes floatModes) []uint32 {
	k := newKernel(modes, laneCount, rowsPerGroup)
	w, ws := k.storageBuffer(0, true), k.storageBuffer(1, true)
	terms := k.workgroupFloats(laneCount * rowsPerGroup)
	sums8 := k.workgroupFloats(8 * rowsPerGroup)
	laneAcc, term, chunk := k.local(k.tF32), k.local(k.tF32), k.local(k.tU32)
	row, lane, ly := k.matrixRow(rowsPerGroup)
	rows, cols, outOff, qxOff, xsOff := k.p(0), k.p(1), k.p(3), k.p(4), k.p(5)
	inRow := k.ult(row, rows)
	nb := k.shr(cols, k.u32(5))
	chunks := k.shr(k.iadd(nb, k.u32(laneCount-1)), k.u32(5))
	k.store(laneAcc, k.f32(0))
	k.store(chunk, k.u32(0))
	k.loop(func() id { return k.ult(k.load(k.tU32, chunk), chunks) }, func() {
		c := k.load(k.tU32, chunk)
		blk := k.iadd(k.imul(c, k.u32(laneCount)), lane)
		k.store(term, k.f32(0))
		k.ifThen(k.land(inRow, k.ult(blk, nb)), func() {
			xbase := k.iadd(qxOff, k.imul(blk, k.u32(8)))
			var xw [8]id
			for i := range xw {
				xw[i] = k.toI(k.word(k.arena, k.iadd(xbase, k.u32(uint32(i)))))
			}
			xbyte := func(e uint32) id { return k.sext(xw[e/4], 8*(e%4)) }
			si := k.i32(0)
			if kind == kindQ8 {
				wbase := k.shr(k.iadd(k.imul(row, cols), k.imul(blk, k.u32(32))), k.u32(2))
				for i := range uint32(8) {
					ww := k.toI(k.word(w, k.iadd(wbase, k.u32(i))))
					for b := range uint32(4) {
						si = k.iaddS(si, k.imulS(k.sext(ww, 8*b), xbyte(4*i+b)))
					}
				}
			} else {
				wbase := k.shr(k.iadd(k.imul(row, k.shr(cols, k.u32(1))), k.imul(blk, k.u32(16))), k.u32(2))
				for i := range uint32(4) {
					ww := k.word(w, k.iadd(wbase, k.u32(i)))
					for b := range uint32(4) {
						j := 4*i + b
						v := k.and(k.shr(ww, k.u32(8*b)), k.u32(0xff))
						lo := k.isubS(k.toI(k.and(v, k.u32(15))), k.i32(8))
						hi := k.isubS(k.toI(k.shr(v, k.u32(4))), k.i32(8))
						si = k.iaddS(si, k.imulS(lo, xbyte(j)))
						si = k.iaddS(si, k.imulS(hi, xbyte(j+16)))
					}
				}
			}
			d := k.fmul(k.toF(k.word(ws, k.iadd(k.imul(row, nb), blk))), k.get(k.at(xsOff, blk)))
			k.store(term, k.fmul(k.sToF(si), d))
		})
		k.store(k.shared(terms, k.iadd(k.imul(ly, k.u32(laneCount)), lane)), k.load(k.tF32, term))
		k.barrier()
		// Lane L of eight adds the chunk's blocks L, L+8, L+16, L+24 that
		// exist, in order: blocks absent are skipped, not added as zeros,
		// which could turn a -0 sum into +0.
		k.ifThen(k.ult(lane, k.u32(8)), func() {
			for j := range uint32(4) {
				bj := k.iadd(blk, k.u32(8*j))
				k.ifThen(k.ult(bj, nb), func() {
					t := k.load(k.tF32, k.shared(terms, k.iadd(k.imul(ly, k.u32(laneCount)), k.iadd(lane, k.u32(8*j)))))
					k.store(laneAcc, k.fadd(k.load(k.tF32, laneAcc), t))
				})
			}
		})
		k.barrier()
	}, func() { k.store(chunk, k.iadd(k.load(k.tU32, chunk), k.u32(1))) })
	k.ifThen(k.ult(lane, k.u32(8)), func() {
		k.store(k.shared(sums8, k.iadd(k.imul(ly, k.u32(8)), lane)), k.load(k.tF32, laneAcc))
	})
	k.barrier()
	k.ifThen(k.land(k.ieq(lane, k.u32(0)), inRow), func() {
		var e [8]id
		for i := range e {
			e[i] = k.load(k.tF32, k.shared(sums8, k.iadd(k.imul(ly, k.u32(8)), k.u32(uint32(i)))))
		}
		k.put(k.at(outOff, row), k.combine8(e))
	})
	return k.finish()
}

// quantizeKernel quantizes n floats to Q8_0, one block per thread, as
// num.QuantizeActs. Push n, inOff, qxOff, xsOff.
func quantizeKernel(modes floatModes) []uint32 {
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
		d := k.fmul(amax, k.fc(1.0/127))
		k.put(k.at(xsOff, blk), d)
		zero := k.lor(k.feq(d, k.f32(0)), k.isNaN(d))
		id := k.recip32(d)
		for wi := range uint32(8) {
			word := k.u32(0)
			for bi := range uint32(4) {
				r := k.roundEven(k.fmul(vals[4*wi+bi], id))
				q := k.op(opConvertFToS, k.tI32, r)
				q = k.selI(k.sgt(q, k.i32(127)), k.i32(127), k.selI(k.slt(q, k.i32(-127)), k.i32(-127), q))
				word = k.or(word, k.shl(k.and(k.toU(q), k.u32(0xff)), k.u32(8*bi)))
			}
			word = k.sel(zero, k.u32(0), word)
			k.setWord(k.arena, k.iadd(qxOff, k.iadd(k.imul(blk, k.u32(8)), k.u32(wi))), word)
		}
	})
	return k.finish()
}

// rmsNormKernel normalizes segments of n floats, one workgroup of 32 per
// segment, as llm's rmsNorm: ss = dot(x, x)·recipN + eps (dot in the
// 32-lane order), out = w·(rsqrt(ss)·x). Out may be in. Push n, inOff,
// outOff, wOff (in the vectors at set 0), recipN, eps, stride (between
// segments).
func rmsNormKernel(modes floatModes) []uint32 {
	k := newKernel(modes, laneCount, 1)
	vec := k.storageBuffer(0, true)
	sums32, sums8 := k.workgroupFloats(laneCount), k.workgroupFloats(8)
	inv := k.workgroupFloats(1)
	acc, j := k.local(k.tF32), k.local(k.tU32)
	lid, wid := k.ids()
	lane := k.component(lid, 0)
	n, stride := k.p(0), k.p(6)
	seg := k.imul(k.component(wid, 0), stride)
	in, out, wOff := k.iadd(k.p(1), seg), k.iadd(k.p(2), seg), k.p(3)
	k.store(acc, k.f32(0))
	k.store(j, lane)
	k.loop(func() id { return k.ult(k.load(k.tU32, j), n) }, func() {
		v := k.get(k.at(in, k.load(k.tU32, j)))
		k.store(acc, k.fadd(k.load(k.tF32, acc), k.fmul(v, v)))
	}, func() { k.store(j, k.iadd(k.load(k.tU32, j), k.u32(laneCount))) })
	k.store(k.shared(sums32, lane), k.load(k.tF32, acc))
	zero := k.u32(0)
	k.reduce32(lane, zero, sums32, sums8, func(dot id) {
		ss := k.fadd(k.fmul(dot, k.toF(k.p(4))), k.toF(k.p(5)))
		k.store(k.shared(inv, zero), k.rsqrt32(ss))
	})
	k.barrier()
	r := k.load(k.tF32, k.shared(inv, zero))
	k.store(j, lane)
	k.loop(func() id { return k.ult(k.load(k.tU32, j), n) }, func() {
		jj := k.load(k.tU32, j)
		wv := k.toF(k.word(vec, k.iadd(wOff, jj)))
		k.put(k.at(out, jj), k.fmul(wv, k.fmul(r, k.get(k.at(in, jj)))))
	}, func() { k.store(j, k.iadd(k.load(k.tU32, j), k.u32(laneCount))) })
	return k.finish()
}

// addVecKernel adds a vector from set 0: out += v. Push n, outOff, vOff.
func addVecKernel(modes floatModes) []uint32 {
	k := newKernel(modes, flatGroup, 1)
	vec := k.storageBuffer(0, true)
	i := k.flat()
	k.ifThen(k.ult(i, k.p(0)), func() {
		o := k.at(k.p(1), i)
		k.put(o, k.fadd(k.get(o), k.toF(k.word(vec, k.iadd(k.p(2), i)))))
	})
	return k.finish()
}

// ropeKernel rotates adjacent pairs of each query and key head. Push
// pairs (qDim/2), qOff, kOff, kvDim, head size, cosOff, sinOff (of the
// position's row of the tables).
func ropeKernel(modes floatModes) []uint32 {
	k := newKernel(modes, flatGroup, 1)
	p := k.flat()
	k.ifThen(k.ult(p, k.p(0)), func() {
		i := k.imul(p, k.u32(2))
		hs := k.p(4)
		pi := k.shr(k.op(opUMod, k.tU32, i, hs), k.u32(1))
		co, si := k.get(k.at(k.p(5), pi)), k.get(k.at(k.p(6), pi))
		rot := func(base id) {
			a, b := k.at(base, i), k.at(base, k.iadd(i, k.u32(1)))
			v0, v1 := k.get(a), k.get(b)
			k.put(a, k.fsub(k.fmul(v0, co), k.fmul(v1, si)))
			k.put(b, k.fadd(k.fmul(v0, si), k.fmul(v1, co)))
		}
		rot(k.p(1))
		k.ifThen(k.ult(i, k.p(3)), func() { rot(k.p(2)) })
	})
	return k.finish()
}

// copyKernel copies words. Push n, src, dst.
func copyKernel(modes floatModes) []uint32 {
	k := newKernel(modes, flatGroup, 1)
	i := k.flat()
	k.ifThen(k.ult(i, k.p(0)), func() {
		k.setWord(k.arena, k.at(k.p(2), i), k.word(k.arena, k.at(k.p(1), i)))
	})
	return k.finish()
}

// scoresKernel computes attention scores: for head h (workgroup y) and
// position t, dot(q_h, k_t of h's key head) in the 32-lane order, times
// invSqrt. Push rows (positions), head size, kOff, kvDim, kvMul, qOff,
// attOff, attStride, invSqrt.
func scoresKernel(modes floatModes) []uint32 {
	k := newKernel(modes, laneCount, rowsPerGroup)
	sums32 := k.workgroupFloats(laneCount * rowsPerGroup)
	sums8 := k.workgroupFloats(8 * rowsPerGroup)
	acc, j := k.local(k.tF32), k.local(k.tU32)
	lid, wid := k.ids()
	lane, ly := k.component(lid, 0), k.component(lid, 1)
	t := k.iadd(k.imul(k.component(wid, 0), k.u32(rowsPerGroup)), ly)
	h := k.component(wid, 1)
	rows, hs, kvDim, kvMul := k.p(0), k.p(1), k.p(3), k.p(4)
	inRow := k.ult(t, rows)
	q := k.iadd(k.p(5), k.imul(h, hs))
	key := k.iadd(k.p(2), k.iadd(k.imul(t, kvDim), k.imul(k.op(opUDiv, k.tU32, h, kvMul), hs)))
	k.store(acc, k.f32(0))
	k.ifThen(inRow, func() {
		k.store(j, lane)
		k.loop(func() id { return k.ult(k.load(k.tU32, j), hs) }, func() {
			jj := k.load(k.tU32, j)
			k.store(acc, k.fadd(k.load(k.tF32, acc), k.fmul(k.get(k.at(q, jj)), k.get(k.at(key, jj)))))
		}, func() { k.store(j, k.iadd(k.load(k.tU32, j), k.u32(laneCount))) })
	})
	k.store(k.shared(sums32, k.iadd(k.imul(ly, k.u32(laneCount)), lane)), k.load(k.tF32, acc))
	k.reduce32(lane, ly, sums32, sums8, func(dot id) {
		k.ifThen(inRow, func() {
			k.put(k.iadd(k.p(6), k.iadd(k.imul(h, k.p(7)), t)), k.fmul(dot, k.toF(k.p(8))))
		})
	})
	return k.finish()
}

// softmaxKernel normalizes each head's scores, one workgroup of 32 per
// head, as llm's softmax: the maximum, exp of the differences, their sum
// in order (one thread), and multiplication by its reciprocal. Push n,
// attOff, attStride.
func softmaxKernel(modes floatModes) []uint32 {
	k := newKernel(modes, laneCount, 1)
	part := k.workgroupFloats(laneCount)
	scale := k.workgroupFloats(1)
	mx, j, sum := k.local(k.tF32), k.local(k.tU32), k.local(k.tF32)
	lid, wid := k.ids()
	lane := k.component(lid, 0)
	n := k.p(0)
	base := k.iadd(k.p(1), k.imul(k.component(wid, 0), k.p(2)))
	zero := k.u32(0)
	// The maximum is exact in any order (Go's max, for -0 and +0 too).
	k.store(mx, k.get(base))
	k.store(j, lane)
	k.loop(func() id { return k.ult(k.load(k.tU32, j), n) }, func() {
		k.store(mx, k.gomax(k.load(k.tF32, mx), k.get(k.at(base, k.load(k.tU32, j)))))
	}, func() { k.store(j, k.iadd(k.load(k.tU32, j), k.u32(laneCount))) })
	k.store(k.shared(part, lane), k.load(k.tF32, mx))
	k.barrier()
	k.ifThen(k.ieq(lane, zero), func() {
		m := k.load(k.tF32, k.shared(part, zero))
		for i := uint32(1); i < laneCount; i++ {
			m = k.gomax(m, k.load(k.tF32, k.shared(part, k.u32(i))))
		}
		k.store(k.shared(scale, zero), m)
	})
	k.barrier()
	m := k.load(k.tF32, k.shared(scale, zero))
	k.store(j, lane)
	k.loop(func() id { return k.ult(k.load(k.tU32, j), n) }, func() {
		o := k.at(base, k.load(k.tU32, j))
		k.put(o, k.exp32(k.fsub(k.get(o), m)))
	}, func() { k.store(j, k.iadd(k.load(k.tU32, j), k.u32(laneCount))) })
	k.barrier()
	k.ifThen(k.ieq(lane, zero), func() {
		k.store(sum, k.f32(0))
		k.store(j, zero)
		k.loop(func() id { return k.ult(k.load(k.tU32, j), n) }, func() {
			k.store(sum, k.fadd(k.load(k.tF32, sum), k.get(k.at(base, k.load(k.tU32, j)))))
		}, func() { k.store(j, k.iadd(k.load(k.tU32, j), k.u32(1))) })
		k.store(k.shared(scale, zero), k.recip32(k.load(k.tF32, sum)))
	})
	k.barrier()
	inv := k.load(k.tF32, k.shared(scale, zero))
	k.store(j, lane)
	k.loop(func() id { return k.ult(k.load(k.tU32, j), n) }, func() {
		o := k.at(base, k.load(k.tU32, j))
		k.put(o, k.fmul(k.get(o), inv))
	}, func() { k.store(j, k.iadd(k.load(k.tU32, j), k.u32(laneCount))) })
	return k.finish()
}

// wsumKernel sums values weighted by attention, per output element, over
// positions in order from zero. Push n (positions), head size, vOff,
// kvDim, kvMul, attOff, attStride, outOff, qDim.
func wsumKernel(modes floatModes) []uint32 {
	k := newKernel(modes, flatGroup, 1)
	acc, t := k.local(k.tF32), k.local(k.tU32)
	g := k.flat()
	k.ifThen(k.ult(g, k.p(8)), func() {
		hs := k.p(1)
		h, i := k.op(opUDiv, k.tU32, g, hs), k.op(opUMod, k.tU32, g, hs)
		att := k.iadd(k.p(5), k.imul(h, k.p(6)))
		val := k.iadd(k.p(2), k.iadd(k.imul(k.op(opUDiv, k.tU32, h, k.p(4)), hs), i))
		k.store(acc, k.f32(0))
		k.store(t, k.u32(0))
		k.loop(func() id { return k.ult(k.load(k.tU32, t), k.p(0)) }, func() {
			tt := k.load(k.tU32, t)
			p := k.fmul(k.get(k.at(att, tt)), k.get(k.iadd(val, k.imul(tt, k.p(3)))))
			k.store(acc, k.fadd(k.load(k.tF32, acc), p))
		}, func() { k.store(t, k.iadd(k.load(k.tU32, t), k.u32(1))) })
		k.put(k.at(k.p(7), g), k.load(k.tF32, acc))
	})
	return k.finish()
}

// residualKernel adds y to x, canonicalizing NaNs if asked. Push n, xOff,
// yOff, canonical.
func residualKernel(modes floatModes) []uint32 {
	k := newKernel(modes, flatGroup, 1)
	i := k.flat()
	k.ifThen(k.ult(i, k.p(0)), func() {
		o := k.at(k.p(1), i)
		v := k.fadd(k.get(o), k.get(k.at(k.p(2), i)))
		k.put(o, k.selF(k.ieq(k.p(3), k.u32(1)), k.canonical(v), v))
	})
	return k.finish()
}

// siluMulKernel sets a = silu(a)·b, silu(v) = v·recip(1 + exp(-v)). Push
// n, aOff, bOff.
func siluMulKernel(modes floatModes) []uint32 {
	k := newKernel(modes, flatGroup, 1)
	i := k.flat()
	k.ifThen(k.ult(i, k.p(0)), func() {
		o := k.at(k.p(1), i)
		v := k.get(o)
		s := k.fmul(v, k.recip32(k.fadd(k.f32(1), k.exp32(k.fneg(v)))))
		k.put(o, k.fmul(s, k.get(k.at(k.p(2), i))))
	})
	return k.finish()
}
