package gpu

// The matrix-vector kernels compute exactly what package num computes on
// the CPU, bit for bit, so a model runs the same on any mix of CPUs and
// GPUs:
//
//   - float32 and bfloat16 weights (num.MatVec32, num.MatVecBF16): lane j
//     of 32 sums elements j, j+32, j+64, ... of a row in order, each
//     product rounded before it is added; the lanes combine as
//     e[j] = (s[j]+s[j+8]) + (s[j+16]+s[j+24]), then
//     ((e0+e1)+(e2+e3)) + ((e4+e5)+(e6+e7)).
//   - Q8_0 and Q4_0 weights (num.MatVecQ8, num.MatVecQ4), with the input
//     quantized to Q8_0 on the host: each block's dot product is an exact
//     integer; block b contributes float32(sum)·float32(dw·dx) to lane
//     b mod 8, in order of b, and the eight lanes combine as above.
//
// A workgroup is 32 threads by rowsPerGroup rows; thread x of row y is
// lane x. Push constants: rows, cols, groups per dispatch row (the grid
// may be two-dimensional, past the 65535 limit of one dimension), and the
// offset of the first row in the output.

const (
	laneCount    = 32
	rowsPerGroup = 4
)

// kernelKind names a weight format.
type kernelKind int

const (
	kindF32 kernelKind = iota
	kindBF16
	kindQ8
	kindQ4
)

// rowIndex computes the workgroup's row for this thread, and its thread
// index within the row.
func (b *builder) rowIndex(pc id, rowsPerGroup uint32) (row, lane, ly id) {
	lid := b.load(b.tV3, b.builtin(builtInLocalID))
	wid := b.load(b.tV3, b.builtin(builtInWorkgroupID))
	lane, ly = b.component(lid, 0), b.component(lid, 1)
	group := b.iadd(b.imul(b.component(wid, 1), b.push(pc, 2)), b.component(wid, 0))
	row = b.iadd(b.imul(group, b.u32(rowsPerGroup)), ly)
	return row, lane, ly
}

// groupRows is how many rows a kernel's workgroup computes.
func groupRows(kind kernelKind) int {
	if kind == kindBF16 {
		return bf16Rows
	}
	return rowsPerGroup
}

// bfloat16 rows take 16 threads, each summing two adjacent lanes from one
// word: lane order is unchanged, and each word is read once.
const bf16Rows = 8

// combine8 adds eight lane sums in the fixed tree.
func (b *builder) combine8(e [8]id) id {
	return b.fadd(b.fadd(b.fadd(e[0], e[1]), b.fadd(e[2], e[3])), b.fadd(b.fadd(e[4], e[5]), b.fadd(e[6], e[7])))
}

// finishRow reduces this row's lane values, kept in lanes 0-7 of eight
// (in tree8) or all 32 (otherwise), and stores the row's result.
func (b *builder) finishRow(out, pc, row, lane, ly, inRow id, sums32, sums8 id, from32 bool) {
	if from32 {
		b.barrier()
		b.ifThen(b.ult(lane, b.u32(8)), func() {
			base := b.imul(ly, b.u32(laneCount))
			at := func(off uint32) id {
				return b.load(b.tF32, b.shared(sums32, b.iadd(base, b.iadd(lane, b.u32(off)))))
			}
			e := b.fadd(b.fadd(at(0), at(8)), b.fadd(at(16), at(24)))
			b.store(b.shared(sums8, b.iadd(b.imul(ly, b.u32(8)), lane)), e)
		})
	}
	b.barrier()
	b.ifThen(b.land(b.ieq(lane, b.u32(0)), inRow), func() {
		var e [8]id
		for i := range e {
			e[i] = b.load(b.tF32, b.shared(sums8, b.iadd(b.imul(ly, b.u32(8)), b.u32(uint32(i)))))
		}
		b.setWord(out, b.iadd(b.push(pc, 3), row), b.toU(b.combine8(e)))
	})
}

// floatKernel multiplies float32 or bfloat16 weights by a float32 vector.
// Bindings: 0 weights, 1 input, 2 output.
func floatKernel(kind kernelKind, modes floatModes) []uint32 {
	if kind == kindBF16 {
		return bf16Kernel(modes)
	}
	b := newBuilder(modes, laneCount, rowsPerGroup)
	w, x, out := b.storageBuffer(0, true), b.storageBuffer(1, true), b.storageBuffer(2, false)
	pc := b.pushConstants(4)
	sums32 := b.workgroupFloats(laneCount * rowsPerGroup)
	sums8 := b.workgroupFloats(8 * rowsPerGroup)
	acc, k := b.local(b.tF32), b.local(b.tU32)

	row, lane, ly := b.rowIndex(pc, rowsPerGroup)
	rows, cols := b.push(pc, 0), b.push(pc, 1)
	inRow := b.ult(row, rows)
	b.store(acc, b.f32(0))
	b.ifThen(inRow, func() {
		base := b.imul(row, cols)
		b.store(k, lane)
		b.loop(func() id { return b.ult(b.load(b.tU32, k), cols) }, func() {
			kk := b.load(b.tU32, k)
			idx := b.iadd(base, kk)
			p := b.fmul(b.toF(b.word(w, idx)), b.toF(b.word(x, kk)))
			b.store(acc, b.fadd(b.load(b.tF32, acc), p))
		}, func() {
			b.store(k, b.iadd(b.load(b.tU32, k), b.u32(laneCount)))
		})
	})
	b.store(b.shared(sums32, b.iadd(b.imul(ly, b.u32(laneCount)), lane)), b.load(b.tF32, acc))
	b.finishRow(out, pc, row, lane, ly, inRow, sums32, sums8, true)
	return b.finish()
}

// bf16Kernel: thread t of a row's 16 sums lanes 2t and 2t+1, reading the
// word holding elements 2t+32k and 2t+1+32k (rows have an even width, so
// words never straddle rows).
func bf16Kernel(modes floatModes) []uint32 {
	b := newBuilder(modes, laneCount/2, bf16Rows)
	w, x, out := b.storageBuffer(0, true), b.storageBuffer(1, true), b.storageBuffer(2, false)
	pc := b.pushConstants(4)
	sums32 := b.workgroupFloats(laneCount * bf16Rows)
	sums8 := b.workgroupFloats(8 * bf16Rows)
	acc0, acc1, k := b.local(b.tF32), b.local(b.tF32), b.local(b.tU32)

	row, t, ly := b.rowIndex(pc, bf16Rows)
	rows, cols := b.push(pc, 0), b.push(pc, 1)
	inRow := b.ult(row, rows)
	even := b.imul(t, b.u32(2))
	b.store(acc0, b.f32(0))
	b.store(acc1, b.f32(0))
	b.ifThen(inRow, func() {
		base := b.imul(row, cols)
		b.store(k, even)
		b.loop(func() id { return b.ult(b.load(b.tU32, k), cols) }, func() {
			kk := b.load(b.tU32, k)
			word := b.word(w, b.shr(b.iadd(base, kk), b.u32(1)))
			p0 := b.fmul(b.toF(b.shl(word, b.u32(16))), b.toF(b.word(x, kk)))
			b.store(acc0, b.fadd(b.load(b.tF32, acc0), p0))
			p1 := b.fmul(b.toF(b.and(word, b.u32(0xffff0000))), b.toF(b.word(x, b.iadd(kk, b.u32(1)))))
			b.store(acc1, b.fadd(b.load(b.tF32, acc1), p1))
		}, func() {
			b.store(k, b.iadd(b.load(b.tU32, k), b.u32(laneCount)))
		})
	})
	slot := b.iadd(b.imul(ly, b.u32(laneCount)), even)
	b.store(b.shared(sums32, slot), b.load(b.tF32, acc0))
	b.store(b.shared(sums32, b.iadd(slot, b.u32(1))), b.load(b.tF32, acc1))
	b.finishRow(out, pc, row, t, ly, inRow, sums32, sums8, true)
	return b.finish()
}

// quantKernel multiplies Q8_0 or Q4_0 weights by a Q8_0 vector. Bindings:
// 0 weights, 1 weight scales, 2 quantized input, 3 input scales, 4 output.
func quantKernel(kind kernelKind, modes floatModes) []uint32 {
	b := newBuilder(modes, laneCount, rowsPerGroup)
	w, ws := b.storageBuffer(0, true), b.storageBuffer(1, true)
	qx, xs := b.storageBuffer(2, true), b.storageBuffer(3, true)
	out := b.storageBuffer(4, false)
	pc := b.pushConstants(4)
	terms := b.workgroupFloats(laneCount * rowsPerGroup)
	sums8 := b.workgroupFloats(8 * rowsPerGroup)
	laneAcc, term, chunk := b.local(b.tF32), b.local(b.tF32), b.local(b.tU32)

	row, lane, ly := b.rowIndex(pc, rowsPerGroup)
	rows, cols := b.push(pc, 0), b.push(pc, 1)
	inRow := b.ult(row, rows)
	nb := b.shr(cols, b.u32(5))
	chunks := b.shr(b.iadd(nb, b.u32(laneCount-1)), b.u32(5))
	b.store(laneAcc, b.f32(0))
	b.store(chunk, b.u32(0))
	b.loop(func() id { return b.ult(b.load(b.tU32, chunk), chunks) }, func() {
		c := b.load(b.tU32, chunk)
		blk := b.iadd(b.imul(c, b.u32(laneCount)), lane)
		b.store(term, b.f32(0))
		b.ifThen(b.land(inRow, b.ult(blk, nb)), func() {
			xbase := b.imul(blk, b.u32(8))
			var xw [8]id
			for i := range xw {
				xw[i] = b.toI(b.word(qx, b.iadd(xbase, b.u32(uint32(i)))))
			}
			xbyte := func(e uint32) id { return b.sext(xw[e/4], 8*(e%4)) }
			si := b.i32(0)
			if kind == kindQ8 {
				wbase := b.shr(b.iadd(b.imul(row, cols), b.imul(blk, b.u32(32))), b.u32(2))
				for i := range uint32(8) {
					ww := b.toI(b.word(w, b.iadd(wbase, b.u32(i))))
					for k := range uint32(4) {
						si = b.iaddS(si, b.imulS(b.sext(ww, 8*k), xbyte(4*i+k)))
					}
				}
			} else {
				wbase := b.shr(b.iadd(b.imul(row, b.shr(cols, b.u32(1))), b.imul(blk, b.u32(16))), b.u32(2))
				for i := range uint32(4) {
					ww := b.word(w, b.iadd(wbase, b.u32(i)))
					for k := range uint32(4) {
						j := 4*i + k
						v := b.and(b.shr(ww, b.u32(8*k)), b.u32(0xff))
						lo := b.isubS(b.toI(b.and(v, b.u32(15))), b.i32(8))
						hi := b.isubS(b.toI(b.shr(v, b.u32(4))), b.i32(8))
						si = b.iaddS(si, b.imulS(lo, xbyte(j)))
						si = b.iaddS(si, b.imulS(hi, xbyte(j+16)))
					}
				}
			}
			d := b.fmul(b.toF(b.word(ws, b.iadd(b.imul(row, nb), blk))), b.toF(b.word(xs, blk)))
			b.store(term, b.fmul(b.sToF(si), d))
		})
		b.store(b.shared(terms, b.iadd(b.imul(ly, b.u32(laneCount)), lane)), b.load(b.tF32, term))
		b.barrier()
		// Lane L of eight adds the chunk's blocks L, L+8, L+16, L+24 that
		// exist, in order: blocks absent are skipped, not added as zeros,
		// which could turn a -0 sum into +0.
		b.ifThen(b.ult(lane, b.u32(8)), func() {
			for j := range uint32(4) {
				bj := b.iadd(blk, b.u32(8*j))
				b.ifThen(b.ult(bj, nb), func() {
					t := b.load(b.tF32, b.shared(terms, b.iadd(b.imul(ly, b.u32(laneCount)), b.iadd(lane, b.u32(8*j)))))
					b.store(laneAcc, b.fadd(b.load(b.tF32, laneAcc), t))
				})
			}
		})
		b.barrier()
	}, func() {
		b.store(chunk, b.iadd(b.load(b.tU32, chunk), b.u32(1)))
	})
	b.ifThen(b.ult(lane, b.u32(8)), func() {
		b.store(b.shared(sums8, b.iadd(b.imul(ly, b.u32(8)), lane)), b.load(b.tF32, laneAcc))
	})
	b.finishRow(out, pc, row, lane, ly, inRow, terms, sums8, false)
	return b.finish()
}
