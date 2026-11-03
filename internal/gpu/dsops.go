package gpu

import "math"

// Kernels of a DeepSeek-V4.1 block (package dsv41, forward.go and
// ops.go), each the CPU's arithmetic operation for operation: the same
// sums in the same order, every product rounded, divisions and square
// roots correctly rounded (crmath.go), exp, log and log1p as package num
// computes them. Indices (selections, row lists, expert ids) are words of
// the arena, -1 for none. Everything is in the arena: the block's vectors
// too, copied there once.

const (
	kindDSNorm kernelKind = iota + kindCount
	kindHCSplit
	kindHCPre
	kindHCPost
	kindRopeDS
	kindActQuant
	kindFP4Quant
	kindRoundScale
	kindCompress
	kindIdxScores
	kindCandPool
	kindTopK
	kindSparseAttn
	kindMoEScore
	kindMoEWeights
	kindFP4Experts
	kindSiluMulDS
	kindMoESum
	kindEngramGate
	kindAllCount
)

func dsKernel(k kernelKind, modes floatModes) []uint32 {
	switch k {
	case kindDSNorm:
		return dsNormKernel(modes)
	case kindHCSplit:
		return hcSplitKernel(modes)
	case kindHCPre:
		return hcPreKernel(modes)
	case kindHCPost:
		return hcPostKernel(modes)
	case kindRopeDS:
		return ropeDSKernel(modes)
	case kindActQuant:
		return actQuantKernel(modes)
	case kindFP4Quant:
		return fp4QuantKernel(modes)
	case kindRoundScale:
		return roundScaleKernel(modes)
	case kindCompress:
		return compressKernel(modes)
	case kindIdxScores:
		return idxScoresKernel(modes)
	case kindCandPool:
		return candPoolKernel(modes)
	case kindTopK:
		return topKKernel(modes)
	case kindSparseAttn:
		return sparseAttnKernel(modes)
	case kindMoEScore:
		return moeScoreKernel(modes)
	case kindMoEWeights:
		return moeWeightsKernel(modes)
	case kindFP4Experts:
		return mxKernelIndexed(modes)
	case kindSiluMulDS:
		return siluMulDSKernel(modes)
	case kindMoESum:
		return moeSumKernel(modes)
	case kindEngramGate:
		return engramGateKernel(modes)
	}
	panic("gpu: unknown kernel")
}

// gomin is Go's min for floats: NaN if either is, -0 for -0 and +0.
func (b *builder) gomin(x, y id) id {
	eq := b.toF(b.or(b.toU(x), b.toU(y)))
	r := b.selF(b.flt(x, y), x, b.selF(b.flt(y, x), y, eq))
	return b.selF(b.lor(b.isNaN(x), b.isNaN(y)), b.toF(b.u32(canonicalNaN)), r)
}

func (b *builder) fabs(x id) id { return b.toF(b.and(b.toU(x), b.u32(0x7fffffff))) }

func (k *kbuilder) iget(off id) id      { return k.word(k.arena, off) }
func (k *kbuilder) iput(off, v id)      { k.setWord(k.arena, off, v) }
func (k *kbuilder) pf(i uint32) id      { return k.toF(k.p(i)) }
func (k *kbuilder) cnt(c id) id         { return k.sel(c, k.u32(1), k.u32(0)) }
func (k *kbuilder) isNone(v id) id      { return k.ieq(v, k.u32(0xffffffff)) }
func (k *kbuilder) floorDiv(x, y id) id { return k.op(opUDiv, k.tU32, x, y) }
func (k *kbuilder) umod(x, y id) id     { return k.op(opUMod, k.tU32, x, y) }

var negInfBits = math.Float32bits(float32(math.Inf(-1)))

// forRange runs body(i) for i from lo while i < hi, by step.
func (k *kbuilder) forRange(lo, hi, step id, body func(i id)) {
	i := k.local(k.tU32)
	k.store(i, lo)
	k.loop(func() id { return k.ult(k.load(k.tU32, i), hi) }, func() {
		body(k.load(k.tU32, i))
	}, func() { k.store(i, k.iadd(k.load(k.tU32, i), step)) })
}

// dot32 is num.Dot32 of n floats at words a(i) and b(i), in one thread:
// sum j of 32 adds elements j, j+32, ... in order, then num's tree.
func (k *kbuilder) dot32(n id, a, b func(i id) id) id {
	var s [32]id
	for j := range s {
		acc := k.local(k.tF32)
		k.store(acc, k.f32(0))
		k.forRange(k.u32(uint32(j)), n, k.u32(32), func(i id) {
			k.store(acc, k.fadd(k.load(k.tF32, acc), k.fmul(a(i), b(i))))
		})
		s[j] = k.load(k.tF32, acc)
	}
	var e [8]id
	for j := range e {
		e[j] = k.fadd(k.fadd(s[j], s[j+8]), k.fadd(s[j+16], s[j+24]))
	}
	return k.combine8(e)
}

// laneDot computes, in a workgroup of 32 lanes, lane j's sum of the
// products of elements j, j+32, ... (num.Dot32's sum j).
func (k *kbuilder) laneDot(lane, n id, a, b func(i id) id) id {
	acc := k.local(k.tF32)
	k.store(acc, k.f32(0))
	k.forRange(lane, n, k.u32(32), func(i id) {
		k.store(acc, k.fadd(k.load(k.tF32, acc), k.fmul(a(i), b(i))))
	})
	return k.load(k.tF32, acc)
}

// rsqrtDS is dsv41's rsqrt: 1/sqrt(v), each correctly rounded.
func (b *builder) rsqrtDS(v id) id { return b.fdiv(b.f32(1), b.fsqrt(v)) }

// dsNormKernel is dsv41.rmsNorm on segs segments of n floats, one
// workgroup of 32 each: out = bf16(w·(x·rsqrt(Dot32(x,x)/n + eps))).
// Push n, in, out, w, segs, stride, eps.
func dsNormKernel(modes floatModes) []uint32 {
	k := newKernel(modes, laneCount, 1)
	sums32, sums8 := k.workgroupFloats(laneCount), k.workgroupFloats(8)
	rs := k.workgroupFloats(1)
	lid, wid := k.ids()
	lane, seg := k.component(lid, 0), k.component(wid, 0)
	n, stride := k.p(0), k.p(5)
	in := k.iadd(k.p(1), k.imul(seg, stride))
	out := k.iadd(k.p(2), k.imul(seg, stride))
	w := k.p(3)
	x := func(i id) id { return k.get(k.at(in, i)) }
	k.store(k.shared(sums32, lane), k.laneDot(lane, n, x, x))
	k.reduce32(lane, k.u32(0), sums32, sums8, func(v id) {
		ms := k.fdiv(v, k.op(opConvertUToF, k.tF32, n))
		k.store(k.shared(rs, k.u32(0)), k.rsqrtDS(k.fadd(ms, k.pf(6))))
	})
	k.barrier()
	r := k.load(k.tF32, k.shared(rs, k.u32(0)))
	k.forRange(lane, n, k.u32(32), func(i id) {
		k.put(k.at(out, i), k.roundBF16(k.fmul(k.get(k.at(w, i)), k.fmul(x(i), r))))
	})
	return k.finish()
}

const opConvertUToF = 112

// hcSplitKernel is dsv41's mixes after the product: the mixing logits at
// mix (24 for hc = 4) times rsqrt(meanSquare(h)+normEps), then
// splitSinkhorn. One workgroup of 32. Push mix, h, n (hc·dim), base,
// scale, iters, eps, normEps, pre, post, comb.
func hcSplitKernel(modes floatModes) []uint32 {
	const hc = 4
	k := newKernel(modes, laneCount, 1)
	sums32, sums8 := k.workgroupFloats(laneCount), k.workgroupFloats(8)
	lid, _ := k.ids()
	lane := k.component(lid, 0)
	mixOff, h, n, base, scale, iters := k.p(0), k.p(1), k.p(2), k.p(3), k.p(4), k.p(5)
	eps, normEps := k.pf(6), k.pf(7)
	preOff, postOff, combOff := k.p(8), k.p(9), k.p(10)
	x := func(i id) id { return k.get(k.at(h, i)) }
	k.store(k.shared(sums32, lane), k.laneDot(lane, n, x, x))
	k.reduce32(lane, k.u32(0), sums32, sums8, func(v id) {
		r := k.rsqrtDS(k.fadd(k.fdiv(v, k.op(opConvertUToF, k.tF32, n)), normEps))
		mix := func(i int) id { return k.fmul(k.get(k.at(mixOff, k.u32(uint32(i)))), r) }
		sc := func(i int) id { return k.get(k.at(scale, k.u32(uint32(i)))) }
		bs := func(i int) id { return k.get(k.at(base, k.u32(uint32(i)))) }
		for j := range hc {
			pre := k.fadd(k.sigmoid(k.fadd(k.fmul(mix(j), sc(0)), bs(j))), eps)
			post := k.fmul(k.f32(2), k.sigmoid(k.fadd(k.fmul(mix(hc+j), sc(1)), bs(hc+j))))
			k.put(k.at(preOff, k.u32(uint32(j))), pre)
			k.put(k.at(postOff, k.u32(uint32(j))), post)
		}
		var comb [hc * hc]id
		for i := range comb {
			comb[i] = k.local(k.tF32)
			k.store(comb[i], k.fadd(k.fmul(mix(2*hc+i), sc(2)), bs(2*hc+i)))
		}
		at := func(j, c int) id { return comb[j*hc+c] }
		ld := func(p id) id { return k.load(k.tF32, p) }
		for j := range hc {
			m := ld(at(j, 0))
			for c := 1; c < hc; c++ {
				m = k.gomax(m, ld(at(j, c)))
			}
			s := k.f32(0)
			for c := range hc {
				k.store(at(j, c), k.exp32(k.fsub(ld(at(j, c)), m)))
				s = k.fadd(s, ld(at(j, c)))
			}
			for c := range hc {
				k.store(at(j, c), k.fadd(k.fdiv(ld(at(j, c)), s), eps))
			}
		}
		colNorm := func() {
			for c := range hc {
				s := k.f32(0)
				for j := range hc {
					s = k.fadd(s, ld(at(j, c)))
				}
				d := k.fadd(s, eps)
				for j := range hc {
					k.store(at(j, c), k.fdiv(ld(at(j, c)), d))
				}
			}
		}
		colNorm()
		k.forRange(k.u32(1), iters, k.u32(1), func(id) {
			for j := range hc {
				s := k.f32(0)
				for c := range hc {
					s = k.fadd(s, ld(at(j, c)))
				}
				d := k.fadd(s, eps)
				for c := range hc {
					k.store(at(j, c), k.fdiv(ld(at(j, c)), d))
				}
			}
			colNorm()
		})
		for i := range comb {
			k.put(k.at(combOff, k.u32(uint32(i))), ld(comb[i]))
		}
	})
	return k.finish()
}

// hcPreKernel is dsv41.hcPre (hc = 4). Push dim, h, p, out.
func hcPreKernel(modes floatModes) []uint32 {
	k := newKernel(modes, flatGroup, 1)
	d := k.flat()
	dim, h, p, out := k.p(0), k.p(1), k.p(2), k.p(3)
	k.ifThen(k.ult(d, dim), func() {
		s := k.f32(0)
		for c := range uint32(4) {
			s = k.fadd(s, k.fmul(k.get(k.at(p, k.u32(c))), k.get(k.at(h, k.iadd(k.imul(k.u32(c), dim), d)))))
		}
		k.put(k.at(out, d), k.roundBF16(s))
	})
	return k.finish()
}

// hcPostKernel is dsv41.hcPost (hc = 4). Push dim, h, y, res, post, comb.
func hcPostKernel(modes floatModes) []uint32 {
	k := newKernel(modes, flatGroup, 1)
	i := k.flat()
	dim, h, y, res, post, comb := k.p(0), k.p(1), k.p(2), k.p(3), k.p(4), k.p(5)
	k.ifThen(k.ult(i, k.imul(dim, k.u32(4))), func() {
		c, d := k.floorDiv(i, dim), k.umod(i, dim)
		s := k.f32(0)
		for j := range uint32(4) {
			cw := k.get(k.at(comb, k.iadd(k.u32(4*j), c)))
			s = k.fadd(s, k.fmul(cw, k.get(k.at(res, k.iadd(k.imul(k.u32(j), dim), d)))))
		}
		k.put(k.at(h, i), k.roundBF16(k.fadd(k.fmul(k.get(k.at(post, c)), k.get(k.at(y, d))), s)))
	})
	return k.finish()
}

// ropeDSKernel is dsv41's rope.apply on segs segments of 2·pairs floats,
// stride apart, with cos and sin interleaved at cs. Push pairs, segs, x,
// stride, cs, inverse.
func ropeDSKernel(modes floatModes) []uint32 {
	k := newKernel(modes, flatGroup, 1)
	i := k.flat()
	pairs, segs, x, stride, cs, inverse := k.p(0), k.p(1), k.p(2), k.p(3), k.p(4), k.p(5)
	k.ifThen(k.ult(i, k.imul(pairs, segs)), func() {
		seg, pr := k.floorDiv(i, pairs), k.umod(i, pairs)
		off := k.iadd(k.iadd(x, k.imul(seg, stride)), k.imul(pr, k.u32(2)))
		a, bb := k.get(off), k.get(k.iadd(off, k.u32(1)))
		c := k.get(k.at(cs, k.imul(pr, k.u32(2))))
		s := k.get(k.at(cs, k.iadd(k.imul(pr, k.u32(2)), k.u32(1))))
		s = k.selF(k.ieq(inverse, k.u32(0)), s, k.fneg(s))
		k.put(off, k.roundBF16(k.fsub(k.fmul(a, c), k.fmul(bb, s))))
		k.put(k.iadd(off, k.u32(1)), k.roundBF16(k.fadd(k.fmul(a, s), k.fmul(bb, c))))
	})
	return k.finish()
}

// actScale is the scale of a block of 32 for act_quant: the power of two
// at least max(amax, 1e-4)/448. vals are the block's floats.
func (k *kbuilder) actScale(vals []id) (s, inv id) {
	amax := k.f32(0)
	for _, v := range vals {
		amax = k.gomax(amax, k.fabs(v))
	}
	t := k.fmul(k.gomax(amax, k.f32(float32(1e-4))), k.f32(float32(1.0/448)))
	tb := k.toU(t)
	e := k.iadd(k.and(k.shr(tb, k.u32(23)), k.u32(0xff)), k.cnt(k.ine(k.and(tb, k.u32(0x7fffff)), k.u32(0))))
	return k.toF(k.shl(e, k.u32(23))), k.toF(k.shl(k.op(opISub, k.tU32, k.u32(254), e), k.u32(23)))
}

// actQuantKernel is num.ActQuantInPlace on n floats at off.
func actQuantKernel(modes floatModes) []uint32 {
	k := newKernel(modes, flatGroup, 1)
	blk := k.flat()
	n, off := k.p(0), k.p(1)
	k.ifThen(k.ult(blk, k.shr(n, k.u32(5))), func() {
		base := k.iadd(off, k.imul(blk, k.u32(32)))
		vals := make([]id, 32)
		for j := range vals {
			vals[j] = k.get(k.iadd(base, k.u32(uint32(j))))
		}
		s, inv := k.actScale(vals)
		for j, v := range vals {
			x := k.fmul(v, inv)
			x = k.selF(k.fgt(x, k.f32(448)), k.f32(448), k.selF(k.flt(x, k.f32(-448)), k.f32(-448), x))
			k.put(k.iadd(base, k.u32(uint32(j))), k.roundBF16(k.fmul(k.e4m3F(k.encodeE4M3(x)), s)))
		}
	})
	return k.finish()
}

var fp4Grid = [8]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6}

// encodeE2M1 is num.EncodeE2M1 of v within ±6.
func (b *builder) encodeE2M1(v id) id {
	a := b.fabs(v)
	code := b.u32(0)
	for i := range 7 {
		mid := (fp4Grid[i] + fp4Grid[i+1]) / 2
		pass := b.fgt(a, b.f32(mid))
		if (i+1)%2 == 0 {
			pass = b.lor(pass, b.feq(a, b.f32(mid)))
		}
		code = b.sel(pass, b.u32(uint32(i+1)), code)
	}
	return b.or(code, b.shr(b.and(b.toU(v), b.u32(0x80000000)), b.u32(28)))
}

// fp4QuantKernel is num.FP4QuantInPlace on n floats at off: kind 0 is
// FP4ScaleE8M0 (blocks of 32), 1 FP4ScaleE4M3 (blocks of 16). Push n, off,
// kind.
func fp4QuantKernel(modes floatModes) []uint32 {
	k := newKernel(modes, flatGroup, 1)
	blk := k.flat()
	n, off, kind := k.p(0), k.p(1), k.p(2)
	e4m3 := k.ine(kind, k.u32(0))
	size := k.sel(e4m3, k.u32(16), k.u32(32))
	k.ifThen(k.ult(blk, k.floorDiv(n, size)), func() {
		base := k.iadd(off, k.imul(blk, size))
		vals := make([]id, 32)
		amax := k.f32(0)
		for j := range vals {
			idx := k.iadd(base, k.u32(uint32(j)))
			if j >= 16 {
				idx = k.sel(e4m3, base, idx) // not part of the block: read anything
			}
			vals[j] = k.get(idx)
			if j < 16 {
				amax = k.gomax(amax, k.fabs(vals[j]))
			} else {
				amax = k.selF(e4m3, amax, k.gomax(amax, k.fabs(vals[j])))
			}
		}
		// FP4ScaleE4M3: E4M3(max(amax, 6/512)/6); FP4ScaleE8M0: pow2Ceil(max(amax, 6·2^-126)·(1/6)).
		sE4 := k.e4m3F(k.encodeE4M3(k.fdiv(k.gomax(amax, k.f32(6.0/512)), k.f32(6))))
		t := k.fmul(k.gomax(amax, k.f32(float32(math.Ldexp(6, -126)))), k.f32(float32(1.0/6)))
		tb := k.toU(t)
		e := k.iadd(k.and(k.shr(tb, k.u32(23)), k.u32(0xff)), k.cnt(k.ine(k.and(tb, k.u32(0x7fffff)), k.u32(0))))
		s := k.selF(e4m3, sE4, k.toF(k.shl(e, k.u32(23))))
		for j, v := range vals {
			write := func() {
				x := k.fdiv(v, s)
				x = k.gomin(k.gomax(x, k.f32(-6)), k.f32(6))
				g := k.e2m1F(k.encodeE2M1(x))
				k.put(k.iadd(base, k.u32(uint32(j))), k.roundBF16(k.fmul(g, s)))
			}
			if j < 16 {
				write()
			} else {
				k.ifThen(k.ieq(kind, k.u32(0)), write)
			}
		}
	})
	return k.finish()
}

// roundScaleKernel sets n floats at off to bf16(x·scale). Push n, off, scale.
func roundScaleKernel(modes floatModes) []uint32 {
	k := newKernel(modes, flatGroup, 1)
	i := k.flat()
	n, off := k.p(0), k.p(1)
	k.ifThen(k.ult(i, n), func() {
		k.put(k.at(off, i), k.roundBF16(k.fmul(k.get(k.at(off, i)), k.pf(2))))
	})
	return k.finish()
}

// compressKernel is dsv41's compressor pooling of a full group: per
// channel a softmax over the ratio tokens' scores weighs their KV. Slots
// are hd apart. Push hd, ratio, kv, score, out.
func compressKernel(modes floatModes) []uint32 {
	k := newKernel(modes, flatGroup, 1)
	d := k.flat()
	hd, ratio, kv, sc, out := k.p(0), k.p(1), k.p(2), k.p(3), k.p(4)
	k.ifThen(k.ult(d, hd), func() {
		slot := func(base, j id) id { return k.get(k.at(base, k.iadd(k.imul(j, hd), d))) }
		m := k.local(k.tF32)
		k.store(m, k.toF(k.u32(negInfBits)))
		k.forRange(k.u32(0), ratio, k.u32(1), func(j id) {
			k.store(m, k.gomax(k.load(k.tF32, m), slot(sc, j)))
		})
		z := k.local(k.tF32)
		k.store(z, k.f32(0))
		k.forRange(k.u32(0), ratio, k.u32(1), func(j id) {
			k.store(z, k.fadd(k.load(k.tF32, z), k.exp32(k.fsub(slot(sc, j), k.load(k.tF32, m)))))
		})
		v := k.local(k.tF32)
		k.store(v, k.f32(0))
		k.forRange(k.u32(0), ratio, k.u32(1), func(j id) {
			e := k.exp32(k.fsub(slot(sc, j), k.load(k.tF32, m)))
			k.store(v, k.fadd(k.load(k.tF32, v), k.fmul(slot(kv, j), k.fdiv(e, k.load(k.tF32, z)))))
		})
		k.put(k.at(out, d), k.roundBF16(k.load(k.tF32, v)))
	})
	return k.finish()
}

// idxScoresKernel is the indexer's scores of keys 0..width-1, -inf from
// visible on, and where a candidate pool (flags at cand, candLen of them)
// excludes them if mask. Push width, visible, q, keys, w, heads, dim, out,
// cand, candLen, mask.
func idxScoresKernel(modes floatModes) []uint32 {
	k := newKernel(modes, flatGroup, 1)
	t := k.flat()
	width, visible, q, keys, w, heads, dim, out := k.p(0), k.p(1), k.p(2), k.p(3), k.p(4), k.p(5), k.p(6), k.p(7)
	cand, candLen, mask := k.p(8), k.p(9), k.p(10)
	k.ifThen(k.ult(t, width), func() {
		res := k.local(k.tF32)
		k.store(res, k.toF(k.u32(negInfBits)))
		k.ifThen(k.ult(t, visible), func() {
			kt := k.iadd(keys, k.imul(t, dim))
			sum := k.local(k.tF32)
			k.store(sum, k.f32(0))
			k.forRange(k.u32(0), heads, k.u32(1), func(h id) {
				qh := k.iadd(q, k.imul(h, dim))
				d := k.roundBF16(k.dot32(dim, func(i id) id { return k.get(k.at(qh, i)) }, func(i id) id { return k.get(k.at(kt, i)) }))
				term := k.roundBF16(k.fmul(k.gomax(d, k.f32(0)), k.get(k.at(w, h))))
				k.store(sum, k.fadd(k.load(k.tF32, sum), term))
			})
			k.store(res, k.roundBF16(k.load(k.tF32, sum)))
		})
		excluded := k.land(k.ine(mask, k.u32(0)), k.lor(k.uge(t, candLen), k.ieq(k.iget(k.at(cand, t)), k.u32(0))))
		k.put(k.at(out, t), k.selF(excluded, k.toF(k.u32(negInfBits)), k.load(k.tF32, res)))
	})
	return k.finish()
}

// beats reports whether value vj at j ranks before vi at i: larger, or
// equal and at a lower index (dsv41.topK's order).
func (k *kbuilder) beats(vj, j, vi, i id) id {
	return k.lor(k.fgt(vj, vi), k.land(k.feq(vj, vi), k.ult(j, i)))
}

// rank counts the values of n at vals that rank before value v at i.
func (k *kbuilder) rank(vals, n, v, i id) id {
	r := k.local(k.tU32)
	k.store(r, k.u32(0))
	k.forRange(k.u32(0), n, k.u32(1), func(j id) {
		k.store(r, k.iadd(k.load(k.tU32, r), k.cnt(k.beats(k.get(k.at(vals, j)), j, v, i))))
	})
	return k.load(k.tU32, r)
}

// topKKernel selects the k largest of n values (dsv41.topK). Mode 0
// writes the indices by rank at out. Mode 1 flags the selected ones at
// flags; mode 2 writes the flagged ones in ascending order at out, -1 for
// those from visible on (the indexer's selection). Push n, k, vals, out,
// mode, flags, visible.
func topKKernel(modes floatModes) []uint32 {
	k := newKernel(modes, flatGroup, 1)
	i := k.flat()
	n, kk, vals, out, mode, flags, visible := k.p(0), k.p(1), k.p(2), k.p(3), k.p(4), k.p(5), k.p(6)
	k.ifThen(k.ult(i, n), func() {
		k.ifThen(k.ult(mode, k.u32(2)), func() {
			r := k.rank(vals, n, k.get(k.at(vals, i)), i)
			sel := k.ult(r, kk)
			k.ifThen(k.ieq(mode, k.u32(0)), func() {
				k.ifThen(sel, func() { k.iput(k.at(out, r), i) })
			})
			k.ifThen(k.ieq(mode, k.u32(1)), func() { k.iput(k.at(flags, i), k.cnt(sel)) })
		})
		k.ifThen(k.ieq(mode, k.u32(2)), func() {
			k.ifThen(k.ine(k.iget(k.at(flags, i)), k.u32(0)), func() {
				pos := k.local(k.tU32)
				k.store(pos, k.u32(0))
				k.forRange(k.u32(0), i, k.u32(1), func(j id) {
					k.store(pos, k.iadd(k.load(k.tU32, pos), k.iget(k.at(flags, j))))
				})
				k.iput(k.at(out, k.load(k.tU32, pos)), k.sel(k.uge(i, visible), k.u32(0xffffffff), i))
			})
		})
	})
	return k.finish()
}

// candPoolKernel is dsv41.candidatePool in three phases over the scores
// of width positions: 0, the best score of each block of size (n of them)
// at bs, +inf for the newest visible position's; 1, flags at sel for the
// best k blocks that have a score; 2, a flag per position at keep. Push
// width, visible, k, size, scores, bs, sel, keep, phase.
func candPoolKernel(modes floatModes) []uint32 {
	k := newKernel(modes, flatGroup, 1)
	i := k.flat()
	width, visible, kk, size, scores, bs, sel, keep, phase := k.p(0), k.p(1), k.p(2), k.p(3), k.p(4), k.p(5), k.p(6), k.p(7), k.p(8)
	n := k.floorDiv(k.iadd(width, k.op(opISub, k.tU32, size, k.u32(1))), size)
	k.ifThen(k.ieq(phase, k.u32(0)), func() {
		k.ifThen(k.ult(i, n), func() {
			m := k.local(k.tF32)
			k.store(m, k.toF(k.u32(negInfBits)))
			end := k.iadd(k.imul(i, size), size)
			end = k.sel(k.ult(width, end), width, end)
			k.forRange(k.imul(i, size), end, k.u32(1), func(t id) {
				k.store(m, k.gomax(k.load(k.tF32, m), k.get(k.at(scores, t))))
			})
			newest := k.land(k.ugt(visible, k.u32(0)), k.ieq(i, k.floorDiv(k.op(opISub, k.tU32, visible, k.u32(1)), size)))
			k.put(k.at(bs, i), k.selF(newest, k.toF(k.u32(0x7f800000)), k.load(k.tF32, m)))
		})
	})
	k.ifThen(k.ieq(phase, k.u32(1)), func() {
		k.ifThen(k.ult(i, n), func() {
			v := k.get(k.at(bs, i))
			kn := k.sel(k.ult(kk, n), kk, n)
			chosen := k.land(k.ult(k.rank(bs, n, v, i), kn), k.op(opLogicalNot, k.tBool, k.feq(v, k.toF(k.u32(negInfBits)))))
			k.iput(k.at(sel, i), k.cnt(chosen))
		})
	})
	k.ifThen(k.ieq(phase, k.u32(2)), func() {
		k.ifThen(k.ult(i, width), func() {
			k.iput(k.at(keep, i), k.iget(k.at(sel, k.floorDiv(i, size))))
		})
	})
	return k.finish()
}

// sparseAttnKernel is dsv41's sparseAttention, a workgroup of 64 per
// head. Rows: nWin word offsets at rows, then nTop compressed entries
// selected at topk (indices into the cache at comp, hd apart); -1 is an
// empty slot. acc holds hd floats per head. Push heads, hd, q, rows, nWin,
// topk, nTop, comp, sink, scale, acc, out.
func sparseAttnKernel(modes floatModes) []uint32 {
	k := newKernel(modes, 64, 1)
	sc := k.workgroupFloats(64)
	state := k.workgroupFloats(3) // m, sum, rescale
	lid, wid := k.ids()
	tid, h := k.component(lid, 0), k.component(wid, 0)
	hd, q, rows, nWin, topk, nTop, comp := k.p(1), k.p(2), k.p(3), k.p(4), k.p(5), k.p(6), k.p(7)
	sink, scale, acc, out := k.p(8), k.pf(9), k.p(10), k.p(11)
	total := k.iadd(nWin, nTop)
	qh := k.iadd(q, k.imul(h, hd))
	acch := k.iadd(acc, k.imul(h, hd))
	rowAt := func(j id) id {
		win := k.iget(k.at(rows, k.sel(k.ult(j, nWin), j, k.u32(0))))
		g := k.iget(k.at(topk, k.sel(k.ult(j, nWin), k.u32(0), k.op(opISub, k.tU32, j, nWin))))
		c := k.sel(k.isNone(g), g, k.iadd(comp, k.imul(g, hd)))
		return k.sel(k.ult(j, nWin), win, c)
	}
	k.forRange(tid, hd, k.u32(64), func(d id) { k.put(k.at(acch, d), k.f32(0)) })
	k.ifThen(k.ieq(tid, k.u32(0)), func() {
		k.store(k.shared(state, k.u32(0)), k.f32(-1e30))
		k.store(k.shared(state, k.u32(1)), k.f32(0))
	})
	k.barrier()
	blocks := k.shr(k.iadd(total, k.u32(63)), k.u32(6))
	k.forRange(k.u32(0), blocks, k.u32(1), func(blk id) {
		first := k.imul(blk, k.u32(64))
		count := k.op(opISub, k.tU32, total, first)
		count = k.sel(k.ult(k.u32(64), count), k.u32(64), count)
		// Each thread scores a row.
		s := k.local(k.tF32)
		k.store(s, k.toF(k.u32(negInfBits)))
		k.ifThen(k.ult(tid, count), func() {
			r := rowAt(k.iadd(first, tid))
			k.ifThen(k.op(opLogicalNot, k.tBool, k.isNone(r)), func() {
				d := k.dot32(hd, func(i id) id { return k.get(k.at(qh, i)) }, func(i id) id { return k.get(k.at(r, i)) })
				k.store(s, k.fmul(d, scale))
			})
		})
		k.store(k.shared(sc, tid), k.load(k.tF32, s))
		k.barrier()
		// One thread: the block's max, the new running max, exps and sums.
		k.ifThen(k.ieq(tid, k.u32(0)), func() {
			bm := k.local(k.tF32)
			k.store(bm, k.toF(k.u32(negInfBits)))
			k.forRange(k.u32(0), count, k.u32(1), func(j id) {
				r := rowAt(k.iadd(first, j))
				k.ifThen(k.op(opLogicalNot, k.tBool, k.isNone(r)), func() {
					k.store(bm, k.gomax(k.load(k.tF32, bm), k.load(k.tF32, k.shared(sc, j))))
				})
			})
			prev := k.load(k.tF32, k.shared(state, k.u32(0)))
			m := k.gomax(prev, k.load(k.tF32, bm))
			rescale := k.exp32(k.fsub(prev, m))
			bsum := k.local(k.tF32)
			k.store(bsum, k.f32(0))
			k.forRange(k.u32(0), count, k.u32(1), func(j id) {
				e := k.exp32(k.fsub(k.load(k.tF32, k.shared(sc, j)), m))
				k.store(k.shared(sc, j), e)
				k.store(bsum, k.fadd(k.load(k.tF32, bsum), e))
			})
			sum := k.load(k.tF32, k.shared(state, k.u32(1)))
			k.store(k.shared(state, k.u32(1)), k.fadd(k.fmul(sum, rescale), k.load(k.tF32, bsum)))
			k.store(k.shared(state, k.u32(0)), m)
			k.store(k.shared(state, k.u32(2)), rescale)
		})
		k.barrier()
		rescale := k.load(k.tF32, k.shared(state, k.u32(2)))
		k.forRange(tid, hd, k.u32(64), func(d id) {
			v := k.local(k.tF32)
			k.store(v, k.f32(0))
			k.forRange(k.u32(0), count, k.u32(1), func(j id) {
				r := rowAt(k.iadd(first, j))
				k.ifThen(k.op(opLogicalNot, k.tBool, k.isNone(r)), func() {
					p := k.fmul(k.roundBF16(k.load(k.tF32, k.shared(sc, j))), k.get(k.at(r, d)))
					k.store(v, k.fadd(k.load(k.tF32, v), p))
				})
			})
			k.put(k.at(acch, d), k.fadd(k.fmul(k.get(k.at(acch, d)), rescale), k.load(k.tF32, v)))
		})
		k.barrier()
	})
	m := k.load(k.tF32, k.shared(state, k.u32(0)))
	sum := k.fadd(k.load(k.tF32, k.shared(state, k.u32(1))), k.exp32(k.fsub(k.get(k.at(sink, h)), m)))
	k.forRange(tid, hd, k.u32(64), func(d id) {
		k.put(k.at(k.iadd(out, k.imul(h, hd)), d), k.roundBF16(k.fdiv(k.get(k.at(acch, d)), sum)))
	})
	return k.finish()
}

// moeScoreKernel is the gate's scores of E experts: sqrt(softplus(z)) (fn
// 0) or sigmoid(z) (1), and those plus the selection bias. Push E, z,
// bias, fn, score, biased.
func moeScoreKernel(modes floatModes) []uint32 {
	k := newKernel(modes, flatGroup, 1)
	e := k.flat()
	n, z, bias, fn, score, biased := k.p(0), k.p(1), k.p(2), k.p(3), k.p(4), k.p(5)
	k.ifThen(k.ult(e, n), func() {
		v := k.get(k.at(z, e))
		s := k.selF(k.ieq(fn, k.u32(0)), k.fsqrt(k.softplus(v)), k.sigmoid(v))
		k.put(k.at(score, e), s)
		k.put(k.at(biased, e), k.fadd(s, k.get(k.at(bias, e))))
	})
	return k.finish()
}

// moeWeightsKernel turns the selection (indices by rank at sel) into the
// experts in ascending order (at order) and their weights (at w): scores
// normalized over the selection, if norm and k > 1, then scaled. One
// thread. Push k, score, sel, norm, scale, order, w.
func moeWeightsKernel(modes floatModes) []uint32 {
	k := newKernel(modes, 1, 1)
	kk, score, sel, norm, scale, order, w := k.p(0), k.p(1), k.p(2), k.p(3), k.pf(4), k.p(5), k.p(6)
	sum := k.local(k.tF32)
	k.store(sum, k.f32(0))
	k.forRange(k.u32(0), kk, k.u32(1), func(i id) {
		k.store(sum, k.fadd(k.load(k.tF32, sum), k.get(k.at(score, k.iget(k.at(sel, i))))))
	})
	doNorm := k.land(k.ine(norm, k.u32(0)), k.ugt(kk, k.u32(1)))
	den := k.fadd(k.load(k.tF32, sum), k.f32(1e-20))
	// The expert ranked i goes to place = how many selected are below it.
	k.forRange(k.u32(0), kk, k.u32(1), func(i id) {
		e := k.iget(k.at(sel, i))
		place := k.local(k.tU32)
		k.store(place, k.u32(0))
		k.forRange(k.u32(0), kk, k.u32(1), func(j id) {
			k.store(place, k.iadd(k.load(k.tU32, place), k.cnt(k.ult(k.iget(k.at(sel, j)), e))))
		})
		wi := k.get(k.at(score, e))
		wi = k.selF(doNorm, k.fdiv(wi, den), wi)
		p := k.load(k.tU32, place)
		k.iput(k.at(order, p), e)
		k.put(k.at(w, p), k.fmul(wi, scale))
	})
	return k.finish()
}

// siluMulDSKernel is dsv41's expert activation on slots of n floats,
// stride apart: g = bf16(weight·(silu(min(g, limit))·clamp(u, ±limit))),
// unweighted if not weighted. Push n, g, u, limit, weighted, w, slots,
// stride.
func siluMulDSKernel(modes floatModes) []uint32 {
	k := newKernel(modes, flatGroup, 1)
	i := k.flat()
	n, g, u, limit, weighted, w, slots, stride := k.p(0), k.p(1), k.p(2), k.pf(3), k.p(4), k.p(5), k.p(6), k.p(7)
	k.ifThen(k.ult(i, k.imul(n, slots)), func() {
		s, j := k.floorDiv(i, n), k.umod(i, n)
		gi := k.at(g, k.iadd(k.imul(s, stride), j))
		ui := k.at(u, k.iadd(k.imul(s, stride), j))
		gv, uv := k.get(gi), k.get(ui)
		has := k.fgt(limit, k.f32(0))
		uv = k.selF(has, k.gomin(k.gomax(uv, k.fneg(limit)), limit), uv)
		gv = k.selF(has, k.gomin(gv, limit), gv)
		v := k.fmul(k.silu(gv), uv)
		v = k.selF(k.ine(weighted, k.u32(0)), k.fmul(k.get(k.at(w, s)), v), v)
		k.put(gi, k.roundBF16(v))
	})
	return k.finish()
}

// moeSumKernel adds the experts' outputs (k slots, stride apart) in order,
// then the shared expert's: y = bf16(Σ + shared). Push dim, outs, k,
// stride, shared, y.
func moeSumKernel(modes floatModes) []uint32 {
	k := newKernel(modes, flatGroup, 1)
	d := k.flat()
	dim, outs, kk, stride, shared, y := k.p(0), k.p(1), k.p(2), k.p(3), k.p(4), k.p(5)
	k.ifThen(k.ult(d, dim), func() {
		s := k.local(k.tF32)
		k.store(s, k.f32(0))
		k.forRange(k.u32(0), kk, k.u32(1), func(j id) {
			k.store(s, k.fadd(k.load(k.tF32, s), k.get(k.at(outs, k.iadd(k.imul(j, stride), d)))))
		})
		k.put(k.at(y, d), k.roundBF16(k.fadd(k.load(k.tF32, s), k.get(k.at(shared, d)))))
	})
	return k.finish()
}

// engramGateKernel is the end of dsv41's engram.apply, a workgroup of 32
// per stream copy c: with x = h_c, key = kv_c, value = kv_hc, the gate
// g = sigmoid(±sqrt(max(|dot|, 1e-6))) of dot = sum32(x·qk_c·key)·rstd·
// scale, and x += g·value in bf16. Push dim, h, kv, qk, eps, scale.
func engramGateKernel(modes floatModes) []uint32 {
	k := newKernel(modes, laneCount, 1)
	sums32, sums8 := k.workgroupFloats(laneCount), k.workgroupFloats(8)
	shared := k.workgroupFloats(3)
	lid, wid := k.ids()
	lane, c := k.component(lid, 0), k.component(wid, 0)
	dim, h, kv, qk, eps, scale := k.p(0), k.p(1), k.p(2), k.p(3), k.pf(4), k.pf(5)
	x := k.iadd(h, k.imul(c, dim))
	key := k.iadd(kv, k.imul(c, dim))
	w := k.iadd(qk, k.imul(c, dim))
	value := k.iadd(kv, k.imul(k.u32(4), dim))
	xv := func(i id) id { return k.get(k.at(x, i)) }
	kvv := func(i id) id { return k.get(k.at(key, i)) }
	msInto := func(slot uint32, f func(i id) id) {
		k.store(k.shared(sums32, lane), k.laneDot(lane, dim, f, f))
		k.reduce32(lane, k.u32(0), sums32, sums8, func(v id) {
			k.store(k.shared(shared, k.u32(slot)), k.fdiv(v, k.op(opConvertUToF, k.tF32, dim)))
		})
		k.barrier()
	}
	msInto(0, xv)
	msInto(1, kvv)
	// sum32 of x·w·key: lane j's elements j, j+32, ..., then halves.
	acc := k.local(k.tF32)
	k.store(acc, k.f32(0))
	k.forRange(lane, dim, k.u32(32), func(i id) {
		k.store(acc, k.fadd(k.load(k.tF32, acc), k.fmul(k.fmul(xv(i), k.get(k.at(w, i))), kvv(i))))
	})
	k.store(k.shared(sums32, lane), k.load(k.tF32, acc))
	k.barrier()
	k.ifThen(k.ieq(lane, k.u32(0)), func() {
		var s [32]id
		for j := range s {
			s[j] = k.load(k.tF32, k.shared(sums32, k.u32(uint32(j))))
		}
		for half := 16; half >= 1; half /= 2 {
			for j := range half {
				s[j] = k.fadd(s[j], s[j+half])
			}
		}
		msx := k.load(k.tF32, k.shared(shared, k.u32(0)))
		msk := k.load(k.tF32, k.shared(shared, k.u32(1)))
		rstd := k.fmul(k.rsqrtDS(k.fadd(msx, eps)), k.rsqrtDS(k.fadd(msk, eps)))
		dot := k.fmul(k.fmul(s[0], rstd), scale)
		root := k.fsqrt(k.gomax(k.fabs(dot), k.f32(float32(1e-6))))
		signed := k.toF(k.or(k.toU(root), k.and(k.toU(dot), k.u32(0x80000000))))
		k.store(k.shared(shared, k.u32(2)), k.sigmoid(signed))
	})
	k.barrier()
	g := k.load(k.tF32, k.shared(shared, k.u32(2)))
	k.forRange(lane, dim, k.u32(32), func(i id) {
		k.put(k.at(x, i), k.roundBF16(k.fadd(xv(i), k.fmul(g, k.get(k.at(value, i))))))
	})
	return k.finish()
}
