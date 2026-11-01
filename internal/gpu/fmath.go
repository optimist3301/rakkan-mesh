package gpu

import (
	"math"
)

// Shader versions of package num's float functions, operation for
// operation: num.Exp32, num.Recip32, num.Rsqrt32, num.Canonical and Go's
// max. They use only correctly rounded additions and multiplications,
// integer operations and exact scalings, so they give the CPU's bits.

// fconst emits a float32 constant.
func (b *builder) fc(v float64) id { return b.f32(float32(v)) }

// pow2 returns 2^k for a signed k in [-126, 127].
func (b *builder) pow2(k id) id {
	return b.toF(b.shl(b.toU(b.iaddS(k, b.i32(127))), b.u32(23)))
}

// scale2 returns v·2^k rounded once (num.Scale2).
func (b *builder) scale2(v, k id) id {
	k1 := b.sdiv(k, b.i32(2))
	return b.fmul(b.fmul(v, b.pow2(k1)), b.pow2(b.isubS(k, k1)))
}

// roundEven rounds |v| < 2^22 to an integer, ties to even, as
// math.RoundToEven: adding and subtracting 1.5·2^23 leaves no fraction.
func (b *builder) roundEven(v id) id {
	m := b.f32(12582912)
	return b.fsub(b.fadd(v, m), m)
}

func (b *builder) canonical(v id) id {
	return b.selF(b.isNaN(v), b.toF(b.u32(0x7fc00000)), v)
}

// gomax is Go's max for floats: NaN if either is, +0 for -0 and +0.
func (b *builder) gomax(x, y id) id {
	eq := b.toF(b.and(b.toU(x), b.toU(y)))
	r := b.selF(b.fgt(x, y), x, b.selF(b.fgt(y, x), y, eq))
	return b.selF(b.lor(b.isNaN(x), b.isNaN(y)), b.toF(b.u32(0x7fc00000)), r)
}

// exp32 is num.Exp32.
func (b *builder) exp32(x id) id {
	n := b.roundEven(b.fmul(x, b.fc(1.4426950408889634)))
	r := b.fsub(x, b.fmul(n, b.fc(0.693145751953125)))
	r = b.fsub(r, b.fmul(n, b.fc(1.428606765330187e-06)))
	p := b.fc(1.0 / 720)
	for _, c := range []float64{1.0 / 120, 1.0 / 24, 1.0 / 6, 0.5, 1, 1} {
		p = b.fadd(b.fmul(p, r), b.fc(c))
	}
	// math.Ldexp, then rounding to float32: one rounding, as scale2.
	v := b.scale2(p, b.op(opConvertFToS, b.tI32, n))
	inf := b.toF(b.u32(0x7f800000))
	v = b.selF(b.fgt(x, b.fc(88.72)), inf, b.selF(b.flt(x, b.fc(-103.97)), b.f32(0), v))
	return b.selF(b.isNaN(x), b.toF(b.u32(0x7fc00000)), v)
}

// split returns m in [1, 2) and e with x = m·2^e, for positive finite x.
func (b *builder) split(x id) (m, e id) {
	bits := b.toU(x)
	sub := b.ieq(b.shr(bits, b.u32(23)), b.u32(0))
	bits = b.sel(sub, b.toU(b.fmul(x, b.fc(math.Ldexp(1, 64)))), bits)
	e0 := b.selI(sub, b.i32(-64), b.i32(0))
	e = b.iaddS(e0, b.isubS(b.toI(b.shr(bits, b.u32(23))), b.i32(127)))
	m = b.toF(b.or(b.and(bits, b.u32(0x7fffff)), b.u32(127<<23)))
	return m, e
}

// recip32 is num.Recip32.
func (b *builder) recip32(x id) id {
	sign := b.and(b.toU(x), b.u32(0x80000000))
	ax := b.toF(b.and(b.toU(x), b.u32(0x7fffffff)))
	m, e := b.split(ax)
	r := b.fsub(b.fc(24.0/17), b.fmul(b.fc(8.0/17), m))
	for range 3 {
		r = b.fadd(r, b.fmul(r, b.fsub(b.f32(1), b.fmul(m, r))))
	}
	negE := b.isubS(b.i32(0), e)
	far := b.lor(b.sgt(e, b.i32(126)), b.slt(e, b.i32(-127)))
	// Both branches are computed; pow2 of an out-of-range exponent is
	// garbage, but then it is not selected.
	near := b.fmul(r, b.pow2(b.selI(far, b.i32(0), negE)))
	r = b.selF(far, b.scale2(r, negE), near)
	r = b.toF(b.or(b.toU(r), sign))
	zero := b.feq(x, b.f32(0))
	r = b.selF(zero, b.toF(b.or(sign, b.u32(0x7f800000))), r)
	r = b.selF(b.isInf(x), b.toF(sign), r)
	return b.selF(b.isNaN(x), b.toF(b.u32(0x7fc00000)), r)
}

// rsqrt32 is num.Rsqrt32.
func (b *builder) rsqrt32(x id) id {
	m, e := b.split(x)
	odd := b.ieq(b.and(b.toU(e), b.u32(1)), b.u32(1))
	m = b.selF(odd, b.fmul(m, b.f32(2)), m)
	e = b.selI(odd, b.isubS(e, b.i32(1)), e)
	r := b.toF(b.op(opISub, b.tU32, b.u32(0x5f3759df), b.shr(b.toU(m), b.u32(1))))
	h := b.fmul(b.f32(0.5), m)
	for range 4 {
		t := b.fmul(b.fmul(h, r), r)
		r = b.fadd(r, b.fmul(r, b.fsub(b.f32(0.5), t)))
	}
	r = b.fmul(r, b.pow2(b.sdiv(b.isubS(b.i32(0), e), b.i32(2))))
	r = b.selF(b.feq(x, b.f32(0)), b.toF(b.u32(0x7f800000)), r)
	r = b.selF(b.isInf(x), b.f32(0), r)
	return b.selF(b.lor(b.isNaN(x), b.flt(x, b.f32(0))), b.toF(b.u32(0x7fc00000)), r)
}
