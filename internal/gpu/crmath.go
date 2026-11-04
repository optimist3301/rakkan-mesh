package gpu

import "math"

// Correctly rounded float32 division and square root, which Vulkan does
// not promise (it allows division 2.5 ulp off): the CPU's a/b and
// math.Sqrt, bit for bit, from 32-bit integer operations only. Long
// division and the digit-by-digit square root of the mantissas give 25 or
// 26 bits and a remainder; roundPack rounds them to nearest, ties to even,
// into a normal or denormal float32. With them, num.Log32, num.Log1p32,
// and DeepSeek-V4.1's sigmoid, SiLU and softplus.

func (b *builder) uge(x, y id) id { return b.op(opLogicalNot, b.tBool, b.ult(x, y)) }

const (
	opLogicalNot = 168
	opBitwiseXor = 198
)

// roundPack rounds q·2^e (q < 2^27 with its highest bit n >= 23, more
// bits below if sticky) to a float32 with sign (0 or 1<<31).
func (b *builder) roundPack(q, n, e, sticky, sign id) id {
	be := b.iaddS(b.iaddS(b.toI(n), e), b.i32(127)) // the biased exponent if normal
	normal := b.sgt(be, b.i32(0))
	field := b.selI(normal, be, b.i32(1))
	// Bits dropped: n-23, and as many more as the exponent is below 1.
	d := b.iaddS(b.isubS(b.toI(n), b.i32(23)), b.isubS(field, be))
	d = b.selI(b.sgt(d, b.i32(27)), b.i32(27), d)
	du := b.toU(d)
	mant := b.shr(q, du)
	rem := b.and(q, b.op(opISub, b.tU32, b.shl(b.u32(1), du), b.u32(1)))
	half := b.sel(b.ieq(du, b.u32(0)), b.u32(0), b.shl(b.u32(1), b.op(opISub, b.tU32, du, b.u32(1))))
	odd := b.ieq(b.and(mant, b.u32(1)), b.u32(1))
	exactHalf := b.land(b.ieq(rem, half), b.ine(du, b.u32(0)))
	up := b.lor(b.ugt(rem, half), b.land(exactHalf, b.lor(sticky, odd)))
	up = b.land(up, b.ine(du, b.u32(0)))
	mant = b.iadd(mant, b.sel(up, b.u32(1), b.u32(0)))
	// The implicit bit carries into the exponent field, and a rounding
	// carry with it.
	bits := b.iadd(b.shl(b.toU(b.isubS(field, b.i32(1))), b.u32(23)), mant)
	bits = b.sel(b.uge(bits, b.u32(0x7f800000)), b.u32(0x7f800000), bits)
	return b.toF(b.or(bits, sign))
}

// unpack splits a finite nonzero |x|'s bits into m in [2^23, 2^24) and e,
// x = m·2^(e-150).
func (b *builder) unpack(bits id) (m, e id) {
	ex := b.and(b.shr(bits, b.u32(23)), b.u32(0xff))
	frac := b.and(bits, b.u32(0x7fffff))
	den := b.ieq(ex, b.u32(0))
	sh := b.op(opISub, b.tU32, b.u32(23), b.msb(b.sel(den, b.or(frac, b.u32(1)), b.u32(1<<23))))
	m = b.sel(den, b.shl(frac, sh), b.or(frac, b.u32(1<<23)))
	e = b.selI(den, b.isubS(b.i32(1), b.toI(sh)), b.toI(ex))
	return m, e
}

const canonicalNaN = 0x7fc00000

// fdiv is x/y, correctly rounded.
func (b *builder) fdiv(x, y id) id {
	ux, uy := b.toU(x), b.toU(y)
	sign := b.and(b.op(opBitwiseXor, b.tU32, ux, uy), b.u32(0x80000000))
	ax, ay := b.and(ux, b.u32(0x7fffffff)), b.and(uy, b.u32(0x7fffffff))
	mx, ex := b.unpack(ax)
	my, ey := b.unpack(ay)
	q, r := b.u32(0), mx
	for range 26 {
		ge := b.uge(r, my)
		r = b.sel(ge, b.op(opISub, b.tU32, r, my), r)
		q = b.or(b.shl(q, b.u32(1)), b.sel(ge, b.u32(1), b.u32(0)))
		r = b.shl(r, b.u32(1))
	}
	// q = floor(mx/my·2^25): x/y = q·2^(ex-ey-25).
	v := b.roundPack(q, b.msb(q), b.isubS(b.isubS(ex, ey), b.i32(25)), b.ine(r, b.u32(0)), sign)
	inf := b.u32(0x7f800000)
	xZero, yZero := b.ieq(ax, b.u32(0)), b.ieq(ay, b.u32(0))
	xInf, yInf := b.ieq(ax, inf), b.ieq(ay, inf)
	nan := b.lor(b.lor(b.ugt(ax, inf), b.ugt(ay, inf)), b.lor(b.land(xZero, yZero), b.land(xInf, yInf)))
	v = b.selF(b.lor(xZero, yInf), b.toF(sign), v)
	v = b.selF(b.lor(xInf, yZero), b.toF(b.or(sign, inf)), v)
	return b.selF(nan, b.toF(b.u32(canonicalNaN)), v)
}

// fsqrt is math.Sqrt in float32, correctly rounded.
func (b *builder) fsqrt(x id) id {
	ux := b.toU(x)
	ax := b.and(ux, b.u32(0x7fffffff))
	m, e := b.unpack(ax)
	t := b.isubS(e, b.i32(150))
	odd := b.ieq(b.and(b.toU(t), b.u32(1)), b.u32(1))
	m = b.sel(odd, b.shl(m, b.u32(1)), m)
	t = b.selI(odd, b.isubS(t, b.i32(1)), t)
	// The root of m·2^26, two bits at a time: 13 pairs of m, 13 of zeros.
	root, rem := b.u32(0), b.u32(0)
	for j := 25; j >= 0; j-- {
		pair := b.u32(0)
		if s := 2*j - 26; s >= 0 {
			pair = b.and(b.shr(m, b.u32(uint32(s))), b.u32(3))
		}
		rem = b.or(b.shl(rem, b.u32(2)), pair)
		trial := b.or(b.shl(root, b.u32(2)), b.u32(1))
		ge := b.uge(rem, trial)
		rem = b.sel(ge, b.op(opISub, b.tU32, rem, trial), rem)
		root = b.or(b.shl(root, b.u32(1)), b.sel(ge, b.u32(1), b.u32(0)))
	}
	// sqrt(x) = root·2^(t/2-13).
	v := b.roundPack(root, b.msb(root), b.isubS(b.sar(t, b.u32(1)), b.i32(13)), b.ine(rem, b.u32(0)), b.u32(0))
	inf := b.u32(0x7f800000)
	v = b.selF(b.ieq(ax, inf), x, v)
	v = b.selF(b.ieq(ax, b.u32(0)), x, v)
	neg := b.land(b.ine(b.and(ux, b.u32(0x80000000)), b.u32(0)), b.ine(ax, b.u32(0)))
	return b.selF(b.lor(neg, b.ugt(ax, inf)), b.toF(b.u32(canonicalNaN)), v)
}

// log32 is num.Log32 for positive finite x.
func (b *builder) log32(x id) id {
	bits := b.toU(x)
	den := b.ieq(b.shr(bits, b.u32(23)), b.u32(0))
	bits = b.sel(den, b.toU(b.fmul(x, b.f32(1<<24))), bits)
	ki := b.isubS(b.toI(b.shr(bits, b.u32(23))), b.selI(den, b.i32(127+24), b.i32(127)))
	f1 := b.toF(b.or(b.and(bits, b.u32(0x7fffff)), b.u32(127<<23)))
	big := b.fgt(f1, b.f32(math.Sqrt2))
	f1 = b.selF(big, b.fmul(f1, b.f32(0.5)), f1)
	ki = b.selI(big, b.iaddS(ki, b.i32(1)), ki)
	f := b.fsub(f1, b.f32(1))
	k := b.sToF(ki)
	s := b.fdiv(f, b.fadd(b.f32(2), f))
	z := b.fmul(s, s)
	w := b.fmul(z, z)
	t1 := b.fmul(w, b.fadd(b.f32(4.0000972152e-01), b.fmul(w, b.f32(2.4279078841e-01))))
	t2 := b.fmul(z, b.fadd(b.f32(6.6666662693e-01), b.fmul(w, b.f32(2.8498786688e-01))))
	r := b.fadd(t2, t1)
	hfsq := b.fmul(b.fmul(b.f32(0.5), f), f)
	inner := b.fsub(hfsq, b.fadd(b.fmul(s, b.fadd(hfsq, r)), b.fmul(k, b.f32(9.0580006145e-06))))
	return b.fsub(b.fmul(k, b.f32(6.9313812256e-01)), b.fsub(inner, f))
}

// log1p32 is num.Log1p32 for x >= 0.
func (b *builder) log1p32(x id) id {
	u := b.fadd(b.f32(1), x)
	c := b.fdiv(b.fsub(x, b.fsub(u, b.f32(1))), u)
	v := b.fadd(b.log32(u), c)
	v = b.selF(b.isInf(u), u, v)
	return b.selF(b.feq(u, b.f32(1)), x, v)
}

// sigmoid is 1/(1+e^-x), silu x/(1+e^-x), softplus log1p(e^x) or x above 20.
func (b *builder) sigmoid(x id) id {
	return b.fdiv(b.f32(1), b.fadd(b.f32(1), b.exp32(b.fneg(x))))
}

func (b *builder) silu(x id) id { return b.fdiv(x, b.fadd(b.f32(1), b.exp32(b.fneg(x)))) }

func (b *builder) softplus(x id) id {
	return b.selF(b.fgt(x, b.f32(20)), x, b.log1p32(b.exp32(x)))
}

// mathTestKernel evaluates the functions above on n pairs (a, b) at inOff
// and in2Off, writing 8 results per pair at outOff: a/b, sqrt|a|,
// 1/sqrt|a|, log32|a|, log1p32|a|, sigmoid a, silu a, softplus a. For tests.
func mathTestKernel(modes floatModes) []uint32 {
	k := newKernel(modes, flatGroup, 1)
	i := k.flat()
	n, inOff, in2Off, outOff := k.p(0), k.p(1), k.p(2), k.p(3)
	k.ifThen(k.ult(i, n), func() {
		a, c := k.get(k.at(inOff, i)), k.get(k.at(in2Off, i))
		abs := k.toF(k.and(k.toU(a), k.u32(0x7fffffff)))
		out := k.at(outOff, k.imul(i, k.u32(8)))
		root := k.fsqrt(abs)
		for j, v := range []id{k.fdiv(a, c), root, k.fdiv(k.f32(1), root), k.log32(abs), k.log1p32(abs), k.sigmoid(a), k.silu(a), k.softplus(a)} {
			k.put(k.iadd(out, k.u32(uint32(j))), v)
		}
	})
	return k.finish()
}
