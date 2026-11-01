package num

import "math"

// Division and square roots for computations that must match on GPUs.
// IEEE division and sqrt are correctly rounded on CPUs, but Vulkan lets
// GPUs approximate them (division to 2.5 ulp), while it requires additions
// and multiplications to be correctly rounded. Recip32 and Rsqrt32 are
// therefore built from those, integer operations and exact scalings by
// powers of two: Newton iterations from a fixed first guess, a fixed number
// of times. They are accurate to about an ulp, not correctly rounded, and
// give the same bits on every processor that rounds additions and
// multiplications correctly. Exp32 already is such a function.

// Pow2 returns 2^k for k in [-126, 127], exactly.
func Pow2(k int) float32 { return math.Float32frombits(uint32(k+127) << 23) }

// Scale2 returns v·2^k, rounded once: v is first scaled by 2^(k/2), which
// for |v| in [2^-2, 2^2] cannot leave the normal range, then by the rest.
// k must be within [-252, 252].
func Scale2(v float32, k int) float32 {
	k1 := k / 2
	return float32(v*Pow2(k1)) * Pow2(k-k1)
}

// split returns m in [1, 2) and e with x = m·2^e, for positive finite x.
func split(x float32) (m float32, e int) {
	b := math.Float32bits(x)
	if b>>23 == 0 { // subnormal: scale into the normal range first
		b = math.Float32bits(x * Pow2(64))
		e = -64
	}
	e += int(b>>23) - 127
	return math.Float32frombits(b&0x7fffff | 127<<23), e
}

// Recip32 returns 1/x to within about an ulp.
func Recip32(x float32) float32 {
	switch {
	case x != x:
		return CanonicalNaN32
	case x == 0:
		return float32(math.Copysign(math.Inf(1), float64(x)))
	case math.IsInf(float64(x), 0):
		return float32(math.Copysign(0, float64(x)))
	}
	neg := x < 0
	if neg {
		x = -x
	}
	m, e := split(x)
	// 1/m for m in [1, 2): the line 24/17 - 8/17·m is within 1/17 of it,
	// and each step r += r·(1 - m·r) squares the relative error.
	r := 24.0/17 - float32(8.0/17*m)
	for range 3 {
		r = r + float32(r*(1-float32(m*r)))
	}
	if e > 126 || e < -127 { // the result is subnormal, zero or infinite
		r = Scale2(r, -e)
	} else {
		r = float32(r * Pow2(-e))
	}
	if neg {
		r = -r
	}
	return r
}

// Rsqrt32 returns 1/sqrt(x) to within about an ulp, for x >= 0.
func Rsqrt32(x float32) float32 {
	switch {
	case x != x || x < 0:
		return CanonicalNaN32
	case x == 0:
		return float32(math.Inf(1))
	case math.IsInf(float64(x), 1):
		return 0
	}
	m, e := split(x)
	if e&1 != 0 { // make the exponent even: m in [1, 4)
		m, e = m*2, e-1
	}
	// The classic bit-level first guess, then Newton's steps for 1/sqrt
	// as corrections, r += r·(1/2 - (m/2)·r·r): near convergence the
	// subtraction is exact, so the last step adds little rounding error.
	r := math.Float32frombits(0x5f3759df - math.Float32bits(m)>>1)
	h := float32(0.5 * m)
	for range 4 {
		t := float32(float32(h*r) * r)
		r = r + float32(r*(0.5-t))
	}
	return float32(r * Pow2(-e/2))
}
