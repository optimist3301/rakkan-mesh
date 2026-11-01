package num

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

func ulps(a, b float64) float64 {
	if a == b {
		return 0
	}
	return math.Abs(a-b) / math.Abs(math.Nextafter(b, math.Inf(1))-b)
}

func TestSincos(t *testing.T) {
	worst := 0.0
	for x := -300.0; x < 300; x += 0.000731 {
		s, c := Sincos(x)
		ws, wc := math.Sincos(x)
		// Absolute error near zeros, relative elsewhere.
		if math.Abs(ws) > 1e-3 {
			worst = max(worst, ulps(s, ws))
		}
		if math.Abs(wc) > 1e-3 {
			worst = max(worst, ulps(c, wc))
		}
		if math.Abs(s-ws) > 1e-15 && math.Abs(ws) <= 1e-3 || math.Abs(c-wc) > 1e-15 && math.Abs(wc) <= 1e-3 {
			t.Fatalf("x=%v: sincos %v %v, math %v %v", x, s, c, ws, wc)
		}
	}
	if worst > 4 {
		t.Fatalf("worst error %.1f ulp", worst)
	}
}

func TestExp(t *testing.T) {
	worst := 0.0
	for x := -700.0; x < 700; x += 0.0173 {
		worst = max(worst, ulps(Exp(x), math.Exp(x)))
	}
	if worst > 4 {
		t.Fatalf("Exp worst error %.1f ulp", worst)
	}
	worst = 0
	for x := float32(-100); x < 88; x += 0.0137 {
		got, want := float64(Exp32(x)), math.Exp(float64(x))
		if want > 1e-37 {
			worst = max(worst, math.Abs(got-want)/want)
		}
	}
	if worst > 4e-7 {
		t.Fatalf("Exp32 worst relative error %.3g", worst)
	}
}

func TestMatVecSplitDoesNotChangeBits(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 1))
	rows, n := 4000, 300 // large enough to split
	w, x := make([]float32, rows*n), make([]float32, n)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	par, seq := make([]float32, rows), make([]float32, rows)
	MatVec(par, w, x)
	for i := range seq {
		seq[i] = Dot(w[i*n:(i+1)*n], x)
	}
	for i := range par {
		if math.Float32bits(par[i]) != math.Float32bits(seq[i]) {
			t.Fatalf("row %d differs", i)
		}
	}
}

func TestDot8AndBF16(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, n := range []int{0, 1, 7, 8, 9, 63, 896, 4865} {
		w16 := make([]uint16, n)
		w := make([]float32, n)
		x := make([]float32, n)
		var ref float64
		for i := range w {
			w16[i] = ToBF16(float32(r.NormFloat64()))
			w[i] = BF16(w16[i])
			x[i] = float32(r.NormFloat64())
			ref += float64(w[i]) * float64(x[i])
		}
		a, b := Dot32(w, x), DotBF16(w16, x)
		if math.Float32bits(a) != math.Float32bits(b) {
			t.Fatalf("n=%d: bf16 %v, f32 %v", n, b, a)
		}
		if d := math.Abs(float64(a) - ref); d > 1e-4*math.Sqrt(float64(n))+1e-6 {
			t.Fatalf("n=%d: %v, exact %v", n, a, ref)
		}
	}
	if ToBF16(1.0) != 0x3f80 || BF16(0x3f80) != 1 || ToBF16(float32(1+1.0/256)) != 0x3f80 || ToBF16(float32(1+3.0/256)) != 0x3f82 {
		t.Fatal("bf16 rounding")
	}
	// The vector code and the portable code agree bit for bit.
	for _, n := range []int{32, 64, 96, 4096} {
		a, b16, x := make([]float32, n), make([]uint16, n), make([]float32, n)
		for i := range a {
			a[i], x[i] = float32(r.NormFloat64()), float32(r.NormFloat64()*1e3)
			b16[i] = ToBF16(float32(r.NormFloat64()))
		}
		var v1, g1, v2, g2 [32]float32
		dot32(&a[0], &x[0], n, &v1)
		dot32Generic(a, x, &g1)
		dotBF16(&b16[0], &x[0], n, &v2)
		dotBF16Generic(b16, x, &g2)
		if v1 != g1 || v2 != g2 {
			t.Fatalf("n=%d: vector and portable sums differ", n)
		}
	}
	t.Logf("AVX2: %v", useAVX2)
	// Large products split across goroutines give the same bits.
	rowsN, cols := 3000, 900
	w16 := make([]uint16, rowsN*cols)
	for i := range w16 {
		w16[i] = ToBF16(float32(r.NormFloat64()))
	}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	par := make([]float32, rowsN)
	MatVecBF16(par, w16, x)
	for i := range par {
		if s := DotBF16(w16[i*cols:(i+1)*cols], x); math.Float32bits(s) != math.Float32bits(par[i]) {
			t.Fatalf("row %d differs when split", i)
		}
	}
}

func BenchmarkDot(b *testing.B) {
	x := make([]float32, 4096)
	w := make([]float32, 4096)
	w16 := make([]uint16, 4096)
	for i := range x {
		x[i], w[i] = float32(i%7), float32(i%5)
	}
	b.Run("Dot", func(b *testing.B) {
		for b.Loop() {
			Dot(w, x)
		}
	})
	b.Run("Dot32", func(b *testing.B) {
		for b.Loop() {
			Dot32(w, x)
		}
	})
	b.Run("DotBF16", func(b *testing.B) {
		for b.Loop() {
			DotBF16(w16, x)
		}
	})
}

func TestQuantized(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	rowsN, cols := 300, 896
	w := make([]float32, rowsN*cols)
	for i := range w {
		w[i] = float32(r.NormFloat64() * 0.05)
	}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	exact := make([]float32, rowsN)
	MatVec32(exact, w, x)
	q8, s8 := QuantizeQ8(w)
	q4, s4 := QuantizeQ4(w)
	got8, got4 := make([]float32, rowsN), make([]float32, rowsN)
	MatVecQ8(got8, q8, s8, x)
	MatVecQ4(got4, q4, s4, x)
	var e8, e4, norm float64
	for i := range exact {
		e8 += math.Pow(float64(got8[i]-exact[i]), 2)
		e4 += math.Pow(float64(got4[i]-exact[i]), 2)
		norm += math.Pow(float64(exact[i]), 2)
	}
	e8, e4 = math.Sqrt(e8/norm), math.Sqrt(e4/norm)
	t.Logf("relative error: Q8_0 %.4f, Q4_0 %.4f", e8, e4)
	if e8 > 0.01 || e4 > 0.15 {
		t.Fatal("quantization error too large")
	}
	// Dequantized Q8_0 weights are within half a step; Q4_0 weights,
	// whose scale minimizes the block's error rather than any one weight's,
	// within two.
	d8, d4 := make([]float32, len(w)), make([]float32, len(w))
	DequantQ8(d8, q8, s8)
	DequantQ4(d4, q4, s4)
	for i := range w {
		if math.Abs(float64(d8[i]-w[i])) > float64(s8[i/QBlock])*0.5001 || math.Abs(float64(d4[i]-w[i])) > math.Abs(float64(s4[i/QBlock]))*2.0001+1e-9 {
			t.Fatalf("weight %d: %v, Q8 %v, Q4 %v", i, w[i], d8[i], d4[i])
		}
	}
	// Vector and portable integer sums agree, including extreme values.
	qx := make([]int8, cols)
	for i := range qx {
		qx[i] = int8(r.IntN(255) - 127)
	}
	for i := range q8[:cols] {
		if i%7 == 0 {
			q8[i] = 127
		} else if i%11 == 0 {
			q8[i] = -127
		}
	}
	for _, n := range []int{cols, cols - 32, 96, 32} { // 28, 27, 3 and 1 blocks
		nb := n / QBlock
		a, b := make([]int32, nb), make([]int32, nb)
		blockSumsQ8(q8[:n], qx[:n], a)
		blockSumsQ8Generic(q8[:n], qx[:n], b)
		if !slices.Equal(a, b) {
			t.Fatalf("%d blocks: Q8 sums differ between vector and portable code", nb)
		}
		blockSumsQ4(q4[:n/2], qx[:n], a)
		blockSumsQ4Generic(q4[:n/2], qx[:n], b)
		if !slices.Equal(a, b) {
			t.Fatalf("%d blocks: Q4 sums differ between vector and portable code", nb)
		}
	}
}

func BenchmarkMatVec(b *testing.B) {
	rowsN, cols := 4864, 896
	w := make([]float32, rowsN*cols)
	for i := range w {
		w[i] = float32(i%13) * 0.01
	}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(i % 7)
	}
	out := make([]float32, rowsN)
	w16 := make([]uint16, len(w))
	for i, v := range w {
		w16[i] = ToBF16(v)
	}
	q8, s8 := QuantizeQ8(w)
	q4, s4 := QuantizeQ4(w)
	b.Run("f32", func(b *testing.B) {
		for b.Loop() {
			MatVec32(out, w, x)
		}
	})
	b.Run("bf16", func(b *testing.B) {
		for b.Loop() {
			MatVecBF16(out, w16, x)
		}
	})
	b.Run("q8_0", func(b *testing.B) {
		for b.Loop() {
			MatVecQ8(out, q8, s8, x)
		}
	})
	b.Run("q4_0", func(b *testing.B) {
		for b.Loop() {
			MatVecQ4(out, q4, s4, x)
		}
	})
}

func TestRecipRsqrt(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	ulpsOf := func(got float32, exact float64) float64 {
		g := float64(got)
		if g == exact {
			return 0
		}
		step := math.Abs(float64(math.Nextafter32(float32(exact), float32(math.Inf(1)))) - float64(float32(exact)))
		if step == 0 {
			step = math.SmallestNonzeroFloat32
		}
		return math.Abs(g-exact) / step
	}
	var worstR, worstS float64
	for i := range 2_000_000 {
		var x float32
		switch i % 4 {
		case 0:
			x = float32(math.Exp(r.Float64()*170 - 85))
		case 1:
			x = math.Float32frombits(uint32(r.Uint32() & 0x7fffffff)) // any positive bits
		case 2:
			x = float32(r.Float64()*4 + 0.25)
		default:
			x = math.Float32frombits(uint32(r.IntN(1 << 23))) // subnormal
		}
		if x == 0 || math.IsInf(float64(x), 0) || x != x {
			continue
		}
		if 1/float64(x) > math.MaxFloat32 {
			if !math.IsInf(float64(Recip32(x)), 1) {
				t.Fatalf("Recip32(%v) = %v, want +Inf", x, Recip32(x))
			}
		} else {
			worstR = max(worstR, ulpsOf(Recip32(x), 1/float64(x)))
		}
		worstS = max(worstS, ulpsOf(Rsqrt32(x), 1/math.Sqrt(float64(x))))
		if y := -x; Recip32(y) != -Recip32(x) {
			t.Fatalf("Recip32(-x) != -Recip32(x) at %v", x)
		}
	}
	t.Logf("worst error: Recip32 %.2f ulp, Rsqrt32 %.2f ulp", worstR, worstS)
	if worstR > 2 || worstS > 2 {
		t.Fatal("too inaccurate")
	}
	inf := float32(math.Inf(1))
	if Recip32(0) != inf || Recip32(inf) != 0 || Rsqrt32(0) != inf || Rsqrt32(inf) != 0 || Recip32(float32(math.NaN())) == Recip32(float32(math.NaN())) {
		t.Fatal("special values")
	}
	if Recip32(4) != 0.25 || Rsqrt32(4) != 0.5 || Rsqrt32(1) != 1 {
		t.Fatalf("exact cases: %v %v %v", Recip32(4), Rsqrt32(4), Rsqrt32(1))
	}
}
