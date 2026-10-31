package num

import (
	"math"
	"math/rand/v2"
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
