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
