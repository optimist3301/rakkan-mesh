package num

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestErfMatchesMath(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	xs := []float64{0, -0, 1e-310, 1e-20, 0.5, 0.84375, 1, 1.25, 2, 1 / 0.35, 3, 5.99, 6, 30, math.Inf(1), math.Inf(-1)}
	for range 100000 {
		xs = append(xs, (r.Float64()*2-1)*7)
	}
	for _, x := range xs {
		got, want := Erf(x), math.Erf(x)
		if d := math.Abs(got - want); d > 4e-16*math.Max(math.Abs(want), 1e-300) && got != want {
			t.Fatalf("Erf(%v) = %v, math.Erf %v", x, got, want)
		}
	}
	if !math.IsNaN(Erf(math.NaN())) {
		t.Error("Erf(NaN)")
	}
}

func TestMatMulBF16MatchesMatVec(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	for _, s := range [][3]int{{1, 1, 1}, {3, 70, 33}, {130, 65, 96}, {200, 129, 1030}} {
		n, m, k := s[0], s[1], s[2]
		w := make([]uint16, m*k)
		for i := range w {
			w[i] = ToBF16(float32(r.NormFloat64()))
		}
		x := make([]float32, n*k)
		for i := range x {
			x[i] = float32(r.NormFloat64())
		}
		out := make([]float32, n*m)
		MatMulBF16(out, w, x, n, m)
		row := make([]float32, m)
		for i := range n {
			MatVecBF16(row, w, x[i*k:(i+1)*k])
			for j := range m {
				if math.Float32bits(row[j]) != math.Float32bits(out[i*m+j]) {
					t.Fatalf("%v: out[%d][%d] = %v, MatVecBF16 %v", s, i, j, out[i*m+j], row[j])
				}
			}
		}
	}
}
