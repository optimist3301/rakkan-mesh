package gpu

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/optimist3301/rakkan-mesh/internal/num"
)

// The CPU's versions of what mathTestKernel computes.
func mathCPU(a, b float32) [8]float32 {
	abs := float32(math.Abs(float64(a)))
	root := float32(math.Sqrt(float64(abs)))
	e := num.Exp32(-a)
	sp := a
	if a <= 20 {
		sp = num.Log1p32(num.Exp32(a))
	}
	return [8]float32{a / b, root, 1 / root, num.Log32(abs), num.Log1p32(abs), 1 / (1 + e), a / (1 + e), sp}
}

// mathCases mixes ordinary values with the edges: denormals, zeros,
// infinities, quotients that overflow or underflow, exact ties.
func mathCases(r *rand.Rand, n int) (a, b []float32) {
	a, b = make([]float32, n), make([]float32, n)
	for i := range a {
		switch r.IntN(10) {
		case 0:
			a[i], b[i] = math.Float32frombits(uint32(r.Uint64())&0x007fffff), float32(r.NormFloat64()) // denormal by normal
		case 1:
			a[i], b[i] = float32(r.NormFloat64()*1e30), float32(r.NormFloat64()*1e-30) // overflow
		case 2:
			a[i], b[i] = float32(r.NormFloat64()*1e-30), float32(r.NormFloat64()*1e20) // into denormals
		case 3:
			// Small integers: exact quotients and halfway cases.
			a[i], b[i] = float32(r.IntN(1<<24)), float32(math.Ldexp(1, r.IntN(30)))
		case 4:
			vals := []float32{0, float32(math.Copysign(0, -1)), float32(math.Inf(1)), float32(math.Inf(-1)), 1, -1}
			a[i], b[i] = vals[r.IntN(len(vals))], vals[r.IntN(len(vals))]
		case 5:
			a[i], b[i] = math.Float32frombits(uint32(r.Uint64())&0x7f7fffff), math.Float32frombits(uint32(r.Uint64())&0x7f7fffff)
		default:
			a[i], b[i] = float32(r.NormFloat64()*float64(r.IntN(40))), float32(r.NormFloat64()+0.001)
		}
	}
	return a, b
}

func TestCorrectlyRoundedMath(t *testing.T) {
	r := rand.New(rand.NewPCG(13, 14))
	const n = 1 << 17
	for _, d := range devices(t) {
		for round := range 8 {
			a, b := mathCases(r, n)
			h := d.host
			out := make([]float32, 8*n)
			bt := h.Begin()
			bt.Write(0, a)
			bt.Write(n, b)
			bt.dispatch(kindMathTest, 0, groups(n, flatGroup), 1, n, 0, n, uint32(2*n))
			bt.Read(2*n, out)
			if err := bt.Submit(); err != nil {
				t.Fatal(err)
			}
			names := []string{"a/b", "sqrt", "1/sqrt", "log", "log1p", "sigmoid", "silu", "softplus"}
			for i := range n {
				want := mathCPU(a[i], b[i])
				abs := float32(math.Abs(float64(a[i])))
				for j, w := range want {
					g := out[8*i+j]
					switch {
					case j == 3 && (abs == 0 || math.IsInf(float64(abs), 0)), j == 4 && math.IsInf(float64(abs), 0):
						continue // outside the functions' domain on the GPU
					case !d.info.modes.denormPreserve && (isDenorm(a[i]) || isDenorm(b[i]) || isDenorm(w)):
						continue
					case w != w && g != g:
						continue
					}
					if math.Float32bits(g) != math.Float32bits(w) {
						t.Fatalf("%s round %d: %s(%v, %v): GPU %v (%#x), CPU %v (%#x)", d.Info().Name, round, names[j], a[i], b[i], g, math.Float32bits(g), w, math.Float32bits(w))
					}
				}
			}
		}
	}
}

func isDenorm(f float32) bool { b := math.Float32bits(f); return b>>23&0xff == 0 && b&0x7fffff != 0 }
