package gpu

import (
	"math"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/num"
)

// devices opens every Vulkan device, software ones included, or skips.
func devices(t *testing.T) []*Device {
	t.Helper()
	infos, err := Devices()
	if err != nil || len(infos) == 0 {
		t.Skipf("no Vulkan devices: %v", err)
	}
	var out []*Device
	for _, info := range infos {
		d, err := Open(info)
		if err != nil {
			t.Fatalf("%s: %v", info.Name, err)
		}
		t.Cleanup(d.Close)
		t.Logf("%s: %s, Vulkan %s, %d MiB, exact %v, modes %+v, max binding %d MiB", info.Name, info.Type, info.API,
			info.Memory>>20, info.Exact, info.modes, info.maxRange>>20)
		out = append(out, d)
	}
	return out
}

// awkward fills v with normal values and the cases that tell exact
// arithmetic apart: denormals, signed zeros, values that cancel.
func awkward(r *rand.Rand, v []float32, scale float64) {
	for i := range v {
		switch r.IntN(40) {
		case 0:
			v[i] = math.Float32frombits(uint32(r.IntN(1 << 23))) // denormal
		case 1:
			v[i] = float32(math.Copysign(0, -1))
		case 2:
			v[i] = 1e-30
		default:
			v[i] = float32(r.NormFloat64() * scale)
		}
	}
}

func sameBits(t *testing.T, what string, got, want []float32) {
	t.Helper()
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s: row %d: GPU %v (%#x), CPU %v (%#x)", what, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

func TestKernelsMatchCPUBitForBit(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, d := range devices(t) {
		name := d.Info().Name
		for _, shape := range [][2]int{{7, 32}, {300, 896}, {64, 4864}, {1000, 96}} {
			rows, cols := shape[0], shape[1]
			w := make([]float32, rows*cols)
			awkward(r, w, 0.05)
			x := make([]float32, cols)
			awkward(r, x, 1)
			got, want := make([]float32, rows), make([]float32, rows)

			m, err := d.Upload(rows, cols, Format{F32: w})
			if err != nil {
				t.Fatal(err)
			}
			if err := m.MulVec(got, x); err != nil {
				t.Fatal(err)
			}
			num.MatVec32(want, w, x)
			sameBits(t, name+" f32", got, want)
			m.Free()

			w16 := make([]uint16, len(w))
			for i, v := range w {
				w16[i] = num.ToBF16(v)
			}
			m, _ = d.Upload(rows, cols, Format{BF16: w16})
			m.MulVec(got, x)
			num.MatVecBF16(want, w16, x)
			sameBits(t, name+" bf16", got, want)
			m.Free()

			q8, s8 := num.QuantizeQ8(w)
			m, _ = d.Upload(rows, cols, Format{Q8: q8, Scales: s8})
			m.MulVec(got, x)
			num.MatVecQ8(want, q8, s8, x)
			sameBits(t, name+" q8_0", got, want)
			m.Free()

			q4, s4 := num.QuantizeQ4(w)
			m, _ = d.Upload(rows, cols, Format{Q4: q4, Scales: s4})
			m.MulVec(got, x)
			num.MatVecQ4(want, q4, s4, x)
			sameBits(t, name+" q4_0", got, want)
			m.Free()
		}
		if d.Used() != 0 {
			t.Fatalf("%s: %d bytes still held", name, d.Used())
		}
	}
}

// Matrices past one binding's limit, or past 65535 workgroups in one
// dimension, are split and still exact.
func TestLargeMatrices(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	for _, d := range devices(t) {
		for _, shape := range [][2]int{{300000, 32}, {40000, 1024}} {
			rows, cols := shape[0], shape[1]
			w := make([]float32, rows*cols)
			for i := range w {
				w[i] = float32(r.NormFloat64())
			}
			x := make([]float32, cols)
			awkward(r, x, 1)
			m, err := d.Upload(rows, cols, Format{F32: w})
			if err != nil {
				t.Fatal(err)
			}
			got, want := make([]float32, rows), make([]float32, rows)
			start := time.Now()
			if err := m.MulVec(got, x); err != nil {
				t.Fatal(err)
			}
			el := time.Since(start)
			num.MatVec32(want, w, x)
			sameBits(t, d.Info().Name, got, want)
			t.Logf("%s: %d×%d in %d parts, %s", d.Info().Name, rows, cols, len(m.parts), el)
			m.Free()
		}
	}
}
