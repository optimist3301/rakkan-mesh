package gpu

import (
	"math/rand/v2"
	"testing"

	"github.com/optimist3301/rakkan-mesh/internal/num"
)

func BenchmarkMulVec(b *testing.B) {
	infos, err := Devices()
	if err != nil || len(infos) == 0 {
		b.Skip("no Vulkan")
	}
	d, err := Open(infos[0])
	if err != nil {
		b.Fatal(err)
	}
	defer d.Close()
	for _, shape := range [][2]int{{32, 32}, {896, 896}, {4864, 896}, {151936, 896}} {
		rows, cols := shape[0], shape[1]
		w := make([]uint16, rows*cols)
		for i := range w {
			w[i] = num.ToBF16(float32(i%13) * 0.01)
		}
		x := make([]float32, cols)
		out := make([]float32, rows)
		m, err := d.Upload(rows, cols, Format{BF16: w})
		if err != nil {
			b.Fatal(err)
		}
		b.Run(d.Info().Type+"/"+itoa(rows)+"x"+itoa(cols), func(b *testing.B) {
			for b.Loop() {
				m.MulVec(out, x)
			}
			b.SetBytes(int64(2 * rows * cols))
		})
		b.Run("cpu/"+itoa(rows)+"x"+itoa(cols), func(b *testing.B) {
			for b.Loop() {
				num.MatVecBF16(out, w, x)
			}
			b.SetBytes(int64(2 * rows * cols))
		})
		m.Free()
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for ; n > 0; n /= 10 {
		b = append([]byte{byte('0' + n%10)}, b...)
	}
	return string(b)
}

// BenchmarkMXMulVec measures DeepSeek-V4.1's products at its shapes:
// attention's wq_b, a shared expert's projection, the six routed experts'
// gate and up projections of a token.
func BenchmarkMXMulVec(b *testing.B) {
	infos, err := Devices()
	if err != nil || len(infos) == 0 {
		b.Skip("no Vulkan")
	}
	info, _ := Best(infos)
	d, err := Open(info)
	if err != nil {
		b.Fatal(err)
	}
	defer d.Close()
	r := rand.New(rand.NewPCG(1, 1))
	for _, sh := range []struct {
		name       string
		rows, cols int
		fp4        bool
	}{{"fp8-wq_b", 32768, 1280, false}, {"fp8-shared", 2304, 5120, false}, {"fp8-32768x5120", 32768, 5120, false}, {"fp4-6experts", 6 * 2 * 2304, 5120, true}} {
		x := mxInput(r, sh.cols)
		var a num.ActFP8
		num.QuantizeActFP8(x, &a)
		var f Format
		var bytes int
		if sh.fp4 {
			w := make([]byte, sh.rows*sh.cols/2)
			for i := range w {
				w[i] = byte(r.IntN(256))
			}
			s := make([]byte, sh.rows*sh.cols/32)
			randomScales(r, s, false)
			f, bytes = Format{FP4: w, E8M0: s}, len(w)
		} else {
			w, s := randomFP8(r, sh.rows, sh.cols, false)
			f, bytes = Format{FP8: w, E8M0: s}, len(w)
		}
		m, err := d.Upload(sh.rows, sh.cols, f)
		if err != nil {
			b.Fatal(err)
		}
		out := make([]float32, sh.rows)
		b.Run(d.Info().Type+"/"+sh.name, func(b *testing.B) {
			for b.Loop() {
				m.MulVec(out, x)
			}
			b.SetBytes(int64(bytes))
		})
		b.Run("cpu/"+sh.name, func(b *testing.B) {
			for b.Loop() {
				if sh.fp4 {
					num.MatVecFP4(out, f.FP4, f.E8M0, &a)
				} else {
					num.MatVecFP8(out, f.FP8, f.E8M0, &a)
				}
			}
			b.SetBytes(int64(bytes))
		})
		m.Free()
	}
}
