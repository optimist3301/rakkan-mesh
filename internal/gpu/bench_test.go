package gpu

import (
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
