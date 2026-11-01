package llm

import (
	"slices"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/gpu"
	"github.com/optimist3301/rakkan-mesh/internal/num"
)

// MeasureSpeed times products with a 4096×4096 bfloat16 matrix, 32 MB
// like a layer's larger matrices, on d or, if d is nil, on the CPU, and
// returns the rate in GB of weights per second: what a forward pass costs
// this node per byte of layers, as the planner needs it. Weights in other
// types stream at about the same rate per byte.
func MeasureSpeed(d *gpu.Device) float64 {
	const rows, cols = 4096, 4096
	w := make([]uint16, rows*cols)
	seed := uint32(1)
	for i := range w {
		seed = seed*1664525 + 1013904223
		w[i] = num.ToBF16(float32(int32(seed>>8)-1<<23) / (1 << 30))
	}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(i%17) / 17
	}
	out := make([]float32, rows)
	run := func() error { num.MatVecBF16(out, w, x); return nil }
	if d != nil {
		if m, err := d.Upload(rows, cols, gpu.Format{BF16: w}); err == nil {
			defer m.Free()
			run = func() error { return m.MulVec(out, x) }
		}
	}
	run() // warm up
	var times []time.Duration
	for range 7 {
		start := time.Now()
		if run() != nil {
			return 0
		}
		times = append(times, time.Since(start))
	}
	slices.Sort(times)
	return float64(2*rows*cols) / times[len(times)/2].Seconds() / 1e9
}
