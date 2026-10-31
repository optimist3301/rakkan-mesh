package num

import (
	"math"
	"runtime"
	"sync"
)

// Dot sums in one chain, so each addition waits for the previous one.
// Dot32 keeps 32 independent sums instead: element i is added, in order, to
// sum i mod 32, each product rounded before it is added (no fused
// multiply-add). The sums then combine in a fixed tree: sum j with sum
// j+16, then j with j+8 (giving eight), then ((0+1)+(2+3))+((4+5)+(6+7)).
// That is exactly four 8-lane vector accumulators with separate multiply
// and add instructions, so AVX2 computes it directly (dot_amd64.s), and the
// portable code below gives the same bits everywhere else. Model weights
// use it; task kernels keep Dot, whose order their semantics specify.

// BF16 converts a bfloat16, the upper half of a float32, exactly.
func BF16(v uint16) float32 { return math.Float32frombits(uint32(v) << 16) }

// ToBF16 rounds a float32 to the nearest bfloat16, ties to even.
func ToBF16(f float32) uint16 {
	b := math.Float32bits(f)
	if f != f {
		return uint16(b>>16) | 0x40 // stays NaN
	}
	return uint16((b + 0x7fff + (b>>16)&1) >> 16)
}

// Dot32 returns the dot product of a and b[:len(a)] in the 32-sum order.
func Dot32(a, b []float32) float32 {
	b = b[:len(a)]
	var s [32]float32
	full := len(a) &^ 31
	if full > 0 {
		dot32(&a[0], &b[0], full, &s)
	}
	for i := full; i < len(a); i++ {
		s[i&31] += float32(a[i] * b[i])
	}
	return combine(&s)
}

// DotBF16 is Dot32 with the first operand in bfloat16: the same result as
// Dot32 on the converted values, since conversion is exact.
func DotBF16(w []uint16, x []float32) float32 {
	x = x[:len(w)]
	var s [32]float32
	full := len(w) &^ 31
	if full > 0 {
		dotBF16(&w[0], &x[0], full, &s)
	}
	for i := full; i < len(w); i++ {
		s[i&31] += float32(BF16(w[i]) * x[i])
	}
	return combine(&s)
}

func combine(s *[32]float32) float32 {
	var e [8]float32
	for j := range e {
		e[j] = (s[j] + s[j+8]) + (s[j+16] + s[j+24])
	}
	return ((e[0] + e[1]) + (e[2] + e[3])) + ((e[4] + e[5]) + (e[6] + e[7]))
}

// dot32Generic accumulates n elements (a multiple of 32) into s.
func dot32Generic(a, b []float32, s *[32]float32) {
	for i := 0; i < len(a); i += 32 {
		a, b := a[i:i+32:i+32], b[i:i+32:i+32]
		for j := range 32 {
			s[j] += float32(a[j] * b[j])
		}
	}
}

func dotBF16Generic(w []uint16, x []float32, s *[32]float32) {
	for i := 0; i < len(w); i += 32 {
		w, x := w[i:i+32:i+32], x[i:i+32:i+32]
		for j := range 32 {
			s[j] += float32(BF16(w[j]) * x[j])
		}
	}
}

// rows runs f over [0, rows) split among goroutines when the work, in
// multiply-adds, is large; each row is computed by one goroutine, so the
// split does not affect results.
func rows(n, cols int, f func(lo, hi int)) {
	workers := min(runtime.GOMAXPROCS(0), n*cols/parallelRows)
	if workers <= 1 {
		f(0, n)
		return
	}
	var wg sync.WaitGroup
	per := (n + workers - 1) / workers
	for lo := 0; lo < n; lo += per {
		hi := min(lo+per, n)
		wg.Go(func() { f(lo, hi) })
	}
	wg.Wait()
}

// MatVec32 sets out[i] = Dot32(row i of w, x); w has len(out) rows of len(x).
func MatVec32(out, w, x []float32) {
	n := len(x)
	rows(len(out), n, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			out[i] = Dot32(w[i*n:(i+1)*n], x)
		}
	})
}

// MatVecBF16 is MatVec32 with w in bfloat16.
func MatVecBF16(out []float32, w []uint16, x []float32) {
	n := len(x)
	rows(len(out), n, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			out[i] = DotBF16(w[i*n:(i+1)*n], x)
		}
	})
}
