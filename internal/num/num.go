// Package num is floating-point arithmetic that gives bit-identical results
// on every architecture, for computations whose results are compared across
// nodes (replicated tasks, pipeline stages of a model).
//
// Two things break that in ordinary Go code. The specification lets the
// compiler fuse x*y+z into one fused multiply-add, rounding once instead of
// twice, even across statements; arm64 compilers do. An explicit conversion
// such as float32(x*y) rounds and forbids the fusion, so every product here
// is converted before it is added. And the math package's transcendental
// functions are assembly on some architectures and, elsewhere, compiled
// code subject to fusion; the ones needed are implemented here instead.
// math.Sqrt is exact (correctly rounded) everywhere and is used directly.
package num

import (
	"math"
	"runtime"
	"sync"
)

// CanonicalNaN32 is the quiet NaN that WebAssembly code is canonicalized to.
var CanonicalNaN32 = math.Float32frombits(0x7fc00000)

// Canonical replaces any NaN with CanonicalNaN32: NaN payloads produced by
// arithmetic differ between processors.
func Canonical(v float32) float32 {
	if v != v {
		return CanonicalNaN32
	}
	return v
}

// Dot returns sum(a[i]*b[i]) summed in order, each product rounded.
func Dot(a, b []float32) float32 {
	var acc float32
	b = b[:len(a)]
	for i, x := range a {
		acc += float32(x * b[i])
	}
	return acc
}

// parallelRows is the work, in multiply-adds, above which MatVec splits
// rows among goroutines.
const parallelRows = 1 << 16

// MatVec sets out[i] = Dot(w[i*n:(i+1)*n], x) for each row i of w, which
// has len(out) rows of n = len(x) columns. Large products are split by rows
// across goroutines; each row is still computed by one goroutine in order,
// so the result does not depend on the split.
func MatVec(out, w, x []float32) {
	n, rows := len(x), len(out)
	workers := min(runtime.GOMAXPROCS(0), rows*n/parallelRows)
	if workers <= 1 {
		for i := range out {
			out[i] = Dot(w[i*n:(i+1)*n], x)
		}
		return
	}
	var wg sync.WaitGroup
	per := (rows + workers - 1) / workers
	for lo := 0; lo < rows; lo += per {
		hi := min(lo+per, rows)
		wg.Go(func() {
			for i := lo; i < hi; i++ {
				out[i] = Dot(w[i*n:(i+1)*n], x)
			}
		})
	}
	wg.Wait()
}

// Exp32 is e^x in float32: range reduction to x = n*ln2 + r with
// |r| <= ln2/2, a degree-6 polynomial for e^r, and exact scaling by 2^n.
// Relative error is within a few ulp.
func Exp32(x float32) float32 {
	switch {
	case x != x:
		return CanonicalNaN32
	case x > 88.72:
		return float32(math.Inf(1))
	case x < -103.97:
		return 0
	}
	const (
		ln2Hi = 0.693145751953125     // ln2 to 16 bits, so n*ln2Hi is exact
		ln2Lo = 1.428606765330187e-06 // ln2 - ln2Hi
	)
	n := float32(math.RoundToEven(float64(float32(x * 1.4426950408889634))))
	r := x - float32(n*ln2Hi)
	r = r - float32(n*ln2Lo)
	p := float32(1.0 / 720)
	for _, c := range []float32{1.0 / 120, 1.0 / 24, 1.0 / 6, 0.5, 1, 1} {
		p = float32(p*r) + c
	}
	return float32(math.Ldexp(float64(p), int(n)))
}

// Exp is e^x in float64, accurate to about 1 ulp, by the same method with a
// degree-13 polynomial.
func Exp(x float64) float64 {
	switch {
	case x != x:
		return x
	case x > 709.78:
		return math.Inf(1)
	case x < -745.13:
		return 0
	}
	const (
		ln2Hi = 6.93147180369123816490e-01
		ln2Lo = 1.90821492927058770002e-10
	)
	n := math.RoundToEven(float64(x * 1.4426950408889634))
	r := x - float64(n*ln2Hi)
	r = r - float64(n*ln2Lo)
	// Taylor series to r^13/13!: |r| <= 0.347, error below 2^-60.
	p := 1.0 / 6227020800
	for k := 12; k >= 0; k-- {
		p = float64(p*r) + 1/factorial(k)
	}
	return math.Ldexp(p, int(n))
}

func factorial(k int) float64 {
	f := 1.0
	for i := 2; i <= k; i++ {
		f *= float64(i)
	}
	return f
}

// Sincos returns sin(x) and cos(x) for |x| < 2^20, accurate to about 1 ulp:
// Cody-Waite reduction by pi/2 with a three-part constant, then the Cephes
// polynomials (the same as Go's math package) with every product rounded.
func Sincos(x float64) (sin, cos float64) {
	// pi/2 in three parts (fdlibm's pio2_1, pio2_2, pio2_2t); the first two
	// have 33 significant bits, so k times either is exact for |k| < 2^20.
	const (
		pio2a = 1.57079632673412561417e+00
		pio2b = 6.07710050630396597660e-11
		pio2c = 2.02226624879595063154e-21
	)
	k := math.RoundToEven(float64(x * (2 / math.Pi)))
	r := x - float64(k*pio2a)
	r = r - float64(k*pio2b)
	r = r - float64(k*pio2c)
	s, c := sinPoly(r), cosPoly(r)
	switch int64(k) & 3 {
	case 0:
		return s, c
	case 1:
		return c, -s
	case 2:
		return -s, -c
	default:
		return -c, s
	}
}

var (
	sinCoef = [...]float64{
		1.58962301576546568060e-10, -2.50507477628578072866e-8, 2.75573136213857245213e-6,
		-1.98412698295895385996e-4, 8.33333333332211858878e-3, -1.66666666666666307295e-1,
	}
	cosCoef = [...]float64{
		-1.13585365213876817300e-11, 2.08757008419747316778e-9, -2.75573141792967388112e-7,
		2.48015872888517045348e-5, -1.38888888888730564116e-3, 4.16666666666665929218e-2,
	}
)

// sinPoly and cosPoly are valid for |r| <= pi/4.
func sinPoly(r float64) float64 {
	z := float64(r * r)
	p := sinCoef[0]
	for _, c := range sinCoef[1:] {
		p = float64(p*z) + c
	}
	return r + float64(float64(r*z)*p)
}

func cosPoly(r float64) float64 {
	z := float64(r * r)
	p := cosCoef[0]
	for _, c := range cosCoef[1:] {
		p = float64(p*z) + c
	}
	return 1 - float64(0.5*z) + float64(float64(z*z)*p)
}
