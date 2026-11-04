package gpu

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/num"
)

// randomFP8 fills a rows×cols E4M3 matrix and its block scales: every
// code but NaN, scales around 1 and, if denormals, the extremes (2^-127,
// the subnormal, and others making the terms subnormal).
func randomFP8(r *rand.Rand, rows, cols int, denormals bool) (w, s []byte) {
	w = make([]byte, rows*cols)
	for i := range w {
		c := byte(r.IntN(256))
		for c&0x7f == 0x7f {
			c = byte(r.IntN(256))
		}
		w[i] = c
	}
	s = make([]byte, (rows+31)/32*(cols/32))
	randomScales(r, s, denormals)
	return w, s
}

func randomScales(r *rand.Rand, s []byte, denormals bool) {
	for i := range s {
		switch r.IntN(20) {
		case 0, 1:
			if !denormals {
				s[i] = byte(60 + r.IntN(10))
			} else if r.IntN(2) == 0 {
				s[i] = 0
			} else {
				s[i] = byte(1 + r.IntN(10)) // results reaching denormals
			}
		case 2:
			s[i] = byte(200 + r.IntN(30))
		default:
			s[i] = byte(110 + r.IntN(30))
		}
	}
}

// mxInput mixes blocks of every kind the quantization treats apart:
// ordinary, all zero (the 1e-4 floor), huge, tiny, signed zeros and
// denormals, values at E4M3 rounding ties.
func mxInput(r *rand.Rand, n int) []float32 {
	x := make([]float32, n)
	for b := 0; b < n; b += 32 {
		blk := x[b : b+32]
		switch r.IntN(6) {
		case 0:
		case 1:
			for i := range blk {
				blk[i] = float32(r.NormFloat64() * 1e30)
			}
		case 2:
			for i := range blk {
				blk[i] = float32(r.NormFloat64() * 1e-6)
			}
		case 3:
			// Ties: with 448 present the scale is 1, and k+0.5 steps of the
			// E4M3 grid sit halfway.
			blk[0] = 448
			for i := 1; i < 32; i++ {
				e := r.IntN(12) - 3
				blk[i] = float32(math.Ldexp(8+float64(r.IntN(8))+0.5, e-3))
				if r.IntN(2) == 0 {
					blk[i] = -blk[i]
				}
			}
		default:
			awkward(r, blk, 3)
		}
	}
	return x
}

// checkUnrounded compares the float32 sums, before bfloat16 hides their
// last bits: an inexact partial sum shows only there.
func checkUnrounded(t *testing.T, what string, m *Matrix, x []float32, cpu func([]float32)) {
	t.Helper()
	got, want := make([]float32, m.rows), make([]float32, m.rows)
	m.unrounded = true
	defer func() { m.unrounded = false }()
	if err := m.MulVec(got, x); err != nil {
		t.Fatal(err)
	}
	cpu(want)
	sameBits(t, what+" unrounded", got, want)
}

func TestQuantizeFP8MatchesCPU(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	for _, d := range devices(t) {
		for _, n := range []int{32, 96, 4096} {
			x := mxInput(r, n)
			var a num.ActFP8
			num.QuantizeActFP8(x, &a)
			h := d.host
			vals, scales := make([]float32, n), make([]float32, n/32)
			b := h.Begin()
			b.Write(0, x)
			b.QuantizeFP8(0, n)
			b.Read(h.qxOff, vals)
			b.Read(h.xsOff, scales)
			if err := b.Submit(); err != nil {
				t.Fatal(err)
			}
			for i := range n {
				got := float64(vals[i])
				if math.Float64bits(got) != math.Float64bits(a.Vals[i]) {
					t.Fatalf("%s: element %d (%v, scale %v): GPU %v, CPU %v", d.Info().Name, i, x[i], a.Scales[i/32], got, a.Vals[i])
				}
			}
			sameBits(t, d.Info().Name+" scales", scales, a.Scales)
		}
	}
}

// Terms become denormal only with weight scales below about 2^-100, which
// models do not have; devices flushing denormals (llvmpipe says it does)
// differ from the CPU there, and are spared those cases.
func TestMXMatVecsMatchCPU(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	for _, d := range devices(t) {
		name := d.Info().Name
		denormals := d.info.modes.denormPreserve
		for _, shape := range [][2]int{{7, 32}, {33, 64}, {300, 896}, {64, 5120}, {1000, 96}, {40, 2304}} {
			rows, cols := shape[0], shape[1]
			x := mxInput(r, cols)
			var a num.ActFP8
			num.QuantizeActFP8(x, &a)
			got, want := make([]float32, rows), make([]float32, rows)

			w, s := randomFP8(r, rows, cols, denormals)
			m, err := d.Upload(rows, cols, Format{FP8: w, E8M0: s})
			if err != nil {
				t.Fatal(err)
			}
			if err := m.MulVec(got, x); err != nil {
				t.Fatal(err)
			}
			num.MatVecFP8(want, w, s, &a)
			sameBits(t, name+" fp8", got, want)
			checkUnrounded(t, name+" fp8", m, x, func(out []float32) { num.MatVecFP8Unrounded(out, w, s, &a) })
			m.Free()

			w4 := make([]byte, rows*cols/2)
			for i := range w4 {
				w4[i] = byte(r.IntN(256))
			}
			s4 := make([]byte, rows*cols/32)
			randomScales(r, s4, denormals)
			m, err = d.Upload(rows, cols, Format{FP4: w4, E8M0: s4})
			if err != nil {
				t.Fatal(err)
			}
			if err := m.MulVec(got, x); err != nil {
				t.Fatal(err)
			}
			num.MatVecFP4(want, w4, s4, &a)
			sameBits(t, name+" fp4", got, want)
			checkUnrounded(t, name+" fp4", m, x, func(out []float32) { num.MatVecFP4Unrounded(out, w4, s4, &a) })
			m.Free()
		}
	}
}

// A block's exact sum is rounded to float32 to even, before the result is
// rounded to bfloat16: rows built so that the two roundings in a row
// differ from one. In units of 2^-18 the sums are 2^25 + 2^18 + 2^17 - 1
// and - 2: float32 rounds both up, to a bfloat16 tie, which goes up to
// 2^25 + 2^19; truncating, or rounding once, gives 2^25 + 2^18.
func TestFP8SumRoundsToEven(t *testing.T) {
	const cols = 32
	x := make([]float32, cols)
	x[0], x[1], x[2] = 256, 256, 256  // ·512 = 2^17, and the scale is 1
	x[3] = float32(math.Ldexp(1, -9)) // ·512 = 1
	var rowsW [][4]byte
	for _, low := range []byte{0x81, 0x82} { // -1, -2 units
		row := [4]byte{num.EncodeE4M3(0.5), 2, 1, low} // 2^25, 2^18, 2^17 with 2^17
		rowsW = append(rowsW, row)
		for i := range row {
			row[i] ^= 0x80 // the negated sums
		}
		rowsW = append(rowsW, row)
	}
	// And the largest sums: 32 products of 448 by 448.
	big := make([]byte, cols)
	for i := range big {
		big[i] = num.EncodeE4M3(448)
	}
	rows := len(rowsW) + 1
	w := make([]byte, rows*cols)
	for i, row := range rowsW {
		copy(w[i*cols:], row[:])
	}
	copy(w[len(rowsW)*cols:], big)
	s := []byte{127} // 1
	xb := slices32(x)
	for i := range xb {
		if i >= 4 {
			xb[i] = 448
		}
	}
	for _, in := range [][]float32{x, xb} {
		var a num.ActFP8
		num.QuantizeActFP8(in, &a)
		want := make([]float32, rows)
		num.MatVecFP8(want, w, s, &a)
		if in[4] == 0 && (want[0] != 130 || want[1] != -130 || want[2] != 130 || want[3] != -130) {
			t.Fatalf("CPU rounding not as planned: %v", want)
		}
		for _, d := range devices(t) {
			m, err := d.Upload(rows, cols, Format{FP8: w, E8M0: s})
			if err != nil {
				t.Fatal(err)
			}
			got := make([]float32, rows)
			if err := m.MulVec(got, in); err != nil {
				t.Fatal(err)
			}
			sameBits(t, d.Info().Name, got, want)
			m.Free()
		}
	}
}

func slices32(x []float32) []float32 { return append([]float32(nil), x...) }

// A tie with an even mantissa stays down: block 0 sums 2^25 + 2 units,
// which is 128 rounded to even (128 + 2^-16 away from zero); block 1 adds
// 0.5, and bfloat16 rounds 128.5 to 128 but 128.5 + 2^-16 to 129.
func TestFP8TieStaysEven(t *testing.T) {
	const cols = 64
	x := make([]float32, cols)
	x[0], x[1], x[32] = 256, float32(math.Ldexp(1, -9)), 256
	w := make([]byte, cols)
	w[0], w[1], w[32] = num.EncodeE4M3(0.5), 2, 1
	s := []byte{127, 127}
	var a num.ActFP8
	num.QuantizeActFP8(x, &a)
	want := make([]float32, 1)
	num.MatVecFP8(want, w, s, &a)
	if want[0] != 128 {
		t.Fatalf("CPU: %v", want[0])
	}
	for _, d := range devices(t) {
		m, err := d.Upload(1, cols, Format{FP8: w, E8M0: s})
		if err != nil {
			t.Fatal(err)
		}
		got := make([]float32, 1)
		if err := m.MulVec(got, x); err != nil {
			t.Fatal(err)
		}
		sameBits(t, d.Info().Name, got, want)
		m.Free()
	}
}

// Matrices larger than a binding are split on whole blocks of 32 rows.
func TestLargeMXMatrices(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 10))
	for _, d := range devices(t) {
		rows, cols := 33000, 4096
		w, s := randomFP8(r, rows, cols, d.info.modes.denormPreserve)
		x := mxInput(r, cols)
		var a num.ActFP8
		num.QuantizeActFP8(x, &a)
		m, err := d.Upload(rows, cols, Format{FP8: w, E8M0: s})
		if err != nil {
			t.Fatal(err)
		}
		got, want := make([]float32, rows), make([]float32, rows)
		start := time.Now()
		if err := m.MulVec(got, x); err != nil {
			t.Fatal(err)
		}
		el := time.Since(start)
		num.MatVecFP8(want, w, s, &a)
		sameBits(t, d.Info().Name, got, want)
		checkUnrounded(t, d.Info().Name, m, x, func(out []float32) { num.MatVecFP8Unrounded(out, w, s, &a) })
		t.Logf("%s: %d×%d FP8 in %d parts, %s", d.Info().Name, rows, cols, len(m.parts), el)
		m.Free()
	}
}

// Partial sums stay exact through cancellation: 15 products of 896, one
// of 225·2^-13, then 15 of -896. Were the large products summed with the
// small one, 13440 + 225·2^-13 would round to a multiple of 2^-10.
func TestFP8SumExactThroughCancellation(t *testing.T) {
	const cols = 32
	x := make([]float32, cols)
	w := make([]byte, cols)
	x[0] = 448 // the scale is 1; its weight is 0
	for i := 1; i < 32; i++ {
		x[i], w[i] = 256, num.EncodeE4M3(3.5)
		if i > 16 {
			w[i] |= 0x80
		}
	}
	x[16], w[16] = 0.9375, num.EncodeE4M3(15.0/512) // 15·2^-4 · 15·2^-9
	s := []byte{127}
	var a num.ActFP8
	num.QuantizeActFP8(x, &a)
	want := make([]float32, 1)
	num.MatVecFP8Unrounded(want, w, s, &a)
	if want[0] != float32(math.Ldexp(225, -13)) {
		t.Fatalf("CPU: %v", want[0])
	}
	for _, d := range devices(t) {
		m, err := d.Upload(1, cols, Format{FP8: w, E8M0: s})
		if err != nil {
			t.Fatal(err)
		}
		checkUnrounded(t, d.Info().Name, m, x, func(out []float32) { num.MatVecFP8Unrounded(out, w, s, &a) })
		m.Free()
	}
}

// Experts given in chunks are uploaded part by part, each part whole
// experts, as from one slice.
func TestFP4Chunks(t *testing.T) {
	r := rand.New(rand.NewPCG(41, 42))
	const rows, cols, per = 70000, 4096, 2000
	w := make([]byte, rows*cols/2)
	for i := range w {
		w[i] = byte(r.IntN(256))
	}
	s := make([]byte, rows*cols/32)
	randomScales(r, s, false)
	var wc, sc [][]byte
	for e := 0; e < rows; e += per {
		wc = append(wc, w[e*cols/2:(e+per)*cols/2])
		sc = append(sc, s[e*cols/32:(e+per)*cols/32])
	}
	x := mxInput(r, cols)
	var a num.ActFP8
	num.QuantizeActFP8(x, &a)
	want := make([]float32, rows)
	num.MatVecFP4(want, w, s, &a)
	for _, d := range devices(t) {
		m, err := d.Upload(rows, cols, Format{FP4Chunks: wc, E8M0Chunks: sc, AlignRows: per})
		if err != nil {
			t.Fatal(err)
		}
		for _, p := range m.parts {
			if p.row0%per != 0 || p.rows%per != 0 {
				t.Fatalf("part of rows %d..%d splits an expert", p.row0, p.row0+p.rows)
			}
		}
		got := make([]float32, rows)
		if err := m.MulVec(got, x); err != nil {
			t.Fatal(err)
		}
		sameBits(t, d.Info().Name, got, want)
		t.Logf("%s: %d parts", d.Info().Name, len(m.parts))
		m.Free()
	}
}

// Several inputs through one reading of the weights give each input's
// own product, bit for bit, unrounded too.
func TestMatVecsBatch(t *testing.T) {
	r := rand.New(rand.NewPCG(51, 52))
	for _, d := range devices(t) {
		for _, shape := range [][2]int{{7, 32}, {300, 896}, {1000, 1280}} {
			rows, cols := shape[0], shape[1]
			w, s := randomFP8(r, rows, cols, d.info.modes.denormPreserve)
			m, err := d.Upload(rows, cols, Format{FP8: w, E8M0: s})
			if err != nil {
				t.Fatal(err)
			}
			for _, raw := range []bool{false, true} {
				m.unrounded = raw
				for _, n := range []int{1, 3, 8, 11} {
					a, err := d.NewArena(n*cols+n*rows+n*cols+n*cols/32+64, n*(cols+rows)+64)
					if err != nil {
						t.Fatal(err)
					}
					xs := make([][]float32, n)
					b := a.Begin()
					for j := range n {
						xs[j] = mxInput(r, cols)
						b.Write(j*cols, xs[j])
					}
					out := n * cols
					b.MatVecsBatch(m, out, rows, 0, cols, n, out+n*rows, out+n*rows+n*cols)
					got := make([][]float32, n)
					for j := range n {
						got[j] = make([]float32, rows)
						b.Read(out+j*rows, got[j])
					}
					if err := b.Submit(); err != nil {
						t.Fatal(err)
					}
					for j := range n {
						var q num.ActFP8
						num.QuantizeActFP8(xs[j], &q)
						want := make([]float32, rows)
						if raw {
							num.MatVecFP8Unrounded(want, w, s, &q)
						} else {
							num.MatVecFP8(want, w, s, &q)
						}
						sameBits(t, fmt.Sprintf("%s %d×%d, input %d of %d, unrounded %v", d.Info().Name, rows, cols, j, n, raw), got[j], want)
					}
					a.Free()
				}
			}
			m.Free()
		}
	}
}
