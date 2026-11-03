package num

import (
	"compress/gzip"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math"
	"math/rand/v2"
	"os"
	"sync"
	"testing"
)

// Vectors from tools/llmref/dsv41/num_vectors.py: the reference kernels'
// answers (kernel_cpu), as bit patterns.
type mxVectors struct {
	RoundE4M3 struct{ In, Out string } `json:"round_e4m3"`
	ActQuant  []struct {
		Rows, Cols           int
		X, Q, Scale, Inplace string
	} `json:"act_quant"`
	FP4Quant []struct {
		Block                     int
		ScaleKind                 string `json:"scale_kind"`
		Rows, Cols                int
		X, Packed, Scale, Inplace string
	} `json:"fp4_quant"`
	FP8Gemm []mxGemm `json:"fp8_gemm"`
	FP4Gemm []mxGemm `json:"fp4_gemm"`
}

type mxGemm struct {
	M, N, K int
	X, W, Y string
	Y32     string
	WScale  string `json:"w_scale"`
}

var mxOnce = sync.OnceValues(func() (*mxVectors, error) {
	f, err := os.Open("testdata/dsv41-mx.json.gz")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	z, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	var v mxVectors
	return &v, json.NewDecoder(z).Decode(&v)
})

func mxLoad(t *testing.T) *mxVectors {
	t.Helper()
	v, err := mxOnce()
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func b64(t *testing.T, s string) []byte {
	t.Helper()
	b, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// bf16s decodes little-endian bfloat16s to float32.
func bf16s(t *testing.T, s string) []float32 {
	b := b64(t, s)
	out := make([]float32, len(b)/2)
	for i := range out {
		out[i] = BF16(binary.LittleEndian.Uint16(b[2*i:]))
	}
	return out
}

func f32s(t *testing.T, s string) []float32 {
	b := b64(t, s)
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

func TestE4M3Codes(t *testing.T) {
	nans := 0
	for c := range 256 {
		v := DecodeE4M3(uint8(c))
		if v != v {
			nans++
			continue
		}
		if got := EncodeE4M3(v); got != uint8(c) {
			t.Errorf("code %#x = %g encodes as %#x", c, v, got)
		}
	}
	if nans != 2 || DecodeE4M3(0x7e) != 448 || DecodeE4M3(0x01) != 1.0/512 || DecodeE4M3(0x38) != 1 {
		t.Errorf("table: %d NaNs, 0x7e=%g 0x01=%g 0x38=%g", nans, DecodeE4M3(0x7e), DecodeE4M3(0x01), DecodeE4M3(0x38))
	}
	if EncodeE4M3(1000) != 0x7e || EncodeE4M3(-1000) != 0xfe {
		t.Error("no saturation")
	}
}

func TestEncodeE4M3MatchesReference(t *testing.T) {
	v := mxLoad(t)
	in, out := f32s(t, v.RoundE4M3.In), b64(t, v.RoundE4M3.Out)
	bad := 0
	for i, x := range in {
		if got := EncodeE4M3(x); got != out[i] {
			if bad++; bad <= 5 {
				t.Errorf("%g (%#08x): got %#x, reference %#x", x, math.Float32bits(x), got, out[i])
			}
		}
	}
	if bad > 0 {
		t.Errorf("%d of %d differ", bad, len(in))
	}
}

func TestEncodeE2M1(t *testing.T) {
	// every multiple of 1/64 in [-6, 6], midpoints included: nearest, ties to even code
	for i := -6 * 64; i <= 6*64; i++ {
		x := float32(i) / 64
		best := 0
		for c := 1; c < 8; c++ {
			db, dc := math.Abs(float64(fp4Grid[best]-abs32(x))), math.Abs(float64(fp4Grid[c]-abs32(x)))
			if dc < db || dc == db && c%2 == 0 && best%2 == 1 {
				best = c
			}
		}
		got := EncodeE2M1(x)
		if int(got&7) != best || (got&8 != 0) != math.Signbit(float64(x)) {
			t.Errorf("%g: code %#x, want magnitude code %d", x, got, best)
		}
		if DecodeE2M1(got) != float32(math.Copysign(float64(fp4Grid[best]), float64(x))) {
			t.Errorf("%g: decodes to %g", x, DecodeE2M1(got))
		}
	}
}

func abs32(x float32) float32 { return float32(math.Abs(float64(x))) }

func TestDecodeE8M0(t *testing.T) {
	for c := range 255 {
		if got, want := DecodeE8M0(uint8(c)), float32(math.Ldexp(1, c-127)); got != want {
			t.Errorf("%d: %g, want %g", c, got, want)
		}
	}
	if v := DecodeE8M0(255); v == v {
		t.Error("255 is not NaN")
	}
}

func TestActQuantMatchesReference(t *testing.T) {
	for n, c := range mxLoad(t).ActQuant {
		x, q, scales, inplace := bf16s(t, c.X), b64(t, c.Q), b64(t, c.Scale), bf16s(t, c.Inplace)
		for r := range c.Rows {
			row := x[r*c.Cols : (r+1)*c.Cols]
			var a ActFP8
			QuantizeActFP8(row, &a)
			for j, v := range a.Vals {
				if got := EncodeE4M3(float32(v)); got != q[r*c.Cols+j] {
					t.Fatalf("case %d row %d value %d (%g): code %#x, reference %#x", n, r, j, row[j], got, q[r*c.Cols+j])
				}
			}
			for b, s := range a.Scales {
				if got := uint8(math.Float32bits(s) >> 23); got != scales[r*c.Cols/QBlock+b] {
					t.Fatalf("case %d row %d block %d: scale byte %d, reference %d", n, r, b, got, scales[r*c.Cols/QBlock+b])
				}
			}
			y := append([]float32(nil), row...)
			ActQuantInPlace(y)
			for j := range y {
				if math.Float32bits(y[j]) != math.Float32bits(inplace[r*c.Cols+j]) {
					t.Fatalf("case %d row %d value %d: in place %g, reference %g", n, r, j, y[j], inplace[r*c.Cols+j])
				}
			}
		}
	}
}

func TestFP4QuantMatchesReference(t *testing.T) {
	for _, c := range mxLoad(t).FP4Quant {
		kind := FP4ScaleE8M0
		if c.ScaleKind == "e4m3" {
			kind = FP4ScaleE4M3
		}
		if kind.Block() != c.Block {
			t.Fatalf("%s: block %d, vectors say %d", c.ScaleKind, kind.Block(), c.Block)
		}
		x, packed, scales, inplace := bf16s(t, c.X), b64(t, c.Packed), b64(t, c.Scale), bf16s(t, c.Inplace)
		gotP, gotS := make([]byte, len(x)/2), make([]byte, len(x)/c.Block)
		QuantizeFP4(x, kind, gotP, gotS)
		for i := range gotS {
			if gotS[i] != scales[i] {
				t.Fatalf("%s: scale %d is %#x, reference %#x", c.ScaleKind, i, gotS[i], scales[i])
			}
		}
		for i := range gotP {
			if gotP[i] != packed[i] {
				t.Fatalf("%s: packed byte %d (values %g %g) is %#x, reference %#x", c.ScaleKind, i, x[2*i], x[2*i+1], gotP[i], packed[i])
			}
		}
		y := append([]float32(nil), x...)
		FP4QuantInPlace(y, kind)
		for i := range y {
			if math.Float32bits(y[i]) != math.Float32bits(inplace[i]) {
				t.Fatalf("%s: value %d (%g) in place %g, reference %g", c.ScaleKind, i, x[i], y[i], inplace[i])
			}
		}
	}
}

func checkGemm(t *testing.T, name string, cases []mxGemm, mv func(out []float32, w, s []byte, x *ActFP8, bf16 bool)) {
	for _, c := range cases {
		x, w, ws := bf16s(t, c.X), b64(t, c.W), b64(t, c.WScale)
		// y: the result; y32: the float32 sum before rounding, where the
		// order of adding blocks shows (bf16 rounding hides it almost always).
		for _, want := range []struct {
			bf16 bool
			y    []float32
		}{{true, bf16s(t, c.Y)}, {false, f32s(t, c.Y32)}} {
			out := make([]float32, c.N)
			var a ActFP8
			for r := range c.M {
				QuantizeActFP8(x[r*c.K:(r+1)*c.K], &a)
				mv(out, w, ws, &a, want.bf16)
				for i, v := range out {
					if ref := want.y[r*c.N+i]; math.Float32bits(v) != math.Float32bits(ref) {
						t.Fatalf("%s %dx%dx%d (bf16 %v): row %d output %d is %g, reference %g", name, c.M, c.N, c.K, want.bf16, r, i, v, ref)
					}
				}
			}
		}
	}
}

// paths runs f with the vector code, where the processor has it, and with
// the portable code.
func paths(t *testing.T, f func(t *testing.T)) {
	saved := useFMA
	defer func() { useFMA = saved }()
	if saved {
		t.Run("avx2", f)
	}
	useFMA = false
	t.Run("portable", f)
}

func TestMatVecFP8MatchesReference(t *testing.T) {
	paths(t, func(t *testing.T) { checkGemm(t, "fp8", mxLoad(t).FP8Gemm, matVecFP8) })
}

func TestMatVecFP4MatchesReference(t *testing.T) {
	paths(t, func(t *testing.T) { checkGemm(t, "fp4", mxLoad(t).FP4Gemm, matVecFP4) })
}

// The vector and portable block sums agree on every weight code (bar the
// E4M3 NaNs) and on activations spanning the whole E4M3 range, including
// odd block counts that leave a remainder for the portable code.
func TestBlockSumsPathsAgree(t *testing.T) {
	if !useFMA {
		t.Skip("no AVX2/FMA")
	}
	rng := rand.New(rand.NewPCG(5, 6))
	for _, nb := range []int{1, 3, 4, 7, 160} {
		x := make([]float32, nb*QBlock)
		for i := range x {
			x[i] = RoundBF16(float32(math.Ldexp(rng.NormFloat64(), rng.IntN(40)-20)))
		}
		var a ActFP8
		QuantizeActFP8(x, &a)
		w8 := make([]byte, nb*QBlock)
		for i := range w8 {
			for w8[i] = byte(rng.Uint32()); w8[i]&0x7f == 0x7f; w8[i] = byte(rng.Uint32()) {
			}
		}
		w4 := make([]byte, nb*QBlock/2)
		for i := range w4 {
			w4[i] = byte(rng.Uint32())
		}
		f8a, f8b := make([]float64, nb), make([]float64, nb)
		i4a, i4b := make([]int32, nb), make([]int32, nb)
		blockSumsFP8(w8, a.Vals, f8a)
		blockSumsFP4(w4, a.ints, i4a)
		useFMA = false
		blockSumsFP8(w8, a.Vals, f8b)
		blockSumsFP4(w4, a.ints, i4b)
		useFMA = true
		for b := range nb {
			if math.Float64bits(f8a[b]) != math.Float64bits(f8b[b]) {
				t.Fatalf("nb %d block %d: FP8 sum %g vector, %g portable", nb, b, f8a[b], f8b[b])
			}
			if i4a[b] != i4b[b] {
				t.Fatalf("nb %d block %d: FP4 sum %d vector, %d portable", nb, b, i4a[b], i4b[b])
			}
		}
	}
}

func TestValidE4M3(t *testing.T) {
	if !ValidE4M3([]byte{0, 0x7e, 0xfe, 0x80}) || ValidE4M3([]byte{1, 0x7f}) || ValidE4M3([]byte{0xff}) {
		t.Error("ValidE4M3")
	}
}

// The row split among goroutines must not change results: compare a
// product large enough to split with row-by-row products.
func TestMatVecFP4Split(t *testing.T) {
	rows, cols := 512, 2048
	rng := rand.New(rand.NewPCG(1, 2))
	w := make([]byte, rows*cols/2)
	for i := range w {
		w[i] = byte(rng.Uint32())
	}
	s := make([]byte, rows*cols/QBlock)
	for i := range s {
		s[i] = byte(120 + rng.IntN(10))
	}
	x := make([]float32, cols)
	for i := range x {
		x[i] = RoundBF16(float32(rng.NormFloat64()))
	}
	var a ActFP8
	QuantizeActFP8(x, &a)
	all := make([]float32, rows)
	MatVecFP4(all, w, s, &a)
	one := make([]float32, 1)
	for r := range rows {
		MatVecFP4(one, w[r*cols/2:], s[r*cols/QBlock:], &a)
		if one[0] != all[r] {
			t.Fatalf("row %d: %g alone, %g in the batch", r, one[0], all[r])
		}
	}
}

func benchMatVec(b *testing.B, rows, cols int, fp4 bool) {
	rng := rand.New(rand.NewPCG(3, 4))
	wb := rows * cols
	if fp4 {
		wb /= 2
	}
	w := make([]byte, wb)
	for i := range w {
		w[i] = byte(rng.Uint32()) & 0x7e // no E4M3 NaN codes
	}
	s := make([]byte, rows*cols/QBlock)
	for i := range s {
		s[i] = 120
	}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	var a ActFP8
	QuantizeActFP8(x, &a)
	out := make([]float32, rows)
	b.SetBytes(int64(wb))
	for b.Loop() {
		if fp4 {
			MatVecFP4(out, w, s, &a)
		} else {
			MatVecFP8(out, w, s, &a)
		}
	}
}

// Shapes of the real model: an expert's w1 (FP4) and an attention wq_b (FP8).
func BenchmarkMatVecFP4Expert(b *testing.B) { benchMatVec(b, 2304, 5120, true) }
func BenchmarkMatVecFP8WqB(b *testing.B)    { benchMatVec(b, 32768, 1280, false) }

// The same with the portable code: what processors without AVX2 and FMA get.
func BenchmarkMatVecFP4ExpertPortable(b *testing.B) { portable(b); benchMatVec(b, 2304, 5120, true) }
func BenchmarkMatVecFP8WqBPortable(b *testing.B)    { portable(b); benchMatVec(b, 32768, 1280, false) }

func portable(b *testing.B) {
	saved := useFMA
	useFMA = false
	b.Cleanup(func() { useFMA = saved })
}
