package sandbox

import (
	"math"
	"math/rand/v2"
	"testing"
)

func newState() *execState {
	l := DefaultLimits()
	l.MaxMemoryPages = 256
	return &execState{lim: l}
}

func randTensor(s *execState, rng *rand.Rand, rows, cols int) int32 {
	t := s.newTensor(dtF32, rows, cols)
	for i := range t.f32 {
		t.f32[i] = float32(rng.NormFloat64())
	}
	return s.addTensor(t)
}

// refMatmul is the reference: the same summation order and rounding. The
// explicit conversion matters here too: Go may fuse a multiply and an add
// even across statements, unless a conversion rounds in between.
func refMatmul(a, b *tensor) []float32 {
	out := make([]float32, a.rows*b.cols)
	for i := range a.rows {
		for j := range b.cols {
			var acc float32
			for p := range a.cols {
				acc += float32(a.f32[i*a.cols+p] * b.f32[p*b.cols+j])
			}
			out[i*b.cols+j] = acc
		}
	}
	return out
}

func TestMatmulExactAndTransposedAgrees(t *testing.T) {
	s := newState()
	rng := rand.New(rand.NewPCG(1, 2))
	a, b := randTensor(s, rng, 7, 33), randTensor(s, rng, 33, 5)
	c := s.tensor(s.tenMatmul(a, b, false))
	want := refMatmul(s.tensor(a), s.tensor(b))
	for i, v := range c.f32 {
		if math.Float32bits(v) != math.Float32bits(want[i]) {
			t.Fatalf("element %d: %v, reference %v", i, v, want[i])
		}
	}
	// b transposed, multiplied with matmul_t, gives the same bits.
	bt := s.newTensor(dtF32, 5, 33)
	for p := range 33 {
		for j := range 5 {
			bt.f32[j*33+p] = s.tensor(b).f32[p*5+j]
		}
	}
	ct := s.tensor(s.tenMatmul(a, s.addTensor(bt), true))
	for i := range c.f32 {
		if math.Float32bits(c.f32[i]) != math.Float32bits(ct.f32[i]) {
			t.Fatalf("matmul_t differs at %d", i)
		}
	}
	if s.tenMatmul(a, a, false) != tenErrArgs {
		t.Fatal("shape mismatch accepted")
	}
}

func TestInt8MatmulIsExact(t *testing.T) {
	s := newState()
	a, b := s.newTensor(dtI8, 3, 64), s.newTensor(dtI8, 64, 2)
	var want [6]int32
	for i := range a.i8 {
		a.i8[i] = int8(i%255 - 127)
	}
	for i := range b.i8 {
		b.i8[i] = int8(127 - i%200)
	}
	for i := range 3 {
		for j := range 2 {
			for p := range 64 {
				want[i*2+j] += int32(a.i8[i*64+p]) * int32(b.i8[p*2+j])
			}
		}
	}
	c := s.tensor(s.tenMatmul(s.addTensor(a), s.addTensor(b), false))
	if c.dtype != dtI32 {
		t.Fatalf("dtype %d", c.dtype)
	}
	for i, v := range c.i32 {
		if v != want[i] {
			t.Fatalf("element %d: %d, want %d", i, v, want[i])
		}
	}
}

func TestQuantizeRoundsAndClamps(t *testing.T) {
	s := newState()
	a := s.newTensor(dtF32, 1, 5)
	copy(a.f32, []float32{0.25, 0.75, -0.25, 1000, -1000})
	q := s.tensor(s.tenQuantize(s.addTensor(a), 0.5))
	want := []int8{0, 2, 0, 127, -127} // round half to even: 0.5 -> 0, 1.5 -> 2
	for i, v := range q.i8 {
		if v != want[i] {
			t.Fatalf("quantized %v, want %v", q.i8, want)
		}
	}
}

func TestExp32(t *testing.T) {
	if exp32(0) != 1 {
		t.Fatalf("exp32(0) = %v", exp32(0))
	}
	worst := 0.0
	for x := float32(-100); x < 88; x += 0.0137 {
		got, want := float64(exp32(x)), math.Exp(float64(x))
		if want < 1e-37 { // subnormal range: absolute error matters
			continue
		}
		worst = max(worst, math.Abs(got-want)/want)
	}
	if worst > 4e-7 {
		t.Fatalf("worst relative error %.3g", worst)
	}
	if !math.IsInf(float64(exp32(100)), 1) || exp32(-200) != 0 {
		t.Fatal("overflow or underflow mishandled")
	}
}

func TestSoftmaxAddReluScale(t *testing.T) {
	s := newState()
	a := s.newTensor(dtF32, 2, 3)
	copy(a.f32, []float32{1, 2, 3, -1, 0, 1000})
	sm := s.tensor(s.tenSoftmax(s.addTensor(a)))
	for r := range 2 {
		var sum float32
		for _, v := range sm.f32[r*3 : r*3+3] {
			sum += v
		}
		if math.Abs(float64(sum)-1) > 1e-6 {
			t.Fatalf("row %d sums to %v", r, sum)
		}
	}
	if !(sm.f32[0] < sm.f32[1] && sm.f32[1] < sm.f32[2]) || sm.f32[5] != 1 {
		t.Fatalf("softmax %v", sm.f32)
	}

	bias := s.newTensor(dtF32, 1, 3)
	copy(bias.f32, []float32{10, -10, 0})
	sum := s.tensor(s.tenAdd(s.addTensor(a), s.addTensor(bias)))
	relu := s.tensor(s.tenRelu(s.addTensor(sum)))
	want := []float32{11, 0, 3, 9, 0, 1000}
	for i, v := range relu.f32 {
		if v != want[i] {
			t.Fatalf("relu(a + bias) = %v, want %v", relu.f32, want)
		}
	}
	acc := s.newTensor(dtI32, 1, 2)
	copy(acc.i32, []int32{4, -6})
	if d := s.tensor(s.tenScale(s.addTensor(acc), 0.5)); d.f32[0] != 2 || d.f32[1] != -3 {
		t.Fatalf("dequantized %v", d.f32)
	}
}

func TestTensorNaNIsCanonical(t *testing.T) {
	s := newState()
	a, b := s.newTensor(dtF32, 1, 1), s.newTensor(dtF32, 1, 1)
	a.f32[0], b.f32[0] = float32(math.Inf(1)), 0
	c := s.tensor(s.tenMatmul(s.addTensor(a), s.addTensor(b), false))
	if math.Float32bits(c.f32[0]) != 0x7fc00000 {
		t.Fatalf("inf*0 = %08x, want canonical NaN", math.Float32bits(c.f32[0]))
	}
}

func TestTensorBudget(t *testing.T) {
	s := newState() // 16 MiB budget
	if s.newTensor(dtF32, 2049, 2048) != nil {
		t.Fatal("tensor over the 16 MiB budget accepted")
	}
	h := s.addTensor(s.newTensor(dtF32, 1024, 1024)) // 4 MiB
	if s.tenAdd(h, h) < 0 || s.tenAdd(h, h) < 0 || s.tenAdd(h, h) < 0 {
		t.Fatal("results within budget refused")
	}
	if got := s.tenAdd(h, h); got != tenErrTooBig {
		t.Fatalf("fifth 4 MiB tensor: %d, want tenErrTooBig", got)
	}
	s.tenFree(h)
	if got := s.tenAdd(2, 3); got < 0 {
		t.Fatalf("after free: %d", got)
	}
}
