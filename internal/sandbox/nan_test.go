package sandbox

import (
	"math"
	"testing"

	w "github.com/optimist3301/rakkan-mesh/internal/wasmtest"
)

const (
	canonNaN32 = 0x7fc00000
	canonNaN64 = 0x7ff8000000000000
)

// Opcodes used below.
var (
	f32Add       = []byte{0x92}
	f32Mul       = []byte{0x94}
	f32Div       = []byte{0x95}
	f32Sqrt      = []byte{0x91}
	f32Neg       = []byte{0x8c}
	f64Div       = []byte{0xa3}
	f32Demote    = []byte{0xb6}
	i32ReinterF  = []byte{0xbc}
	i64ReinterF  = []byte{0xbd}
	i64Eq        = []byte{0x51}
	f32x4Div     = []byte{0xfd, 0xe7, 0x01}
	f32x4Extract = func(lane byte) []byte { return []byte{0xfd, 0x1f, lane} }
	f64x2Div     = []byte{0xfd, 0xf3, 0x01}
	f64x2Extract = func(lane byte) []byte { return []byte{0xfd, 0x21, lane} }
)

func exitCode(t *testing.T, body []byte) uint32 {
	t.Helper()
	res, err := run(t, taskModule(1, nil, body).Encode(), nil, testLimits())
	if err != nil {
		t.Fatal(err)
	}
	return res.ExitCode
}

func TestNaNCanonicalization(t *testing.T) {
	f32 := func(v float32) uint32 { return math.Float32bits(v) }
	cases := []struct {
		name string
		body []byte
		want uint32
	}{
		// x86 produces 0xffc00000 here natively.
		{"f32 0/0", w.Cat(w.F32Const(0), w.F32Const(0), f32Div, i32ReinterF), canonNaN32},
		{"f32 sqrt(-1)", w.Cat(w.F32Const(f32(-1)), f32Sqrt, i32ReinterF), canonNaN32},
		// Hardware propagates (and quiets) input payloads.
		{"f32 NaN payload", w.Cat(w.F32Const(0xffa00001), w.F32Const(f32(1)), f32Add, i32ReinterF), canonNaN32},
		{"f32.demote NaN payload", w.Cat(w.F64Const(0xfff0000000000001), f32Demote, i32ReinterF), canonNaN32},
		// Ordinary values, including signed zero, are untouched.
		{"f32 1/4", w.Cat(w.F32Const(f32(1)), w.F32Const(f32(4)), f32Div, i32ReinterF), f32(0.25)},
		{"f32 -0*1", w.Cat(w.F32Const(f32(float32(math.Copysign(0, -1)))), w.F32Const(f32(1)), f32Mul, i32ReinterF), 0x80000000},
		// Bitwise operators keep NaN payloads by specification.
		{"f32.neg keeps payload", w.Cat(w.F32Const(0x7fa00001), f32Neg, i32ReinterF), 0xffa00001},
		{"f64 0/0", w.Cat(w.F64Const(0), w.F64Const(0), f64Div, i64ReinterF, w.I64Const(canonNaN64), i64Eq), 1},
		{"f32x4 NaN lane", w.Cat(w.V128Const([4]uint32{f32(1), 0, f32(3), 0xffa00001}), w.V128Const([4]uint32{f32(2), 0, f32(3), f32(1)}), f32x4Div, f32x4Extract(1), i32ReinterF), canonNaN32},
		{"f32x4 payload lane", w.Cat(w.V128Const([4]uint32{f32(1), 0, f32(3), 0xffa00001}), w.V128Const([4]uint32{f32(2), 0, f32(3), f32(1)}), f32x4Div, f32x4Extract(3), i32ReinterF), canonNaN32},
		{"f32x4 ordinary lane", w.Cat(w.V128Const([4]uint32{f32(1), 0, f32(3), 0xffa00001}), w.V128Const([4]uint32{f32(2), 0, f32(3), f32(1)}), f32x4Div, f32x4Extract(0), i32ReinterF), f32(0.5)},
		{"f64x2 NaN lane", w.Cat(w.V128Const([4]uint32{}), w.V128Const([4]uint32{}), f64x2Div, f64x2Extract(1), i64ReinterF, w.I64Const(canonNaN64), i64Eq), 1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := exitCode(t, c.body); got != c.want {
				t.Fatalf("got 0x%08x, want 0x%08x", got, c.want)
			}
		})
	}
}
