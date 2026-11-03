package dsv41

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/optimist3301/rakkan-mesh/internal/num"
)

// linearKind is how the reference multiplies by a weight matrix, which
// follows from the weight's type in the checkpoint and where it is used.
type linearKind int

const (
	// FP8 weights, 32x32 block scales: the input quantized to FP8 first,
	// the output rounded to bf16 (fp8_gemm).
	kindFP8 linearKind = iota
	// FP4 weights, scales per 32 of a row: likewise (fp4_gemm).
	kindFP4
	// bf16 weights and input, bf16 output (F.linear in bf16).
	kindBF16
	// bf16-stored weights the reference widens to float32, float32 input
	// and output (F.linear in float32).
	kindBF16F32
	// float32 weights, float32 input and output.
	kindF32
)

type linear struct {
	kind       linearKind
	rows, cols int
	q, scales  []byte    // FP8 codes or packed FP4, E8M0 scales
	bf         []uint16  // bf16 weights
	f          []float32 // float32 weights
}

// apply sets out (rows) to the product with x (cols); act is scratch for
// the quantized input.
func (l *linear) apply(out, x []float32, act *num.ActFP8) {
	switch l.kind {
	case kindFP8:
		num.QuantizeActFP8(x, act)
		num.MatVecFP8(out, l.q, l.scales, act)
	case kindFP4:
		num.QuantizeActFP8(x, act)
		num.MatVecFP4(out, l.q, l.scales, act)
	case kindBF16:
		num.MatVecBF16(out, l.bf, x)
		for i, v := range out {
			out[i] = bf16(v)
		}
	case kindBF16F32:
		num.MatVecBF16(out, l.bf, x)
	case kindF32:
		num.MatVec32(out, l.f, x)
	}
}

// applyRows is apply for rows [lo, hi) of an FP8 matrix, with the input
// already quantized: wo_a multiplies each group by its own block of rows.
func (l *linear) applyRowsFP8(out []float32, lo int, act *num.ActFP8) {
	nb := l.cols / num.QBlock
	num.MatVecFP8(out, l.q[lo*l.cols:], l.scales[(lo/num.QBlock)*nb:], act)
}

// weights reads tensors of a checkpoint by name.
type weights struct {
	t    map[string]*Tensor
	used map[string]bool
}

func (w *weights) get(name string) (*Tensor, error) {
	t, ok := w.t[name]
	if !ok {
		return nil, fmt.Errorf("dsv41: tensor %s missing", name)
	}
	w.used[name] = true
	return t, nil
}

func shapeIs(t *Tensor, dims ...int) bool {
	if len(t.Shape) != len(dims) {
		return false
	}
	for i, d := range dims {
		if t.Shape[i] != d {
			return false
		}
	}
	return true
}

// vec reads a 1-D tensor (bf16 or float32) as float32.
func (w *weights) vec(name string, n int) ([]float32, error) {
	t, err := w.get(name)
	if err != nil {
		return nil, err
	}
	if t.Elems() != n {
		return nil, fmt.Errorf("dsv41: tensor %s has shape %v, want %d elements", name, t.Shape, n)
	}
	return toF32(name, t)
}

func toF32(name string, t *Tensor) ([]float32, error) {
	out := make([]float32, t.Elems())
	switch t.DType {
	case "BF16":
		for i := range out {
			out[i] = num.BF16(binary.LittleEndian.Uint16(t.Data[2*i:]))
		}
	case "F32":
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(t.Data[4*i:]))
		}
	default:
		return nil, fmt.Errorf("dsv41: tensor %s has type %s, want BF16 or F32", name, t.DType)
	}
	return out, nil
}

func toBF16(t *Tensor) []uint16 {
	out := make([]uint16, t.Elems())
	for i := range out {
		out[i] = binary.LittleEndian.Uint16(t.Data[2*i:])
	}
	return out
}

// lin reads a weight matrix [rows, cols]. FP8 and FP4 matrices bring their
// scales; a bf16 matrix becomes kind, which must be kindBF16 or kindBF16F32
// (an FP8 or FP4 matrix ignores it).
func (w *weights) lin(name string, rows, cols int, kind linearKind) (*linear, error) {
	t, err := w.get(name + ".weight")
	if err != nil {
		return nil, err
	}
	l := &linear{kind: kind, rows: rows, cols: cols}
	switch t.DType {
	case "F8_E4M3":
		s, err := w.get(name + ".scale")
		if err != nil {
			return nil, err
		}
		if !shapeIs(t, rows, cols) || s.DType != "F8_E8M0" || !shapeIs(s, (rows+31)/32, cols/32) {
			return nil, fmt.Errorf("dsv41: %s: FP8 weight %v, scales %s %v for [%d, %d]", name, t.Shape, s.DType, s.Shape, rows, cols)
		}
		if !num.ValidE4M3(t.Data) {
			return nil, fmt.Errorf("dsv41: %s: NaN in FP8 weights", name)
		}
		l.kind, l.q, l.scales = kindFP8, t.Data, s.Data
	case "I8":
		s, err := w.get(name + ".scale")
		if err != nil {
			return nil, err
		}
		if !shapeIs(t, rows, cols/2) || s.DType != "F8_E8M0" || !shapeIs(s, rows, cols/32) {
			return nil, fmt.Errorf("dsv41: %s: FP4 weight %v, scales %s %v for [%d, %d]", name, t.Shape, s.DType, s.Shape, rows, cols)
		}
		l.kind, l.q, l.scales = kindFP4, t.Data, s.Data
	case "BF16":
		if !shapeIs(t, rows, cols) {
			return nil, fmt.Errorf("dsv41: %s: shape %v, want [%d, %d]", name, t.Shape, rows, cols)
		}
		if kind != kindBF16 && kind != kindBF16F32 {
			return nil, fmt.Errorf("dsv41: %s: unexpected bf16 weight", name)
		}
		l.bf = toBF16(t)
	case "F32":
		if !shapeIs(t, rows, cols) {
			return nil, fmt.Errorf("dsv41: %s: shape %v, want [%d, %d]", name, t.Shape, rows, cols)
		}
		f, _ := toF32(name, t)
		l.kind, l.f = kindF32, f
	default:
		return nil, fmt.Errorf("dsv41: %s: weight type %s", name, t.DType)
	}
	return l, nil
}
