package sandbox

import (
	"context"
	"encoding/binary"
	"math"

	"github.com/optimist3301/rakkan-mesh/internal/num"
	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
)

// Tensor kernels: matrix operations for machine-learning inference, run as
// host code. Weights typically come from an attached blob (from_blob), read
// host-side without passing through the task's memory.
//
// Tensors are matrices (a vector is 1 x n) of float32, int8 or int32,
// row-major, little-endian, held in a per-task table by handle and counted
// against the same budget as images.
//
// Results are bit-identical on every architecture, which replicated
// verification depends on:
//   - every float32 product is converted explicitly before being added:
//     the Go specification lets compilers fuse x*y+z into one FMA, which
//     rounds once instead of twice and which arm64 compilers do emit; an
//     explicit conversion forbids the fusion;
//   - sums run in a fixed order;
//   - exp, for softmax, comes from package num rather than math, which
//     uses assembly on some architectures;
//   - NaN results are canonicalized, as in WebAssembly code;
//   - int8 matrix products accumulate in int32, exactly: the arithmetic of
//     quantized inference.
//
// Functions return a handle or a count, or a negative error code.
const (
	TensorModule     = "rakkan_tensor_v1"
	TensorCapability = "tensor/1"
)

// Data types.
const (
	dtF32 = 0
	dtI8  = 1
	dtI32 = 2
)

// Gas, calibrated like WebAssembly instructions: a multiply-accumulate
// takes about 1.2 ns, so a 256^3 product costs as much gas as the time the
// same product in plain WebAssembly costs about 8 times.
const (
	macGas     = 8  // per multiply-accumulate
	elemGas    = 4  // per element of elementwise operations
	softmaxGas = 40 // per element: exp and division
)

const (
	tenErrArgs   = -1 // bad handle, arguments, shapes or types
	tenErrTooBig = -3 // over the memory budget
	tenErrBlob   = -5 // blob too short for the requested tensor
	maxTensorDim = 1 << 24
)

var tensorABI = map[string]signature{
	// from_memory(src, dtype, rows, cols), from_blob(index, offset, dtype,
	// rows, cols) -> handle
	"from_memory": sig([]api.ValueType{i32, i32, i32, i32}, i32),
	"from_blob":   sig([]api.ValueType{i32, i64, i32, i32, i32}, i32),
	// to_memory(h, dst, len) -> bytes written
	"to_memory": sig([]api.ValueType{i32, i32, i32}, i32),
	"rows":      sig([]api.ValueType{i32}, i32),
	"cols":      sig([]api.ValueType{i32}, i32),
	"dtype":     sig([]api.ValueType{i32}, i32),
	// matmul(a, b): a[m,k] x b[k,n]; matmul_t(a, b): a[m,k] x b[n,k]^T.
	// f32 x f32 -> f32, i8 x i8 -> i32.
	"matmul":   sig([]api.ValueType{i32, i32}, i32),
	"matmul_t": sig([]api.ValueType{i32, i32}, i32),
	// add(a, b): same shape, or b a single row added to every row of a.
	"add": sig([]api.ValueType{i32, i32}, i32),
	// scale(a, s): a * s, as f32 (an i32 tensor is dequantized).
	"scale": sig([]api.ValueType{i32, f32}, i32),
	"relu":  sig([]api.ValueType{i32}, i32),
	// softmax(a): row-wise, f32.
	"softmax": sig([]api.ValueType{i32}, i32),
	// quantize(a, s): round(a / s) clamped to int8.
	"quantize": sig([]api.ValueType{i32, f32}, i32),
	// argmax(h, dst, len): the column of each row's maximum, as i32s
	// written to dst -> rows.
	"argmax": sig([]api.ValueType{i32, i32, i32}, i32),
	"free":   sig([]api.ValueType{i32}, i32),
}

type tensor struct {
	dtype      int32
	rows, cols int
	f32        []float32
	i8         []int8
	i32        []int32
}

func (t *tensor) size() int { return t.rows * t.cols }

func elemSize(dtype int32) int64 {
	switch dtype {
	case dtF32, dtI32:
		return 4
	case dtI8:
		return 1
	}
	return 0
}

type tensorTable struct {
	tensors map[int32]*tensor
	next    int32
}

func (s *execState) tensor(h int32) *tensor { return s.tensors.tensors[h] }

// newTensor allocates a zeroed tensor within the budget.
func (s *execState) newTensor(dtype int32, rows, cols int) *tensor {
	es := elemSize(dtype)
	if es == 0 || rows <= 0 || cols <= 0 || rows > maxTensorDim || cols > maxTensorDim {
		return nil
	}
	n := int64(rows) * int64(cols)
	if !s.fits(n * es) {
		return nil
	}
	t := &tensor{dtype: dtype, rows: rows, cols: cols}
	switch dtype {
	case dtF32:
		t.f32 = make([]float32, n)
	case dtI8:
		t.i8 = make([]int8, n)
	case dtI32:
		t.i32 = make([]int32, n)
	}
	s.hostBytes += n * es
	return t
}

func (s *execState) addTensor(t *tensor) int32 {
	tt := &s.tensors
	if tt.tensors == nil {
		tt.tensors = map[int32]*tensor{}
	}
	tt.next++
	tt.tensors[tt.next] = t
	return tt.next
}

// result stores t, or reports why it could not be made.
func (s *execState) result(t *tensor) int32 {
	if t == nil {
		return tenErrTooBig
	}
	return s.addTensor(t)
}

func (s *execState) instantiateTensor(ctx context.Context, rt wazero.Runtime) error {
	b := rt.NewHostModuleBuilder(TensorModule)
	fns := map[string]any{
		"from_memory": s.tenFromMemory,
		"from_blob":   s.tenFromBlob,
		"to_memory":   s.tenToMemory,
		"rows":        func(h int32) int32 { return s.tenDim(h, 0) },
		"cols":        func(h int32) int32 { return s.tenDim(h, 1) },
		"dtype":       func(h int32) int32 { return s.tenDim(h, 2) },
		"matmul":      func(a, b int32) int32 { return s.tenMatmul(a, b, false) },
		"matmul_t":    func(a, b int32) int32 { return s.tenMatmul(a, b, true) },
		"add":         s.tenAdd,
		"scale":       s.tenScale,
		"relu":        s.tenRelu,
		"softmax":     s.tenSoftmax,
		"quantize":    s.tenQuantize,
		"argmax":      s.tenArgmax,
		"free":        s.tenFree,
	}
	for name, fn := range fns {
		b = b.NewFunctionBuilder().WithFunc(fn).Export(name)
	}
	_, err := b.Instantiate(ctx)
	return err
}

// decodeInto fills t from little-endian bytes.
func decodeInto(t *tensor, raw []byte) {
	switch t.dtype {
	case dtF32:
		for i := range t.f32 {
			t.f32[i] = num.Canonical(math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:])))
		}
	case dtI8:
		for i := range t.i8 {
			t.i8[i] = int8(raw[i])
		}
	case dtI32:
		for i := range t.i32 {
			t.i32[i] = int32(binary.LittleEndian.Uint32(raw[4*i:]))
		}
	}
}

func (s *execState) tenFromMemory(_ context.Context, m api.Module, src uint32, dtype, rows, cols int32) int32 {
	s.charge(0)
	t := s.newTensor(dtype, int(rows), int(cols))
	if t == nil {
		return tenErrArgs
	}
	n := uint32(int64(t.size()) * elemSize(dtype))
	s.charge(n)
	raw, ok := m.Memory().Read(src, n)
	if !ok {
		s.trap(errMemory("from_memory", src, n))
	}
	decodeInto(t, raw)
	return s.addTensor(t)
}

func (s *execState) tenFromBlob(index uint32, offset int64, dtype, rows, cols int32) int32 {
	s.charge(0)
	if index >= uint32(len(s.blobs)) || offset < 0 {
		return tenErrArgs
	}
	t := s.newTensor(dtype, int(rows), int(cols))
	if t == nil {
		return tenErrTooBig
	}
	n := int64(t.size()) * elemSize(dtype)
	b := s.blobs[index]
	if offset+n > b.Size() {
		s.freeTensor(t)
		return tenErrBlob
	}
	s.chargeUnits(n >> 4)
	raw := make([]byte, n)
	if got, err := b.ReadAt(raw, offset); int64(got) < n {
		s.trap(errBlob(index, err))
	}
	decodeInto(t, raw)
	return s.addTensor(t)
}

func (s *execState) tenToMemory(_ context.Context, m api.Module, h int32, dst, n uint32) int32 {
	t := s.tensor(h)
	if t == nil {
		s.charge(0)
		return tenErrArgs
	}
	size := int64(t.size()) * elemSize(t.dtype)
	if int64(n) < size {
		s.charge(0)
		return tenErrArgs
	}
	s.charge(uint32(size))
	raw := make([]byte, size)
	switch t.dtype {
	case dtF32:
		for i, v := range t.f32 {
			binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
		}
	case dtI8:
		for i, v := range t.i8 {
			raw[i] = byte(v)
		}
	case dtI32:
		for i, v := range t.i32 {
			binary.LittleEndian.PutUint32(raw[4*i:], uint32(v))
		}
	}
	if !m.Memory().Write(dst, raw) {
		s.trap(errMemory("to_memory", dst, uint32(size)))
	}
	return int32(size)
}

func (s *execState) tenDim(h int32, which int) int32 {
	s.charge(0)
	t := s.tensor(h)
	if t == nil {
		return tenErrArgs
	}
	return [...]int32{int32(t.rows), int32(t.cols), t.dtype}[which]
}

func (s *execState) tenMatmul(ha, hb int32, transposed bool) int32 {
	s.charge(0)
	a, b := s.tensor(ha), s.tensor(hb)
	if a == nil || b == nil || a.dtype != b.dtype || a.dtype == dtI32 {
		return tenErrArgs
	}
	m, k := a.rows, a.cols
	n := b.cols
	if transposed {
		n = b.rows
		if b.cols != k {
			return tenErrArgs
		}
	} else if b.rows != k {
		return tenErrArgs
	}
	out := dtF32
	if a.dtype == dtI8 {
		out = dtI32
	}
	c := s.newTensor(int32(out), m, n)
	if c == nil {
		return tenErrTooBig
	}
	s.chargeUnits(int64(m) * int64(n) * int64(k) * macGas)
	// bAt returns element (row p of the k dimension, column j).
	stride, step := n, 1 // b[k,n]: element (p, j) at p*n + j
	if transposed {
		stride, step = 1, k // b[n,k]: element (p, j) at j*k + p
	}
	for i := range m {
		for j := range n {
			bj := j * step
			if a.dtype == dtF32 {
				var acc float32
				ar := a.f32[i*k : i*k+k]
				for p, av := range ar {
					// The conversion rounds the product, forbidding fusion
					// into an FMA (see the package comment).
					acc += float32(av * b.f32[p*stride+bj])
				}
				c.f32[i*n+j] = num.Canonical(acc)
			} else {
				var acc int32
				ar := a.i8[i*k : i*k+k]
				for p, av := range ar {
					acc += int32(av) * int32(b.i8[p*stride+bj])
				}
				c.i32[i*n+j] = acc
			}
		}
	}
	return s.addTensor(c)
}

func (s *execState) tenAdd(ha, hb int32) int32 {
	s.charge(0)
	a, b := s.tensor(ha), s.tensor(hb)
	if a == nil || b == nil || a.dtype != b.dtype || a.dtype == dtI8 || a.cols != b.cols || b.rows != a.rows && b.rows != 1 {
		return tenErrArgs
	}
	c := s.newTensor(a.dtype, a.rows, a.cols)
	if c == nil {
		return tenErrTooBig
	}
	s.chargeUnits(int64(a.size()) * elemGas)
	for i := range a.size() {
		bi := i
		if b.rows == 1 {
			bi = i % a.cols
		}
		if a.dtype == dtF32 {
			c.f32[i] = num.Canonical(a.f32[i] + b.f32[bi])
		} else {
			c.i32[i] = a.i32[i] + b.i32[bi]
		}
	}
	return s.addTensor(c)
}

func (s *execState) tenScale(h int32, k float32) int32 {
	s.charge(0)
	a := s.tensor(h)
	if a == nil || a.dtype == dtI8 {
		return tenErrArgs
	}
	c := s.newTensor(dtF32, a.rows, a.cols)
	if c == nil {
		return tenErrTooBig
	}
	s.chargeUnits(int64(a.size()) * elemGas)
	for i := range c.f32 {
		v := float32(0)
		if a.dtype == dtF32 {
			v = a.f32[i]
		} else {
			v = float32(a.i32[i])
		}
		c.f32[i] = num.Canonical(v * k)
	}
	return s.addTensor(c)
}

func (s *execState) tenRelu(h int32) int32 {
	s.charge(0)
	a := s.tensor(h)
	if a == nil || a.dtype == dtI8 {
		return tenErrArgs
	}
	c := s.newTensor(a.dtype, a.rows, a.cols)
	if c == nil {
		return tenErrTooBig
	}
	s.chargeUnits(int64(a.size()) * elemGas)
	for i := range a.size() {
		if a.dtype == dtF32 {
			if v := a.f32[i]; v > 0 {
				c.f32[i] = v
			} else if v != v {
				c.f32[i] = num.CanonicalNaN32
			}
		} else if v := a.i32[i]; v > 0 {
			c.i32[i] = v
		}
	}
	return s.addTensor(c)
}

func (s *execState) tenSoftmax(h int32) int32 {
	s.charge(0)
	a := s.tensor(h)
	if a == nil || a.dtype != dtF32 {
		return tenErrArgs
	}
	c := s.newTensor(dtF32, a.rows, a.cols)
	if c == nil {
		return tenErrTooBig
	}
	s.chargeUnits(int64(a.size()) * softmaxGas)
	for r := range a.rows {
		row, out := a.f32[r*a.cols:(r+1)*a.cols], c.f32[r*a.cols:(r+1)*a.cols]
		mx := row[0]
		for _, v := range row[1:] {
			if v > mx {
				mx = v
			}
		}
		var sum float32
		for i, v := range row {
			out[i] = num.Exp32(v - mx)
			sum += out[i]
		}
		for i := range out {
			out[i] = num.Canonical(out[i] / sum)
		}
	}
	return s.addTensor(c)
}

func (s *execState) tenQuantize(h int32, k float32) int32 {
	s.charge(0)
	a := s.tensor(h)
	if a == nil || a.dtype != dtF32 || !(k > 0) {
		return tenErrArgs
	}
	c := s.newTensor(dtI8, a.rows, a.cols)
	if c == nil {
		return tenErrTooBig
	}
	s.chargeUnits(int64(a.size()) * elemGas)
	for i, v := range a.f32 {
		q := float32(math.RoundToEven(float64(v / k)))
		c.i8[i] = int8(max(-127, min(127, q)))
	}
	return s.addTensor(c)
}

func (s *execState) tenArgmax(_ context.Context, m api.Module, h int32, dst, n uint32) int32 {
	a := s.tensor(h)
	if a == nil || a.dtype == dtI8 || uint64(n) < uint64(a.rows)*4 {
		s.charge(0)
		return tenErrArgs
	}
	s.charge(uint32(a.rows) * 4)
	s.chargeUnits(int64(a.size()) * elemGas)
	out := make([]byte, a.rows*4)
	for r := range a.rows {
		best := 0
		for j := 1; j < a.cols; j++ {
			i := r*a.cols + j
			if a.dtype == dtF32 && a.f32[i] > a.f32[r*a.cols+best] || a.dtype == dtI32 && a.i32[i] > a.i32[r*a.cols+best] {
				best = j
			}
		}
		binary.LittleEndian.PutUint32(out[4*r:], uint32(best))
	}
	if !m.Memory().Write(dst, out) {
		s.trap(errMemory("argmax", dst, uint32(len(out))))
	}
	return int32(a.rows)
}

func (s *execState) freeTensor(t *tensor) {
	s.hostBytes -= int64(t.size()) * elemSize(t.dtype)
}

func (s *execState) tenFree(h int32) int32 {
	s.charge(0)
	t := s.tensor(h)
	if t == nil {
		return tenErrArgs
	}
	s.freeTensor(t)
	delete(s.tensors.tensors, h)
	return 0
}
