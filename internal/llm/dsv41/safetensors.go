package dsv41

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
)

// Tensor is a tensor of a safetensors file: its type, shape and bytes.
type Tensor struct {
	DType string
	Shape []int
	Data  []byte
}

// Elems is the number of elements.
func (t *Tensor) Elems() int {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

var dtypeSize = map[string]int{
	"F32": 4, "BF16": 2, "F16": 2, "I64": 8, "I32": 4, "I8": 1, "U8": 1, "BOOL": 1,
	"F8_E4M3": 1, "F8_E8M0": 1,
}

// NewTensor makes a tensor of data, checking its size against the shape.
func NewTensor(dtype string, shape []int, data []byte) (*Tensor, error) {
	t := &Tensor{DType: dtype, Shape: shape, Data: data}
	size, ok := dtypeSize[dtype]
	if !ok || len(data) != t.Elems()*size {
		return nil, fmt.Errorf("safetensors: tensor of type %s, shape %v, %d bytes", dtype, shape, len(data))
	}
	return t, nil
}

// ReadSafetensors reads a whole safetensors file into memory.
func ReadSafetensors(path string) (map[string]*Tensor, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ParseSafetensors(b)
}

// ParseSafetensors splits a safetensors file into its tensors, which keep
// referring to b.
func ParseSafetensors(b []byte) (map[string]*Tensor, error) {
	if len(b) < 8 {
		return nil, fmt.Errorf("safetensors: %d bytes", len(b))
	}
	n := binary.LittleEndian.Uint64(b)
	if n > 100<<20 || 8+n > uint64(len(b)) {
		return nil, fmt.Errorf("safetensors: header of %d bytes", n)
	}
	var h map[string]json.RawMessage
	if err := json.Unmarshal(b[8:8+n], &h); err != nil {
		return nil, fmt.Errorf("safetensors header: %w", err)
	}
	data := b[8+n:]
	out := make(map[string]*Tensor, len(h))
	for name, raw := range h {
		if name == "__metadata__" {
			continue
		}
		var e struct {
			DType   string   `json:"dtype"`
			Shape   []int    `json:"shape"`
			Offsets [2]int64 `json:"data_offsets"`
		}
		if err := json.Unmarshal(raw, &e); err != nil {
			return nil, fmt.Errorf("safetensors tensor %s: %w", name, err)
		}
		t := &Tensor{DType: e.DType, Shape: e.Shape}
		size, ok := dtypeSize[e.DType]
		lo, hi := e.Offsets[0], e.Offsets[1]
		switch {
		case !ok:
			return nil, fmt.Errorf("safetensors tensor %s: type %s", name, e.DType)
		case lo < 0 || hi < lo || hi > int64(len(data)) || hi-lo != int64(t.Elems()*size):
			return nil, fmt.Errorf("safetensors tensor %s: bad data range", name)
		}
		t.Data = data[lo:hi]
		out[name] = t
	}
	return out, nil
}
