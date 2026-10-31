// Package wasmtest assembles small WebAssembly binaries by hand so tests do not
// depend on external toolchains such as wat2wasm.
package wasmtest

// Value types.
const (
	I32 byte = 0x7f
	I64 byte = 0x7e
)

// Opcodes used by tests.
const (
	OpUnreachable byte = 0x00
	OpLoop        byte = 0x03
	OpEnd         byte = 0x0b
	OpBr          byte = 0x0c
	OpCall        byte = 0x10
	OpDrop        byte = 0x1a
	OpLocalGet    byte = 0x20
	OpLocalSet    byte = 0x21
	OpLocalTee    byte = 0x22
	OpGlobalGet   byte = 0x23
	OpGlobalSet   byte = 0x24
	OpIf          byte = 0x04
	OpI32Add      byte = 0x6a
	OpI32Sub      byte = 0x6b
	OpI32Eqz      byte = 0x45
	OpBrIf        byte = 0x0d
	OpMemoryGrow  byte = 0x40
	OpI32Const    byte = 0x41
	BlockEmpty    byte = 0x40
	PrefixFC      byte = 0xfc
	MemoryFill    byte = 0x0b // after PrefixFC
)

type Import struct {
	Module, Name    string
	Params, Results []byte
}

type Func struct {
	Params, Results []byte
	Locals          []byte // one entry per local
	Body            []byte // without the trailing end opcode
	Export          string
}

type Data struct {
	Offset uint32
	Bytes  []byte
}

type Global struct {
	Type    byte
	Mutable bool
	Init    []byte // constant expression without the trailing end opcode
}

type Module struct {
	Imports      []Import
	Globals      []Global
	Start        *uint32 // function index
	Funcs        []Func
	Memory       bool
	MemMin       uint32
	MemMax       *uint32
	ExportMemory bool
	Data         []Data
}

// I32Const encodes an i32.const instruction.
func I32Const(v int32) []byte {
	return append([]byte{OpI32Const}, sleb(int64(v))...)
}

// Call encodes a call instruction.
func Call(idx uint32) []byte {
	return append([]byte{OpCall}, uleb(uint64(idx))...)
}

// Cat concatenates instruction fragments.
func Cat(parts ...[]byte) []byte {
	var out []byte
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}

// Encode produces the binary module.
func (m Module) Encode() []byte {
	out := []byte{0x00, 'a', 's', 'm', 0x01, 0x00, 0x00, 0x00}

	// One type per import and function keeps the encoder trivial.
	var types [][]byte
	for _, im := range m.Imports {
		types = append(types, funcType(im.Params, im.Results))
	}
	for _, f := range m.Funcs {
		types = append(types, funcType(f.Params, f.Results))
	}
	if len(types) > 0 {
		out = section(out, 1, vec(types))
	}

	if len(m.Imports) > 0 {
		var ims [][]byte
		for i, im := range m.Imports {
			e := Cat(name(im.Module), name(im.Name), []byte{0x00}, uleb(uint64(i)))
			ims = append(ims, e)
		}
		out = section(out, 2, vec(ims))
	}

	if len(m.Funcs) > 0 {
		var idx [][]byte
		for i := range m.Funcs {
			idx = append(idx, uleb(uint64(len(m.Imports)+i)))
		}
		out = section(out, 3, vec(idx))
	}

	if m.Memory {
		var lim []byte
		if m.MemMax != nil {
			lim = Cat([]byte{0x01}, uleb(uint64(m.MemMin)), uleb(uint64(*m.MemMax)))
		} else {
			lim = Cat([]byte{0x00}, uleb(uint64(m.MemMin)))
		}
		out = section(out, 5, vec([][]byte{lim}))
	}

	if len(m.Globals) > 0 {
		var gs [][]byte
		for _, g := range m.Globals {
			mut := byte(0)
			if g.Mutable {
				mut = 1
			}
			gs = append(gs, Cat([]byte{g.Type, mut}, g.Init, []byte{OpEnd}))
		}
		out = section(out, 6, vec(gs))
	}

	var exports [][]byte
	for i, f := range m.Funcs {
		if f.Export != "" {
			exports = append(exports, Cat(name(f.Export), []byte{0x00}, uleb(uint64(len(m.Imports)+i))))
		}
	}
	if m.Memory && m.ExportMemory {
		exports = append(exports, Cat(name("memory"), []byte{0x02, 0x00}))
	}
	if len(exports) > 0 {
		out = section(out, 7, vec(exports))
	}

	if m.Start != nil {
		out = section(out, 8, uleb(uint64(*m.Start)))
	}

	if len(m.Funcs) > 0 {
		var bodies [][]byte
		for _, f := range m.Funcs {
			var locals [][]byte
			for _, t := range f.Locals {
				locals = append(locals, []byte{0x01, t})
			}
			b := Cat(vec(locals), f.Body, []byte{OpEnd})
			bodies = append(bodies, Cat(uleb(uint64(len(b))), b))
		}
		out = section(out, 10, vec(bodies))
	}

	if len(m.Data) > 0 {
		var segs [][]byte
		for _, d := range m.Data {
			segs = append(segs, Cat([]byte{0x00}, I32Const(int32(d.Offset)), []byte{OpEnd}, uleb(uint64(len(d.Bytes))), d.Bytes))
		}
		out = section(out, 11, vec(segs))
	}
	return out
}

func funcType(params, results []byte) []byte {
	return Cat([]byte{0x60}, uleb(uint64(len(params))), params, uleb(uint64(len(results))), results)
}

func section(out []byte, id byte, content []byte) []byte {
	out = append(out, id)
	out = append(out, uleb(uint64(len(content)))...)
	return append(out, content...)
}

func vec(items [][]byte) []byte {
	out := uleb(uint64(len(items)))
	for _, it := range items {
		out = append(out, it...)
	}
	return out
}

func name(s string) []byte {
	return append(uleb(uint64(len(s))), s...)
}

func uleb(v uint64) []byte {
	var out []byte
	for {
		b := byte(v & 0x7f)
		v >>= 7
		if v != 0 {
			b |= 0x80
		}
		out = append(out, b)
		if v == 0 {
			return out
		}
	}
}

func sleb(v int64) []byte {
	var out []byte
	for {
		b := byte(v & 0x7f)
		v >>= 7
		done := (v == 0 && b&0x40 == 0) || (v == -1 && b&0x40 != 0)
		if !done {
			b |= 0x80
		}
		out = append(out, b)
		if done {
			return out
		}
	}
}
