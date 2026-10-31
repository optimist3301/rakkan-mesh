package meter

import (
	"errors"
	"testing"

	"github.com/tetratelabs/wazero"

	w "github.com/optimist3301/rakkan-mesh/internal/wasmtest"
)

func entry(body []byte) w.Module {
	return w.Module{
		Funcs:        []w.Func{{Results: []byte{w.I32}, Locals: []byte{w.I32}, Body: body, Export: "run"}},
		Memory:       true,
		MemMin:       1,
		ExportMemory: true,
	}
}

// compiles checks that wazero accepts the instrumented module.
func compiles(t *testing.T, wasm []byte) {
	t.Helper()
	rt := wazero.NewRuntime(t.Context())
	defer rt.Close(t.Context())
	if _, err := rt.CompileModule(t.Context(), wasm); err != nil {
		t.Fatalf("instrumented module does not compile: %v", err)
	}
}

func TestInstrumentedModulesCompile(t *testing.T) {
	cases := map[string][]byte{
		"empty body": w.I32Const(0),
		"nested blocks and if/else": w.Cat(
			[]byte{0x02, w.BlockEmpty, 0x03, w.BlockEmpty},
			w.I32Const(1), []byte{w.OpIf, w.BlockEmpty, w.OpBr, 2, 0x05, w.OpBr, 1, w.OpEnd},
			[]byte{w.OpEnd, w.OpEnd},
			w.I32Const(0),
		),
		"dead code after br":       w.Cat([]byte{0x02, 0x7f}, w.I32Const(1), []byte{w.OpBr, 0}, w.I32Const(2), []byte{w.OpEnd}),
		"block with result":        w.Cat([]byte{0x02, 0x7f}, w.I32Const(5), []byte{w.OpEnd}),
		"br_table":                 w.Cat([]byte{0x02, w.BlockEmpty}, w.I32Const(0), []byte{0x0e, 1, 0, 0, w.OpEnd}, w.I32Const(0)),
		"bulk memory":              w.Cat(w.I32Const(0), w.I32Const(0), w.I32Const(8), []byte{w.PrefixFC, w.MemoryFill, 0}, w.I32Const(0)),
		"memory.grow uses scratch": w.Cat(w.I32Const(1), []byte{w.OpMemoryGrow, 0, w.OpLocalSet, 0, w.OpLocalGet, 0}),
		"simd":                     w.Cat([]byte{0xfd, 12}, make([]byte, 16), []byte{0xfd, 27, 0}), // v128.const, i32x4.extract_lane 0
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			out, err := Instrument(entry(body).Encode())
			if err != nil {
				t.Fatal(err)
			}
			compiles(t, out)
		})
	}
}

func TestNoExistingSections(t *testing.T) {
	// No globals, no exports: both sections are created in order.
	m := w.Module{Funcs: []w.Func{{Body: nil}}}
	out, err := Instrument(m.Encode())
	if err != nil {
		t.Fatal(err)
	}
	compiles(t, out)
}

func TestRejects(t *testing.T) {
	aliasExport := entry(w.I32Const(0)).Encode()
	// Hand-append an export of global 0 to a module without globals.
	cases := map[string][]byte{
		"code referencing the injected global":     entry([]byte{w.OpGlobalGet, 0}).Encode(),
		"code referencing a local beyond declared": entry([]byte{w.OpLocalGet, 1}).Encode(),
		"threads prefix":                           entry(w.Cat([]byte{0xfe, 0x03, 0x00}, w.I32Const(0))).Encode(),
		"tail call":                                entry(w.Cat([]byte{0x12, 0x00})).Encode(),
		"relaxed simd":                             entry(w.Cat([]byte{0xfd, 0x80, 0x02}, w.I32Const(0))).Encode(),
		"truncated":                                aliasExport[:len(aliasExport)-3],
		"not wasm":                                 []byte("hello"),
	}
	// A table section claiming ~2^32 entries in a few bytes; wazero 1.12
	// preallocates from the declared count and exhausts host memory.
	cases["table count bomb"] = []byte("\x00asm\x01\x00\x00\x00\x04\x05\xff\xff\xff\xff\x0f")
	// Same for a data segment declaring a 4 GiB payload.
	cases["data length bomb"] = []byte("\x00asm\x01\x00\x00\x00\x0b\x07\x01\x01\xff\xff\xff\xff\x0f")

	reserved := entry(w.I32Const(0))
	reserved.Funcs[0].Export = GasExport
	cases["reserved export name"] = reserved.Encode()

	for name, wasm := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := Instrument(wasm); !errors.Is(err, ErrUnsupported) {
				t.Fatalf("got %v, want ErrUnsupported", err)
			}
		})
	}
}

func TestStructuralLimits(t *testing.T) {
	m := entry(w.I32Const(0))
	m.Funcs[0].Locals = nil
	wasm := m.Encode()
	// Patch the single local group count to an absurd number of locals by
	// building the body by hand: 1 group of 2^20 i32s.
	body := w.Cat([]byte{0x01, 0x80, 0x80, 0x40, w.I32}, w.I32Const(0), []byte{w.OpEnd})
	code := w.Cat([]byte{0x01, byte(len(body))}, body)
	wasm = replaceSection(t, wasm, 10, code)
	if _, err := Instrument(wasm); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("got %v, want ErrUnsupported", err)
	}
}

func withoutCustomSections(t *testing.T, wasm []byte) []byte {
	t.Helper()
	m, err := parse(wasm)
	if err != nil {
		t.Fatal(err)
	}
	return encode(m.sections)
}

func replaceSection(t *testing.T, wasm []byte, id byte, content []byte) []byte {
	t.Helper()
	m, err := parse(wasm)
	if err != nil {
		t.Fatal(err)
	}
	for i := range m.sections {
		if m.sections[i].id == id {
			m.sections[i].data = content
		}
	}
	return encode(m.sections)
}

func FuzzInstrument(f *testing.F) {
	f.Add(entry(w.I32Const(0)).Encode())
	f.Add(entry(w.Cat([]byte{w.OpLoop, w.BlockEmpty, w.OpBr, 0, w.OpEnd}, w.I32Const(0))).Encode())
	f.Add(entry(w.Cat(w.I32Const(0), w.I32Const(0), w.I32Const(8), []byte{w.PrefixFC, w.MemoryFill, 0}, w.I32Const(0))).Encode())
	f.Fuzz(func(t *testing.T, wasm []byte) {
		out, err := Instrument(wasm)
		if err != nil {
			return
		}
		// Whatever wazero accepts after instrumentation, it must also accept
		// before: metering never makes an invalid module valid. Custom
		// sections are dropped by design, so compare against the original
		// without them.
		rt := wazero.NewRuntime(t.Context())
		defer rt.Close(t.Context())
		if _, err := rt.CompileModule(t.Context(), out); err == nil {
			if _, err := rt.CompileModule(t.Context(), withoutCustomSections(t, wasm)); err != nil {
				t.Fatalf("instrumentation made an invalid module valid: %v", err)
			}
		}
	})
}
