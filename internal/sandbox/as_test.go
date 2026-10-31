package sandbox

import (
	"bytes"
	"encoding/binary"
	"errors"
	"image/jpeg"
	"os"
	"path/filepath"
	"strings"
	"testing"

	w "github.com/optimist3301/rakkan-mesh/internal/wasmtest"
)

// Modules compiled by the AssemblyScript compiler (sdk/as, `npm run build`)
// exercise what hand-assembled ones do not: a start function, bulk memory
// operations, a runtime allocator, and the bindings' abort.
func asModule(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "..", "sdk", "as", "build", name+".wasm"))
	if err != nil {
		t.Skipf("AssemblyScript build not present: %v", err)
	}
	return b
}

func TestAssemblyScriptSearchMatchesHandwritten(t *testing.T) {
	mod := asModule(t, "search")
	for _, c := range []struct{ start, count, secret uint32 }{{0, 100_000, 77_777}, {500_000, 1000, 1}} {
		in := make([]byte, 12)
		binary.LittleEndian.PutUint32(in[0:], c.start)
		binary.LittleEndian.PutUint32(in[4:], c.count)
		binary.LittleEndian.PutUint32(in[8:], w.Mix32(c.secret))
		got, err := run(t, mod, in, DefaultLimits())
		if err != nil {
			t.Fatal(err)
		}
		want, _ := run(t, w.Search(), in, DefaultLimits())
		if got.ExitCode != want.ExitCode || !bytes.Equal(got.Output, want.Output) {
			t.Fatalf("AssemblyScript: exit %d %x; handwritten: exit %d %x", got.ExitCode, got.Output, want.ExitCode, want.Output)
		}
	}
}

func TestAssemblyScriptSepia(t *testing.T) {
	mod := asModule(t, "sepia")
	src := gradientPNG(t, 1200, 900)
	a, err := run(t, mod, src, imageLimits())
	if err != nil {
		t.Fatal(err)
	}
	if a.ExitCode != 0 {
		t.Fatalf("exit %d, logs %q", a.ExitCode, a.Logs)
	}
	img, err := jpeg.Decode(bytes.NewReader(a.Output))
	if err != nil {
		t.Fatal(err)
	}
	if b := img.Bounds(); b.Dx() != 800 || b.Dy() != 600 {
		t.Fatalf("output %dx%d", b.Dx(), b.Dy())
	}
	if string(a.Logs) != "sepia 800x600\n" {
		t.Fatalf("logs %q", a.Logs)
	}
	b, _ := run(t, mod, src, imageLimits())
	if !bytes.Equal(a.Output, b.Output) || a.GasUsed != b.GasUsed {
		t.Fatal("not deterministic")
	}
	// Bad input fails cleanly with the task's own exit code.
	if res, err := run(t, mod, []byte("not an image"), imageLimits()); err != nil || res.ExitCode != 1 {
		t.Fatalf("bad input: exit %v, err %v", res, err)
	}
}

func TestAssemblyScriptWordCount(t *testing.T) {
	mod := asModule(t, "wordcount")
	// Matches straddle piece boundaries (64 KiB) and range boundaries.
	text := bytes.Repeat([]byte("the mesh runs tasks; meshmesh is not mesh-free. "), 9000)
	r := NewRunner(DefaultLimits())
	defer r.Close(t.Context())
	for _, word := range []string{"mesh", "meshmesh", "s; m", "absent"} {
		want := uint32(countOverlapping(text, []byte(word)))
		for _, parts := range []int{1, 3, 7} {
			var total uint32
			step := (len(text) + parts - 1) / parts
			for off := 0; off < len(text); off += step {
				in := binary.LittleEndian.AppendUint64(nil, uint64(off))
				in = binary.LittleEndian.AppendUint64(in, uint64(step))
				in = append(in, word...)
				res, err := r.Run(t.Context(), Task{Module: mod, Input: in, Blobs: []BlobSource{memBlob{data: text}}, Limits: DefaultLimits()})
				if err != nil {
					t.Fatal(err)
				}
				if res.ExitCode != 0 {
					t.Fatalf("exit %d, logs %q", res.ExitCode, res.Logs)
				}
				total += binary.LittleEndian.Uint32(res.Output)
			}
			if total != want {
				t.Fatalf("%q in %d parts: %d, want %d", word, parts, total, want)
			}
		}
	}
}

func countOverlapping(s, sub []byte) int {
	n := 0
	for i := 0; i+len(sub) <= len(s); i++ {
		if bytes.Equal(s[i:i+len(sub)], sub) {
			n++
		}
	}
	return n
}

func TestAssemblyScriptAbortIsLoggedAndTraps(t *testing.T) {
	mod := asModule(t, "test-abort")
	r := NewRunner(DefaultLimits())
	defer r.Close(t.Context())
	_, err := r.Run(t.Context(), Task{Module: mod, Limits: DefaultLimits()})
	if !errors.Is(err, ErrTrap) || !strings.Contains(err.Error(), "abort: Index out of range") {
		t.Fatalf("got %v, want ErrTrap carrying the abort message", err)
	}
	t.Log(err)
}
