package sandbox

import (
	"bytes"
	"encoding/binary"
	"errors"
	"image/jpeg"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
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

func f32Bytes(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return b
}

func randF32(rng *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64() * 0.5)
	}
	return v
}

// matmulTRef computes x[rows,k] W[n,k]^T + b with the kernels' rounding.
func matmulTRef(x []float32, rows, k int, w []float32, n int, b []float32) []float32 {
	out := make([]float32, rows*n)
	for i := range rows {
		for j := range n {
			var acc float32
			for p := range k {
				acc += float32(x[i*k+p] * w[j*k+p])
			}
			out[i*n+j] = acc + b[j]
		}
	}
	return out
}

func TestAssemblyScriptMLPMatchesReferenceBitForBit(t *testing.T) {
	mod := asModule(t, "mlp")
	const nIn, nHidden, nOut, rows = 64, 32, 10, 16
	rng := rand.New(rand.NewPCG(7, 7))
	w1, b1 := randF32(rng, nHidden*nIn), randF32(rng, nHidden)
	w2, b2 := randF32(rng, nOut*nHidden), randF32(rng, nOut)
	x := randF32(rng, rows*nIn)
	weights := slices.Concat(f32Bytes(w1), f32Bytes(b1), f32Bytes(w2), f32Bytes(b2))

	in := binary.LittleEndian.AppendUint32(nil, nIn)
	in = binary.LittleEndian.AppendUint32(in, nHidden)
	in = binary.LittleEndian.AppendUint32(in, nOut)
	in = binary.LittleEndian.AppendUint32(in, rows)
	in = append(in, f32Bytes(x)...)

	r := NewRunner(DefaultLimits())
	defer r.Close(t.Context())
	res, err := r.Run(t.Context(), Task{Module: mod, Input: in, Blobs: []BlobSource{memBlob{data: weights}}, Limits: DefaultLimits()})
	if err != nil {
		t.Fatal(err)
	}

	hidden := matmulTRef(x, rows, nIn, w1, nHidden, b1)
	for i, v := range hidden {
		hidden[i] = max(v, 0)
	}
	logits := matmulTRef(hidden, rows, nHidden, w2, nOut, b2)
	if want := f32Bytes(logits); !bytes.Equal(res.Output[:len(want)], want) {
		t.Fatal("logits differ from the reference")
	}
	for r := range rows {
		best := 0
		for j := range nOut {
			if logits[r*nOut+j] > logits[r*nOut+best] {
				best = j
			}
		}
		if got := binary.LittleEndian.Uint32(res.Output[4*rows*nOut+4*r:]); got != uint32(best) {
			t.Fatalf("row %d: class %d, want %d", r, got, best)
		}
	}
}

func matmulInput(rng *rand.Rand, m, k, n int) []byte {
	in := binary.LittleEndian.AppendUint32(nil, uint32(m))
	in = binary.LittleEndian.AppendUint32(in, uint32(k))
	in = binary.LittleEndian.AppendUint32(in, uint32(n))
	in = append(in, f32Bytes(randF32(rng, m*k))...)
	return append(in, f32Bytes(randF32(rng, k*n))...)
}

// WebAssembly has no fused multiply-add, so a product computed in plain
// AssemblyScript rounds exactly as the kernel does: the results must match.
func TestKernelMatmulMatchesWebAssembly(t *testing.T) {
	wasmMod, kernelMod := asModule(t, "matmul-wasm"), asModule(t, "matmul-kernel")
	in := matmulInput(rand.New(rand.NewPCG(3, 4)), 40, 70, 30)
	a, err := run(t, wasmMod, in, DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	b, err := run(t, kernelMod, in, DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a.Output, b.Output) {
		t.Fatal("kernel and WebAssembly products differ")
	}
}

func BenchmarkMatmul256(b *testing.B) {
	in := matmulInput(rand.New(rand.NewPCG(1, 1)), 256, 256, 256)
	for _, name := range []string{"matmul-wasm", "matmul-kernel"} {
		b.Run(name, func(b *testing.B) {
			mod, err := os.ReadFile(filepath.Join("..", "..", "sdk", "as", "build", name+".wasm"))
			if err != nil {
				b.Skip(err)
			}
			r := NewRunner(DefaultLimits())
			defer r.Close(b.Context())
			task := Task{Module: mod, Input: in, Limits: DefaultLimits()}
			r.Run(b.Context(), task) // compile once
			for b.Loop() {
				res, err := r.Run(b.Context(), task)
				if err != nil {
					b.Fatal(err)
				}
				b.ReportMetric(float64(res.GasUsed), "gas/op")
			}
		})
	}
}
