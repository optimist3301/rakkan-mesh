package sandbox

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"testing"
	"time"

	w "github.com/optimist3301/rakkan-mesh/internal/wasmtest"
)

// Function indices of the host imports declared by taskModule.
const (
	fnInputLen uint32 = iota
	fnInputRead
	fnOutputWrite
	fnLog
)

// taskModule builds a module importing the full host ABI with a single
// exported entry point "run" of type () -> i32.
func taskModule(memPages uint32, locals []byte, body []byte) w.Module {
	return w.Module{
		Imports: []w.Import{
			{Module: HostModule, Name: "input_len", Results: []byte{w.I32}},
			{Module: HostModule, Name: "input_read", Params: []byte{w.I32, w.I32, w.I32}, Results: []byte{w.I32}},
			{Module: HostModule, Name: "output_write", Params: []byte{w.I32, w.I32}},
			{Module: HostModule, Name: "log", Params: []byte{w.I32, w.I32}},
		},
		Funcs:        []w.Func{{Results: []byte{w.I32}, Locals: locals, Body: body, Export: DefaultEntry}},
		Memory:       true,
		MemMin:       memPages,
		ExportMemory: true,
	}
}

// echoModule copies its input to its output.
func echoModule() []byte {
	return taskModule(1, []byte{w.I32}, w.Cat(
		w.Call(fnInputLen), []byte{w.OpLocalSet, 0},
		w.I32Const(0), w.I32Const(0), []byte{w.OpLocalGet, 0}, w.Call(fnInputRead), []byte{w.OpDrop},
		w.I32Const(0), []byte{w.OpLocalGet, 0}, w.Call(fnOutputWrite),
		w.I32Const(0),
	)).Encode()
}

func testLimits() Limits {
	l := DefaultLimits()
	l.MaxMemoryPages = 16
	l.MaxOutputBytes = 1024
	l.MaxLogBytes = 8
	l.Timeout = 2 * time.Second
	return l
}

func run(t *testing.T, module, input []byte, lim Limits) (*Result, error) {
	t.Helper()
	r := NewRunner(DefaultLimits())
	t.Cleanup(func() { r.Close(context.Background()) })
	return r.Run(context.Background(), Task{Module: module, Input: input, Limits: lim})
}

func TestEcho(t *testing.T) {
	in := []byte("hello, mesh")
	res, err := run(t, echoModule(), in, testLimits())
	if err != nil {
		t.Fatal(err)
	}
	if res.ExitCode != 0 || !bytes.Equal(res.Output, in) {
		t.Fatalf("got exit=%d output=%q", res.ExitCode, res.Output)
	}
}

func TestExitCode(t *testing.T) {
	res, err := run(t, taskModule(1, nil, w.I32Const(42)).Encode(), nil, testLimits())
	if err != nil {
		t.Fatal(err)
	}
	if res.ExitCode != 42 {
		t.Fatalf("exit code %d, want 42", res.ExitCode)
	}
}

func TestInfiniteLoopTimesOut(t *testing.T) {
	lim := testLimits()
	lim.Timeout = 100 * time.Millisecond
	body := w.Cat([]byte{w.OpLoop, w.BlockEmpty, w.OpBr, 0, w.OpEnd}, w.I32Const(0))

	for range 5 {
		start := time.Now()
		_, err := run(t, taskModule(1, nil, body).Encode(), nil, lim)
		if !errors.Is(err, ErrTimeout) {
			t.Fatalf("got %v, want ErrTimeout", err)
		}
		if el := time.Since(start); el > 300*time.Millisecond {
			t.Fatalf("interruption took %s", el)
		}
	}
}

func TestMemoryGrowBeyondLimitFails(t *testing.T) {
	// memory.grow returns -1 when the runtime refuses to grow.
	body := w.Cat(w.I32Const(1000), []byte{w.OpMemoryGrow, 0x00})
	res, err := run(t, taskModule(1, nil, body).Encode(), nil, testLimits())
	if err != nil {
		t.Fatal(err)
	}
	if int32(res.ExitCode) != -1 {
		t.Fatalf("memory.grow returned %d, want -1", int32(res.ExitCode))
	}
}

func TestDeclaredMaxAboveLimitIsClamped(t *testing.T) {
	m := taskModule(1, nil, w.Cat(w.I32Const(100), []byte{w.OpMemoryGrow, 0x00}))
	maxPages := uint32(65536)
	m.MemMax = &maxPages
	res, err := run(t, m.Encode(), nil, testLimits())
	if err != nil {
		t.Fatal(err)
	}
	if int32(res.ExitCode) != -1 {
		t.Fatalf("memory.grow returned %d, want -1", int32(res.ExitCode))
	}
}

func TestInitialMemoryAboveLimitRejected(t *testing.T) {
	_, err := run(t, taskModule(64, nil, w.I32Const(0)).Encode(), nil, testLimits())
	if !errors.Is(err, ErrInvalidModule) {
		t.Fatalf("got %v, want ErrInvalidModule", err)
	}
}

func TestOutputLimit(t *testing.T) {
	body := w.Cat(w.I32Const(0), w.I32Const(PageSize), w.Call(fnOutputWrite), w.I32Const(0))
	_, err := run(t, taskModule(1, nil, body).Encode(), nil, testLimits())
	if !errors.Is(err, ErrOutputLimit) {
		t.Fatalf("got %v, want ErrOutputLimit", err)
	}
}

func TestHostCallOutOfBounds(t *testing.T) {
	body := w.Cat(w.I32Const(PageSize-4), w.I32Const(0), w.I32Const(64), w.Call(fnInputRead), []byte{w.OpDrop}, w.I32Const(0))
	_, err := run(t, taskModule(1, nil, body).Encode(), bytes.Repeat([]byte{1}, 64), testLimits())
	if !errors.Is(err, ErrMemoryAccess) {
		t.Fatalf("got %v, want ErrMemoryAccess", err)
	}
}

func TestLogTruncated(t *testing.T) {
	m := taskModule(1, nil, w.Cat(w.I32Const(0), w.I32Const(32), w.Call(fnLog), w.I32Const(0)))
	m.Data = []w.Data{{Offset: 0, Bytes: []byte("0123456789abcdef0123456789abcdef")}}
	res, err := run(t, m.Encode(), nil, testLimits())
	if err != nil {
		t.Fatal(err)
	}
	if string(res.Logs) != "01234567" || !res.LogTruncated {
		t.Fatalf("logs %q truncated=%v", res.Logs, res.LogTruncated)
	}
}

func TestUnreachableTraps(t *testing.T) {
	_, err := run(t, taskModule(1, nil, []byte{w.OpUnreachable}).Encode(), nil, testLimits())
	if !errors.Is(err, ErrTrap) {
		t.Fatalf("got %v, want ErrTrap", err)
	}
}

func TestForbiddenImports(t *testing.T) {
	cases := map[string]w.Import{
		"wasi":            {Module: "wasi_snapshot_preview1", Name: "fd_write", Params: []byte{w.I32, w.I32, w.I32, w.I32}, Results: []byte{w.I32}},
		"unknown host fn": {Module: HostModule, Name: "spawn", Results: []byte{w.I32}},
		"wrong signature": {Module: HostModule, Name: "input_len", Results: []byte{w.I64}},
	}
	for name, im := range cases {
		t.Run(name, func(t *testing.T) {
			m := w.Module{
				Imports:      []w.Import{im},
				Funcs:        []w.Func{{Results: []byte{w.I32}, Body: w.I32Const(0), Export: DefaultEntry}},
				Memory:       true,
				MemMin:       1,
				ExportMemory: true,
			}
			_, err := run(t, m.Encode(), nil, testLimits())
			if !errors.Is(err, ErrForbiddenImport) {
				t.Fatalf("got %v, want ErrForbiddenImport", err)
			}
		})
	}
}

func TestMissingEntry(t *testing.T) {
	m := taskModule(1, nil, w.I32Const(0))
	m.Funcs[0].Export = "main"
	_, err := run(t, m.Encode(), nil, testLimits())
	if !errors.Is(err, ErrMissingEntry) {
		t.Fatalf("got %v, want ErrMissingEntry", err)
	}
}

func TestMemoryNotExported(t *testing.T) {
	m := taskModule(1, nil, w.I32Const(0))
	m.ExportMemory = false
	_, err := run(t, m.Encode(), nil, testLimits())
	if !errors.Is(err, ErrInvalidModule) {
		t.Fatalf("got %v, want ErrInvalidModule", err)
	}
}

func TestSizeLimits(t *testing.T) {
	lim := testLimits()
	lim.MaxModuleBytes = 16
	if _, err := run(t, echoModule(), nil, lim); !errors.Is(err, ErrModuleTooLarge) {
		t.Fatalf("got %v, want ErrModuleTooLarge", err)
	}
	lim = testLimits()
	lim.MaxInputBytes = 4
	if _, err := run(t, echoModule(), []byte("too long"), lim); !errors.Is(err, ErrInputTooLarge) {
		t.Fatalf("got %v, want ErrInputTooLarge", err)
	}
}

func TestGarbageModule(t *testing.T) {
	_, err := run(t, []byte("\x00asm\x01\x00\x00\x00garbage"), nil, testLimits())
	if !errors.Is(err, ErrInvalidModule) {
		t.Fatalf("got %v, want ErrInvalidModule", err)
	}
}

func TestLimitsAbovePolicyRejected(t *testing.T) {
	lim := DefaultLimits()
	lim.Timeout = time.Hour
	if _, err := run(t, echoModule(), nil, lim); !errors.Is(err, ErrLimitPolicy) {
		t.Fatalf("got %v, want ErrLimitPolicy", err)
	}
}

func TestConcurrentRuns(t *testing.T) {
	r := NewRunner(DefaultLimits())
	defer r.Close(context.Background())
	mod := echoModule()

	errs := make(chan error, 32)
	for i := range 32 {
		go func() {
			in := []byte{byte(i)}
			res, err := r.Run(context.Background(), Task{Module: mod, Input: in, Limits: testLimits()})
			if err == nil && !bytes.Equal(res.Output, in) {
				err = errors.New("output mismatch")
			}
			errs <- err
		}()
	}
	for range 32 {
		if err := <-errs; err != nil {
			t.Fatal(err)
		}
	}
}

func TestRepeatedRunsAreCached(t *testing.T) {
	r := NewRunner(DefaultLimits())
	defer r.Close(t.Context())
	mod := w.CompileBomb(500) // slow enough to compile that a cache hit is obvious
	lim := DefaultLimits()

	first, err := r.Run(t.Context(), Task{Module: mod, Limits: lim})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := r.metered.get(sha256.Sum256(mod)); !ok {
		t.Fatal("instrumented module not cached")
	}
	lim.MaxMemoryPages = 32 // different limits share the cache
	second, err := r.Run(t.Context(), Task{Module: mod, Limits: lim})
	if err != nil {
		t.Fatal(err)
	}
	if second.GasUsed != first.GasUsed {
		t.Fatalf("gas %d then %d", first.GasUsed, second.GasUsed)
	}
	if second.Elapsed*5 > first.Elapsed {
		t.Fatalf("second run took %s, first %s: cache miss?", second.Elapsed, first.Elapsed)
	}
}

func TestMeteredCacheEviction(t *testing.T) {
	c := newMeteredCache(10)
	keys := [][32]byte{{1}, {2}, {3}}
	c.add(keys[0], make([]byte, 4))
	c.add(keys[1], make([]byte, 4))
	c.get(keys[0]) // keys[1] is now least recently used
	c.add(keys[2], make([]byte, 4))
	if _, ok := c.get(keys[1]); ok {
		t.Fatal("least recently used entry kept")
	}
	for _, k := range []([32]byte){keys[0], keys[2]} {
		if _, ok := c.get(k); !ok {
			t.Fatalf("entry %v evicted", k[0])
		}
	}
	c.add([32]byte{4}, make([]byte, 11)) // larger than the cache: ignored
	if c.size != 8 {
		t.Fatalf("size %d, want 8", c.size)
	}
}

func TestSearchModule(t *testing.T) {
	const secret = 123_456
	in := make([]byte, 12)
	binary.LittleEndian.PutUint32(in[0:], 100_000)
	binary.LittleEndian.PutUint32(in[4:], 50_000)
	binary.LittleEndian.PutUint32(in[8:], w.Mix32(secret))
	res, err := run(t, w.Search(), in, DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	if res.ExitCode != 0 || binary.LittleEndian.Uint32(res.Output) != secret {
		t.Fatalf("exit %d output %x", res.ExitCode, res.Output)
	}
	binary.LittleEndian.PutUint32(in[0:], 0) // range without the secret
	if res, err = run(t, w.Search(), in, DefaultLimits()); err != nil || res.ExitCode != 1 {
		t.Fatalf("exit %d err %v, want 1", res.ExitCode, err)
	}
}
