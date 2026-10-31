package sandbox

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/meter"
	w "github.com/optimist3301/rakkan-mesh/internal/wasmtest"
)

// Expected costs below follow cost schedule version 1. Each instruction costs
// 1 unless stated, including the end of the function body.
func init() {
	if meter.ScheduleVersion != 1 {
		panic("gas tests assume cost schedule version 1")
	}
}

func gasUsed(t *testing.T, module []byte, lim Limits) int64 {
	t.Helper()
	res, err := run(t, module, nil, lim)
	if err != nil {
		t.Fatal(err)
	}
	return res.GasUsed
}

func TestGasStraightLine(t *testing.T) {
	// i32.const, end
	if got := gasUsed(t, taskModule(1, nil, w.I32Const(42)).Encode(), testLimits()); got != 2 {
		t.Fatalf("gas %d, want 2", got)
	}
}

// countdown loops n times: entry segment (const, set, loop) = 3, each
// iteration (get, const, sub, tee, br_if) = 5, loop end = 1, tail (const, end) = 2.
func countdown(n int32) []byte {
	return taskModule(1, []byte{w.I32}, w.Cat(
		w.I32Const(n), []byte{w.OpLocalSet, 0},
		[]byte{w.OpLoop, w.BlockEmpty},
		[]byte{w.OpLocalGet, 0}, w.I32Const(1), []byte{w.OpI32Sub, w.OpLocalTee, 0, w.OpBrIf, 0},
		[]byte{w.OpEnd},
		w.I32Const(0),
	)).Encode()
}

func TestGasExactForLoops(t *testing.T) {
	for _, n := range []int32{1, 1000, 100_000} {
		if got, want := gasUsed(t, countdown(n), testLimits()), 6+5*int64(n); got != want {
			t.Fatalf("n=%d: gas %d, want %d", n, got, want)
		}
	}
}

func TestInfiniteLoopRunsOutOfGas(t *testing.T) {
	lim := testLimits()
	lim.MaxGas = 1_000_000
	body := w.Cat([]byte{w.OpLoop, w.BlockEmpty, w.OpBr, 0, w.OpEnd}, w.I32Const(0))
	if _, err := run(t, taskModule(1, nil, body).Encode(), nil, lim); !errors.Is(err, ErrOutOfGas) {
		t.Fatalf("got %v, want ErrOutOfGas", err)
	}
}

func TestOutOfGasAtEntry(t *testing.T) {
	lim := testLimits()
	lim.MaxGas = 1
	if _, err := run(t, taskModule(1, nil, w.I32Const(42)).Encode(), nil, lim); !errors.Is(err, ErrOutOfGas) {
		t.Fatalf("got %v, want ErrOutOfGas", err)
	}
}

func TestGasBulkMemoryByLength(t *testing.T) {
	fill := func(n int32) []byte {
		return taskModule(1, nil, w.Cat(
			w.I32Const(0), w.I32Const(0), w.I32Const(n), []byte{w.PrefixFC, w.MemoryFill, 0x00},
			w.I32Const(0),
		)).Encode()
	}
	// 6 static + n/16
	if got := gasUsed(t, fill(PageSize), testLimits()); got != 6+PageSize/16 {
		t.Fatalf("gas %d, want %d", got, 6+PageSize/16)
	}
	lim := testLimits()
	lim.MaxGas = 1000
	if _, err := run(t, fill(PageSize), nil, lim); !errors.Is(err, ErrOutOfGas) {
		t.Fatalf("got %v, want ErrOutOfGas before filling", err)
	}
}

func TestGasMemoryGrowByPages(t *testing.T) {
	// const, memory.grow, end + 1 page
	m := taskModule(1, nil, w.Cat(w.I32Const(1), []byte{w.OpMemoryGrow, 0x00})).Encode()
	if got, want := gasUsed(t, m, testLimits()), int64(3+meter.MemoryPageCost); got != want {
		t.Fatalf("gas %d, want %d", got, want)
	}
}

func TestGasHostCall(t *testing.T) {
	// const, const, call (4), const, end = 8; host: 10 + 160/16
	body := w.Cat(w.I32Const(0), w.I32Const(160), w.Call(fnOutputWrite), w.I32Const(0))
	if got, want := gasUsed(t, taskModule(1, nil, body).Encode(), testLimits()), int64(8+meter.HostCallCost+10); got != want {
		t.Fatalf("gas %d, want %d", got, want)
	}
}

func TestStartFunctionRunsMetered(t *testing.T) {
	m := taskModule(1, nil, w.I32Const(0))
	// Start function writes "abc": const, const, call (4), end = 7; host: 10.
	m.Funcs = append(m.Funcs, w.Func{Body: w.Cat(w.I32Const(0), w.I32Const(3), w.Call(fnOutputWrite))})
	start := uint32(5)
	m.Start = &start
	m.Data = []w.Data{{Offset: 0, Bytes: []byte("abc")}}

	res, err := run(t, m.Encode(), nil, testLimits())
	if err != nil {
		t.Fatal(err)
	}
	if string(res.Output) != "abc" {
		t.Fatalf("output %q, want start function output", res.Output)
	}
	if want := int64(2 + 7 + meter.HostCallCost); res.GasUsed != want {
		t.Fatalf("gas %d, want %d", res.GasUsed, want)
	}
}

func TestExistingGlobalsKeepTheirIndices(t *testing.T) {
	m := taskModule(1, nil, []byte{w.OpGlobalGet, 0})
	m.Globals = []w.Global{{Type: w.I32, Mutable: true, Init: w.I32Const(7)}}
	res, err := run(t, m.Encode(), nil, testLimits())
	if err != nil {
		t.Fatal(err)
	}
	if res.ExitCode != 7 {
		t.Fatalf("exit code %d, want 7", res.ExitCode)
	}
}

func TestReservedEntryRejected(t *testing.T) {
	r := NewRunner(DefaultLimits())
	defer r.Close(t.Context())
	_, err := r.Run(t.Context(), Task{Module: echoModule(), Entry: meter.StartExport, Limits: testLimits()})
	if !errors.Is(err, ErrMissingEntry) {
		t.Fatalf("got %v, want ErrMissingEntry", err)
	}
}

func TestCallerCancellation(t *testing.T) {
	r := NewRunner(DefaultLimits())
	defer r.Close(t.Context())
	ctx, cancel := context.WithCancel(t.Context())
	time.AfterFunc(50*time.Millisecond, cancel)

	// A loop that also calls the host, so interruption races with charges.
	body := w.Cat(
		[]byte{w.OpLoop, w.BlockEmpty},
		w.Call(fnInputLen), []byte{w.OpDrop, w.OpBr, 0, w.OpEnd},
		w.I32Const(0),
	)
	_, err := r.Run(ctx, Task{Module: taskModule(1, nil, body).Encode(), Limits: DefaultLimits()})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
}
