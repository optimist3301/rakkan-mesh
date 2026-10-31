// Package sandbox executes untrusted WebAssembly tasks under resource limits.
//
// Isolation layers implemented here:
//   - the module may only import the rakkan host ABI (see abi.go);
//   - execution is bounded by deterministic gas (see package meter), with a
//     wall clock timeout and caller cancellation as a backstop, both delivered
//     by draining the gas counter (see watch);
//   - linear memory is capped per task;
//   - input, output and log sizes are bounded.
//
// OS-level isolation (separate worker process, seccomp, cgroups) is layered on
// top of this package.
package sandbox

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/meter"
	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
)

// Task is a single invocation of a WebAssembly module.
type Task struct {
	Module []byte
	Entry  string // defaults to DefaultEntry
	Input  []byte
	Limits Limits
}

// Result is the outcome of a task that ran to completion.
type Result struct {
	ExitCode     uint32
	Output       []byte
	Logs         []byte
	LogTruncated bool
	GasUsed      int64
	Elapsed      time.Duration
}

// Runner executes tasks. It is safe for concurrent use.
type Runner struct {
	ceiling Limits
	cache   wazero.CompilationCache
}

// NewRunner returns a runner that refuses tasks requesting more than ceiling.
func NewRunner(ceiling Limits) *Runner {
	return &Runner{ceiling: ceiling, cache: wazero.NewCompilationCache()}
}

// Close releases compiled code held in the cache.
func (r *Runner) Close(ctx context.Context) error {
	return r.cache.Close(ctx)
}

// Run executes t. A nil error means the entry point returned; its exit code is
// in the result. Errors wrap one of the sentinel errors in this package.
func (r *Runner) Run(ctx context.Context, t Task) (*Result, error) {
	lim := t.Limits
	if err := lim.Within(r.ceiling); err != nil {
		return nil, err
	}
	if len(t.Module) > lim.MaxModuleBytes {
		return nil, fmt.Errorf("%w: %d bytes, limit %d", ErrModuleTooLarge, len(t.Module), lim.MaxModuleBytes)
	}
	if len(t.Input) > lim.MaxInputBytes {
		return nil, fmt.Errorf("%w: %d bytes, limit %d", ErrInputTooLarge, len(t.Input), lim.MaxInputBytes)
	}
	entry := t.Entry
	if entry == "" {
		entry = DefaultEntry
	}

	start := time.Now()
	ctx, cancel := context.WithTimeout(ctx, lim.Timeout)
	defer cancel()

	// A runtime per task gives each task its own memory ceiling; the shared
	// compilation cache keeps repeated modules cheap.
	//
	// WithCloseOnContextDone is deliberately not used: it slows loops ~30x
	// in wazero 1.12, and the gas checks already provide interruption points.
	cfg := wazero.NewRuntimeConfig().
		WithCoreFeatures(api.CoreFeaturesV2).
		WithMemoryLimitPages(lim.MaxMemoryPages).
		WithCompilationCache(r.cache)
	rt := wazero.NewRuntimeWithConfig(ctx, cfg)
	defer rt.Close(context.Background())

	st := &execState{input: t.Input, lim: lim}
	if err := st.instantiateHost(ctx, rt); err != nil {
		return nil, fmt.Errorf("instantiate host module: %w", err)
	}

	metered, err := meter.Instrument(t.Module)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidModule, err)
	}
	cm, err := rt.CompileModule(ctx, metered)
	if err != nil {
		return nil, st.classify(ctx, fmt.Errorf("%w: %v", ErrInvalidModule, err))
	}
	if err := validate(cm, entry); err != nil {
		return nil, err
	}

	// No FS, stdio, args, env or clocks are configured: the guest only sees
	// the host ABI.
	mod, err := rt.InstantiateModule(ctx, cm, wazero.NewModuleConfig().WithName("task").WithStartFunctions())
	if err != nil {
		return nil, st.classify(ctx, err)
	}

	st.gas = mod.ExportedGlobal(meter.GasExport).(api.MutableGlobal)
	st.gas.Set(api.EncodeI64(lim.MaxGas))

	stop := st.watch(ctx)
	var ret []uint64
	if start := mod.ExportedFunction(meter.StartExport); start != nil {
		_, err = start.Call(ctx)
	}
	if err == nil {
		ret, err = mod.ExportedFunction(entry).Call(ctx)
	}
	stop()
	if err != nil {
		return nil, st.classify(ctx, err)
	}
	if ctx.Err() != nil {
		// Finished while being interrupted; the balance is unreliable.
		return nil, st.classify(ctx, nil)
	}
	// Checks run only at function entry, loop headers and dynamic charges,
	// so straight-line code may finish slightly in debt.
	remaining := int64(st.gas.Get())
	if remaining < 0 {
		return nil, st.outOfGas()
	}
	return &Result{
		ExitCode:     api.DecodeU32(ret[0]),
		Output:       st.output,
		Logs:         st.logs,
		LogTruncated: st.logTruncated,
		GasUsed:      lim.MaxGas - remaining,
		Elapsed:      time.Since(start),
	}, nil
}

// execState is the per-task state behind the host ABI.
type execState struct {
	input        []byte
	lim          Limits
	output       []byte
	logs         []byte
	logTruncated bool
	gas          api.MutableGlobal // remaining gas, set once instantiated
	gasMu        sync.Mutex        // orders host charges with interruption
	fault        error             // set by host functions before they trap
}

// interrupted is stored in the gas counter to stop the guest. It is far
// below zero so that no charges in flight can bring it back up.
const interrupted = math.MinInt64 / 2

// watch interrupts the guest when ctx ends by draining its gas, so the next
// check (at the latest the next loop iteration or call) traps. Guest code
// updates the counter with a non-atomic read-modify-write that may overwrite
// the store, so it is repeated until the guest stops. The returned function
// stops watching and waits until no further stores can happen.
func (s *execState) watch(ctx context.Context) (stop func()) {
	done := make(chan struct{})
	var wg sync.WaitGroup
	wg.Go(func() {
		select {
		case <-done:
			return
		case <-ctx.Done():
		}
		tick := time.NewTicker(time.Millisecond)
		defer tick.Stop()
		for {
			s.gasMu.Lock()
			s.gas.Set(api.EncodeI64(interrupted))
			s.gasMu.Unlock()
			select {
			case <-done:
				return
			case <-tick.C:
			}
		}
	})
	return func() {
		close(done)
		wg.Wait()
	}
}

// trap aborts guest execution with err as the reported cause.
func (s *execState) trap(err error) {
	s.fault = err
	panic(err)
}

func (s *execState) outOfGas() error {
	return fmt.Errorf("%w: budget %d", ErrOutOfGas, s.lim.MaxGas)
}

// charge bills a host call transferring n bytes, trapping when the balance
// goes negative.
func (s *execState) charge(n uint32) {
	s.gasMu.Lock()
	rem := int64(s.gas.Get()) - meter.HostCallCost - int64(n>>meter.BytesPerGasShift)
	s.gas.Set(api.EncodeI64(rem))
	s.gasMu.Unlock()
	if rem < 0 {
		s.trap(s.outOfGas())
	}
}

// classify maps a runtime error to a sandbox error.
func (s *execState) classify(ctx context.Context, err error) error {
	switch {
	case errors.Is(ctx.Err(), context.DeadlineExceeded):
		return fmt.Errorf("%w after %s", ErrTimeout, s.lim.Timeout)
	case ctx.Err() != nil:
		return ctx.Err()
	case s.fault != nil:
		return s.fault
	case s.gas != nil && int64(s.gas.Get()) < 0:
		return s.outOfGas()
	case errors.Is(err, ErrInvalidModule):
		return err
	}
	return fmt.Errorf("%w: %v", ErrTrap, err)
}

func (s *execState) instantiateHost(ctx context.Context, rt wazero.Runtime) error {
	_, err := rt.NewHostModuleBuilder(HostModule).
		NewFunctionBuilder().WithFunc(s.inputLen).Export("input_len").
		NewFunctionBuilder().WithFunc(s.inputRead).Export("input_read").
		NewFunctionBuilder().WithFunc(s.outputWrite).Export("output_write").
		NewFunctionBuilder().WithFunc(s.log).Export("log").
		Instantiate(ctx)
	return err
}

func (s *execState) inputLen() uint32 {
	s.charge(0)
	return uint32(len(s.input))
}

func (s *execState) inputRead(_ context.Context, m api.Module, dst, offset, n uint32) uint32 {
	s.charge(n)
	if uint64(offset) >= uint64(len(s.input)) {
		return 0
	}
	chunk := s.input[offset:]
	if uint64(n) < uint64(len(chunk)) {
		chunk = chunk[:n]
	}
	if !m.Memory().Write(dst, chunk) {
		s.trap(fmt.Errorf("%w: input_read dst=%d len=%d", ErrMemoryAccess, dst, len(chunk)))
	}
	return uint32(len(chunk))
}

func (s *execState) outputWrite(_ context.Context, m api.Module, src, n uint32) {
	s.charge(n)
	if uint64(len(s.output))+uint64(n) > uint64(s.lim.MaxOutputBytes) {
		s.trap(fmt.Errorf("%w: limit %d bytes", ErrOutputLimit, s.lim.MaxOutputBytes))
	}
	buf, ok := m.Memory().Read(src, n)
	if !ok {
		s.trap(fmt.Errorf("%w: output_write src=%d len=%d", ErrMemoryAccess, src, n))
	}
	s.output = append(s.output, buf...)
}

func (s *execState) log(_ context.Context, m api.Module, src, n uint32) {
	s.charge(n)
	buf, ok := m.Memory().Read(src, n)
	if !ok {
		s.trap(fmt.Errorf("%w: log src=%d len=%d", ErrMemoryAccess, src, n))
	}
	if room := s.lim.MaxLogBytes - len(s.logs); len(buf) > room {
		buf = buf[:max(room, 0)]
		s.logTruncated = true
	}
	s.logs = append(s.logs, buf...)
}
