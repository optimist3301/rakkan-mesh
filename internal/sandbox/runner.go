// Package sandbox executes untrusted WebAssembly tasks under resource limits.
//
// Isolation layers implemented here:
//   - the module may only import the rakkan host ABI (see abi.go);
//   - execution is bounded by deterministic gas (see package meter), with a
//     wall clock timeout and caller cancellation as a backstop, both delivered
//     through an interrupt flag checked alongside the gas (see watch);
//   - linear memory is capped per task;
//   - input, output and log sizes are bounded.
//
// OS-level isolation (separate worker process, seccomp, cgroups) is layered on
// top of this package.
package sandbox

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
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
	// Blobs are read-only data the task reads in pieces with blob_read,
	// without copying them whole: shared data, or inputs too large for
	// memory. At most MaxBlobs.
	Blobs  []BlobSource
	Limits Limits
}

// MaxBlobs bounds the blobs attached to one task.
const MaxBlobs = 16

// BlobSource is data attached to a task. ReadAt follows io.ReaderAt; an
// error other than io.EOF aborts the task with ErrBlobUnavailable.
type BlobSource interface {
	Size() int64
	ReadAt(p []byte, off int64) (int, error)
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
	metered *meteredCache
}

// meteredCacheBytes bounds the instrumented modules a Runner keeps.
const meteredCacheBytes = 64 << 20

// NewRunner returns a runner that refuses tasks requesting more than ceiling.
func NewRunner(ceiling Limits) *Runner {
	return &Runner{ceiling: ceiling, cache: wazero.NewCompilationCache(), metered: newMeteredCache(meteredCacheBytes)}
}

// instrument returns the metered module, from cache when possible.
func (r *Runner) instrument(module []byte) ([]byte, error) {
	key := sha256.Sum256(module)
	if m, ok := r.metered.get(key); ok {
		return m, nil
	}
	m, err := meter.Instrument(module)
	if err != nil {
		return nil, err
	}
	r.metered.add(key, m)
	return m, nil
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
	if len(t.Blobs) > MaxBlobs {
		return nil, fmt.Errorf("%w: %d blobs attached, limit %d", ErrLimitPolicy, len(t.Blobs), MaxBlobs)
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

	st := &execState{input: t.Input, blobs: t.Blobs, lim: lim}
	if err := st.instantiateHost(ctx, rt); err != nil {
		return nil, fmt.Errorf("instantiate host module: %w", err)
	}

	metered, err := r.instrument(t.Module)
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
	st.interrupt = mod.ExportedGlobal(meter.InterruptExport).(api.MutableGlobal)

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
	blobs        []BlobSource
	images       imageTable
	lim          Limits
	output       []byte
	logs         []byte
	logTruncated bool
	gas          api.MutableGlobal // remaining gas, set once instantiated
	interrupt    api.MutableGlobal // non-zero stops the guest
	fault        error             // set by host functions before they trap
}

// watch interrupts the guest when ctx ends by setting its interrupt flag,
// which the guest checks at every function entry and loop iteration (see
// package meter). The returned function stops watching and waits until the
// watcher can no longer touch the instance.
func (s *execState) watch(ctx context.Context) (stop func()) {
	done := make(chan struct{})
	var wg sync.WaitGroup
	wg.Go(func() {
		select {
		case <-done:
		case <-ctx.Done():
			s.interrupt.Set(1)
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
	s.chargeUnits(meter.HostCallCost + int64(n>>meter.BytesPerGasShift))
}

// chargeUnits bills host work in gas units, trapping when the balance goes
// negative. Host work is charged before it is done.
func (s *execState) chargeUnits(units int64) {
	rem := int64(s.gas.Get()) - units
	s.gas.Set(api.EncodeI64(rem))
	if rem < 0 {
		s.trap(s.outOfGas())
	}
}

func errBlob(index uint32, err error) error {
	return fmt.Errorf("%w: blob %d: %v", ErrBlobUnavailable, index, err)
}

func errMemory(fn string, ptr, n uint32) error {
	return fmt.Errorf("%w: %s ptr=%d len=%d", ErrMemoryAccess, fn, ptr, n)
}

func errOutput(limit int) error {
	return fmt.Errorf("%w: limit %d bytes", ErrOutputLimit, limit)
}

// classify maps a runtime error to a sandbox error, with the end of the
// task's log appended: a failing task's own diagnostics, such as an
// AssemblyScript abort message, are usually the best explanation.
func (s *execState) classify(ctx context.Context, err error) error {
	err = s.classifyErr(ctx, err)
	if len(s.logs) == 0 {
		return err
	}
	tail := s.logs[max(0, len(s.logs)-logTail):]
	return fmt.Errorf("%w; task log: %q", err, bytes.TrimSpace(tail))
}

// logTail bounds the log appended to a task error.
const logTail = 1 << 10

func (s *execState) classifyErr(ctx context.Context, err error) error {
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
		NewFunctionBuilder().WithFunc(s.blobCount).Export("blob_count").
		NewFunctionBuilder().WithFunc(s.blobSize).Export("blob_size").
		NewFunctionBuilder().WithFunc(s.blobRead).Export("blob_read").
		Instantiate(ctx)
	if err != nil {
		return err
	}
	return s.instantiateImage(ctx, rt)
}

func (s *execState) blobCount() uint32 {
	s.charge(0)
	return uint32(len(s.blobs))
}

func (s *execState) blobSize(index uint32) int64 {
	s.charge(0)
	if index >= uint32(len(s.blobs)) {
		return -1
	}
	return s.blobs[index].Size()
}

func (s *execState) blobRead(_ context.Context, m api.Module, index, dst uint32, offset int64, n uint32) int32 {
	s.charge(n)
	if index >= uint32(len(s.blobs)) {
		return -1
	}
	b := s.blobs[index]
	if offset < 0 || offset >= b.Size() {
		return 0
	}
	n = uint32(min(int64(n), b.Size()-offset))
	// Read straight into guest memory.
	buf, ok := m.Memory().Read(dst, n)
	if !ok {
		s.trap(fmt.Errorf("%w: blob_read dst=%d len=%d", ErrMemoryAccess, dst, n))
	}
	// n stops at the blob's end, so a short read is a failure.
	if got, err := b.ReadAt(buf, offset); uint32(got) < n {
		s.trap(errBlob(index, err))
	}
	return int32(n)
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
