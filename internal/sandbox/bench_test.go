package sandbox

import (
	"context"
	"testing"

	"github.com/tetratelabs/wazero"
)

const benchIterations = 10_000_000

// BenchmarkLoopMetered and BenchmarkLoopUnmetered compare a tight loop with
// and without gas instrumentation, excluding compilation.
func BenchmarkLoopMetered(b *testing.B) {
	r := NewRunner(DefaultLimits())
	defer r.Close(context.Background())
	task := Task{Module: countdown(benchIterations), Limits: DefaultLimits()}
	r.Run(context.Background(), task) // warm the compilation cache
	b.ResetTimer()
	for b.Loop() {
		if _, err := r.Run(context.Background(), task); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLoopUnmetered(b *testing.B) {
	ctx := context.Background()
	rt := wazero.NewRuntime(ctx)
	defer rt.Close(ctx)
	rt.NewHostModuleBuilder(HostModule).
		NewFunctionBuilder().WithFunc(func() uint32 { return 0 }).Export("input_len").
		NewFunctionBuilder().WithFunc(func(uint32, uint32, uint32) uint32 { return 0 }).Export("input_read").
		NewFunctionBuilder().WithFunc(func(uint32, uint32) {}).Export("output_write").
		NewFunctionBuilder().WithFunc(func(uint32, uint32) {}).Export("log").
		Instantiate(ctx)
	mod, err := rt.Instantiate(ctx, countdown(benchIterations))
	if err != nil {
		b.Fatal(err)
	}
	run := mod.ExportedFunction(DefaultEntry)
	for b.Loop() {
		if _, err := run.Call(ctx); err != nil {
			b.Fatal(err)
		}
	}
}
