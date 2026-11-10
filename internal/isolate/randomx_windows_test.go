package isolate

import (
	"fmt"
	"os"
	"testing"

	"github.com/optimist3301/rakkan-mesh/internal/sandbox"
)

// The test binary doubles as the worker, as it does on Linux.
func TestMain(m *testing.M) {
	if !IsWorker() {
		os.Exit(m.Run())
	}
	if err := RunWorker(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	os.Exit(0)
}

func newRandomXPool(t *testing.T, cfg Config) *Pool {
	cfg.Workers = 1
	cfg.Ceiling = sandbox.DefaultLimits()
	p, err := NewPool(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { p.Close() })
	return p
}
