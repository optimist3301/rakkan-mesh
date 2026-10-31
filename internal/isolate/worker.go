package isolate

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/sandbox"
)

// workerEnv marks a process as a sandbox worker and carries its isolation
// settings. It is the only environment variable a worker receives.
const workerEnv = "RAKKAN_SANDBOX_WORKER"

// maxRequestFrame bounds requests read by a worker. The node is trusted and
// validates limits before sending, so this only guards against corruption.
const maxRequestFrame = 1 << 30

// Isolation selects the OS-level protections applied to worker processes.
type Isolation struct {
	// Namespaces runs workers in fresh user, mount, PID, network, IPC and
	// UTS namespaces, with an empty read-only root filesystem and no
	// capabilities.
	Namespaces bool
	// Seccomp restricts workers to the system calls the Go runtime and the
	// WebAssembly engine need; anything else kills the worker.
	Seccomp bool
	// Cgroup is a cgroup v2 directory delegated to the node, or "self" for
	// the node's own cgroup. Each worker gets a child cgroup enforcing
	// MemoryBytes, a process limit and CPUs. Empty disables cgroups.
	Cgroup string
	// MemoryBytes caps a worker's memory: through the cgroup when enabled,
	// otherwise as an address space limit above the worker's startup size.
	// Zero derives it from the task memory ceiling.
	MemoryBytes int64
	// CPUs limits a worker's CPU bandwidth when cgroups are enabled; zero
	// means unlimited.
	CPUs float64
}

// StrictIsolation enables every protection that needs no system setup.
func StrictIsolation() Isolation {
	return Isolation{Namespaces: true, Seccomp: true}
}

func (iso Isolation) encode(stage int) string {
	var parts []string
	parts = append(parts, "v1", "stage="+strconv.Itoa(stage))
	if iso.Namespaces {
		parts = append(parts, "ns")
	}
	if iso.Seccomp {
		parts = append(parts, "seccomp")
	}
	parts = append(parts, "mem="+strconv.FormatInt(iso.MemoryBytes, 10))
	if iso.Cgroup != "" {
		parts = append(parts, "cgroup")
	}
	return strings.Join(parts, ",")
}

func decodeIsolation(s string) (iso Isolation, stage int, err error) {
	parts := strings.Split(s, ",")
	if len(parts) == 0 || parts[0] != "v1" {
		return iso, 0, fmt.Errorf("unsupported worker settings %q", s)
	}
	for _, p := range parts[1:] {
		k, v, _ := strings.Cut(p, "=")
		switch k {
		case "stage":
			stage, err = strconv.Atoi(v)
		case "ns":
			iso.Namespaces = true
		case "seccomp":
			iso.Seccomp = true
		case "cgroup":
			iso.Cgroup = "inherited"
		case "mem":
			iso.MemoryBytes, err = strconv.ParseInt(v, 10, 64)
		default:
			err = fmt.Errorf("unknown worker setting %q", k)
		}
		if err != nil {
			return iso, 0, err
		}
	}
	return iso, stage, nil
}

// IsWorker reports whether this process was started as a sandbox worker.
// Binaries embedding a Pool must call RunWorker early in main when it does.
func IsWorker() bool {
	return os.Getenv(workerEnv) != ""
}

// RunWorker hardens the current process and serves tasks on stdin and
// stdout until stdin is closed. It returns only on failure or shutdown.
func RunWorker() error {
	if err := hardenFromEnv(); err != nil {
		return err
	}
	return serve(os.Stdin, os.Stdout)
}

func hardenFromEnv() error {
	iso, stage, err := decodeIsolation(os.Getenv(workerEnv))
	if err != nil {
		return err
	}
	if err := harden(iso, stage); err != nil {
		return fmt.Errorf("harden worker: %w", err)
	}
	return nil
}

// prepareRuntime tunes the Go runtime for a single-task process.
func prepareRuntime(iso Isolation) {
	// One task runs at a time; fewer Ps means fewer threads.
	runtime.GOMAXPROCS(2)
	if iso.MemoryBytes > 0 {
		// Collect garbage before the hard limit kills the worker.
		debug.SetMemoryLimit(iso.MemoryBytes / 10 * 8)
	}
}

// runTask runs one task with the garbage collector off. Compiled
// WebAssembly cannot be preempted, so a stop-the-world phase started while
// the guest runs waits for it to finish, and every other goroutine,
// including the one that delivers the timeout, stalls behind it. Memory is
// bounded by the worker's cgroup or address space limit instead.
func runTask(runner *sandbox.Runner, task sandbox.Task) (*sandbox.Result, error) {
	gcPercent := debug.SetGCPercent(-1)
	memLimit := debug.SetMemoryLimit(math.MaxInt64)
	defer func() {
		debug.SetGCPercent(gcPercent)
		debug.SetMemoryLimit(memLimit)
	}()
	return runner.Run(context.Background(), task)
}

func serve(r io.Reader, w io.Writer) error {
	br := bufio.NewReader(r)
	bw := bufio.NewWriter(w)
	// The node already checked the task against its policy.
	runner := sandbox.NewRunner(sandbox.Limits{
		MaxModuleBytes: math.MaxInt,
		MaxMemoryPages: sandbox.MaxPages,
		MaxInputBytes:  math.MaxInt,
		MaxOutputBytes: math.MaxInt,
		MaxLogBytes:    math.MaxInt,
		MaxGas:         sandbox.MaxGas,
		Timeout:        time.Duration(math.MaxInt64),
	})
	// Signal readiness: hardening is complete.
	if err := writeFrame(bw, []byte{protoVersion}); err != nil {
		return err
	}
	if err := bw.Flush(); err != nil {
		return err
	}
	for {
		frame, err := readFrame(br, maxRequestFrame)
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		task, err := decodeRequest(frame)
		if err != nil {
			return err
		}
		res, err := runTask(runner, task)
		if err := writeFrame(bw, encodeResponse(res, err)); err != nil {
			return err
		}
		if err := bw.Flush(); err != nil {
			return err
		}
	}
}
