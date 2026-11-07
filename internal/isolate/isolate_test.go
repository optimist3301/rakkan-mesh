//go:build linux

package isolate

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
	"unsafe"

	"golang.org/x/sys/unix"

	"github.com/optimist3301/rakkan-mesh/internal/sandbox"
	w "github.com/optimist3301/rakkan-mesh/internal/wasmtest"
)

// The test binary doubles as the worker. Its first argument selects a mode:
// none for a real worker, "probe=<name>" to attempt a forbidden operation
// after hardening, or a misbehaviour (crash, hang, lie) to exercise the
// node's defences.
func TestMain(m *testing.M) {
	if !IsWorker() {
		os.Exit(m.Run())
	}
	mode := ""
	if len(os.Args) > 1 {
		mode = os.Args[1]
	}
	if mode == "" {
		if err := RunWorker(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	}
	if err := hardenFromEnv(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if name, ok := strings.CutPrefix(mode, "probe="); ok {
		fmt.Println(probe(name))
		os.Exit(0)
	}
	misbehave(mode)
}

func probe(name string) string {
	result := func(err error) string {
		if err != nil {
			return "denied: " + err.Error()
		}
		return "allowed"
	}
	switch name {
	case "read-file":
		_, err := os.ReadFile("/etc/passwd")
		return result(err)
	case "list-root":
		var names []string
		err := filepath.WalkDir("/", func(path string, d fs.DirEntry, err error) error {
			if err == nil && !d.IsDir() {
				names = append(names, path)
			}
			return err
		})
		if err != nil {
			return result(err)
		}
		return "files: " + strings.Join(names, " ")
	case "write-file":
		return result(os.WriteFile("/x", []byte("x"), 0o644))
	case "socket":
		fd, err := syscall.Socket(syscall.AF_INET, syscall.SOCK_STREAM, 0)
		if err == nil {
			syscall.Close(fd)
		}
		return result(err)
	case "connect":
		c, err := net.DialTimeout("tcp", "1.1.1.1:53", time.Second)
		if err == nil {
			c.Close()
		}
		return result(err)
	case "fork":
		_, err := syscall.ForkExec(os.Args[0], []string{"x"}, nil)
		return result(err)
	case "mount":
		return result(unix.Mount("tmpfs", "/", "tmpfs", 0, ""))
	case "caps":
		hdr := unix.CapUserHeader{Version: unix.LINUX_CAPABILITY_VERSION_3}
		var data [2]unix.CapUserData
		if err := unix.Capget(&hdr, &data[0]); err != nil {
			return result(err)
		}
		return fmt.Sprintf("caps: %x %x %x", data[0].Effective, data[0].Permitted, data[0].Inheritable)
	case "prctl-vma":
		// What the Go runtime does to name its memory. Kernels without
		// support answer EINVAL; either way the call must not be fatal.
		mem, err := syscall.Mmap(-1, 0, 4096, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
		if err != nil {
			return result(err)
		}
		name := []byte("probe\x00")
		_, _, errno := syscall.Syscall6(syscall.SYS_PRCTL, 0x53564d41, 0,
			uintptr(unsafe.Pointer(&mem[0])), 4096, uintptr(unsafe.Pointer(&name[0])), 0)
		if errno != 0 && errno != syscall.EINVAL {
			return "denied: " + errno.Error()
		}
		return "allowed"
	case "prctl-other":
		_, _, errno := syscall.Syscall(syscall.SYS_PRCTL, 15 /* PR_SET_NAME */, 0, 0)
		return result(errno)
	case "alloc":
		b := make([]byte, 1<<30)
		for i := range b {
			if i%4096 == 0 {
				b[i] = 1
			}
		}
		return result(nil)
	}
	return "unknown probe"
}

func misbehave(mode string) {
	out := bufio.NewWriter(os.Stdout)
	writeFrame(out, []byte{protoVersion})
	out.Flush()
	in := bufio.NewReader(os.Stdin)
	if _, err := readFrame(in, maxRequestFrame); err != nil {
		os.Exit(1)
	}
	switch mode {
	case "badread":
		// Ask for a blob that was never attached.
		writeFrame(out, []byte{msgBlobRead, 0, 0, 0, 99, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16})
		out.Flush()
		time.Sleep(time.Hour)
	case "crash":
		os.Exit(3)
	case "hang":
		time.Sleep(time.Hour)
	case "lie":
		writeFrame(out, encodeResponse(&sandbox.Result{GasUsed: sandbox.MaxGas}, nil))
		out.Flush()
		time.Sleep(time.Hour)
	}
	os.Exit(4)
}

func requireIsolationSupport(t *testing.T) {
	t.Helper()
	cmd := exec.Command("/bin/true")
	cmd.SysProcAttr = StrictIsolation().sysProcAttr()
	if err := cmd.Run(); err != nil {
		t.Skipf("user namespaces unavailable: %v", err)
	}
}

func testLimits() sandbox.Limits {
	l := sandbox.DefaultLimits()
	l.MaxMemoryPages = 16
	l.Timeout = 2 * time.Second
	return l
}

func newPool(t *testing.T, cfg Config) *Pool {
	t.Helper()
	requireIsolationSupport(t)
	if cfg.Ceiling == (sandbox.Limits{}) {
		cfg.Ceiling = sandbox.DefaultLimits()
	}
	if cfg.Workers == 0 {
		cfg.Workers = 1
	}
	if raceEnabled && cfg.Isolation.MemoryBytes == 0 {
		cfg.Isolation.MemoryBytes = 1 << 40
	}
	p, err := NewPool(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { p.Close() })
	return p
}

func TestPoolRunsTask(t *testing.T) {
	p := newPool(t, Config{Isolation: StrictIsolation()})
	in := []byte("hello from the other process")
	res, err := p.Run(t.Context(), "", sandbox.Task{Module: w.Echo(), Input: in, Limits: testLimits()})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(res.Output, in) {
		t.Fatalf("output %q", res.Output)
	}

	// Gas is deterministic across process boundaries.
	r := sandbox.NewRunner(sandbox.DefaultLimits())
	defer r.Close(t.Context())
	local, err := r.Run(t.Context(), sandbox.Task{Module: w.Echo(), Input: in, Limits: testLimits()})
	if err != nil {
		t.Fatal(err)
	}
	if local.GasUsed != res.GasUsed {
		t.Fatalf("gas %d in worker, %d in process", res.GasUsed, local.GasUsed)
	}
}

func TestPoolFailedTaskGasCrossesBoundary(t *testing.T) {
	p := newPool(t, Config{Isolation: StrictIsolation()})
	task := sandbox.Task{Module: w.CountdownTrap(100_000), Limits: testLimits()}
	_, err := p.Run(t.Context(), "", task)
	gas, ok := sandbox.GasUsed(err)
	if !errors.Is(err, sandbox.ErrTrap) || !ok {
		t.Fatalf("got %v, gas %d (%v)", err, gas, ok)
	}
	r := sandbox.NewRunner(sandbox.DefaultLimits())
	defer r.Close(t.Context())
	_, lerr := r.Run(t.Context(), task)
	if local, _ := sandbox.GasUsed(lerr); local != gas {
		t.Fatalf("gas %d in worker, %d in process", gas, local)
	}
	// Before running, there is no gas to report.
	if _, err := p.Run(t.Context(), "", sandbox.Task{Module: []byte("junk"), Limits: testLimits()}); !errors.Is(err, sandbox.ErrInvalidModule) {
		t.Fatalf("got %v", err)
	} else if _, ok := sandbox.GasUsed(err); ok {
		t.Fatal("invalid module reports gas")
	}
}

func TestDecodeResponseRejectsGasOutsideBudget(t *testing.T) {
	lim := testLimits()
	resp := encodeResponse(nil, sandbox.WithGasUsed(sandbox.ErrTrap, lim.MaxGas+1))
	if _, _, err := decodeResponse(resp[1:], lim); err == nil {
		t.Fatal("accepted a failed task's gas over its budget")
	}
	resp = encodeResponse(nil, sandbox.WithGasUsed(sandbox.ErrTrap, 7))
	_, taskErr, err := decodeResponse(resp[1:], lim)
	if gas, ok := sandbox.GasUsed(taskErr); err != nil || !errors.Is(taskErr, sandbox.ErrTrap) || !ok || gas != 7 {
		t.Fatalf("decoded %v, %v, gas %d (%v)", taskErr, err, gas, ok)
	}
}

func TestPoolSandboxErrorsCrossBoundary(t *testing.T) {
	p := newPool(t, Config{Isolation: StrictIsolation()})
	lim := testLimits()

	lim.MaxGas = 1000
	if _, err := p.Run(t.Context(), "alice", sandbox.Task{Module: w.Spin(), Limits: lim}); !errors.Is(err, sandbox.ErrOutOfGas) {
		t.Fatalf("got %v, want ErrOutOfGas", err)
	}
	lim = testLimits()
	lim.Timeout = 200 * time.Millisecond
	if _, err := p.Run(t.Context(), "alice", sandbox.Task{Module: w.Spin(), Limits: lim}); !errors.Is(err, sandbox.ErrTimeout) {
		t.Fatalf("got %v, want ErrTimeout", err)
	}
	if _, err := p.Run(t.Context(), "", sandbox.Task{Module: []byte("junk"), Limits: testLimits()}); !errors.Is(err, sandbox.ErrInvalidModule) {
		t.Fatalf("got %v, want ErrInvalidModule", err)
	}
	// Sandbox errors leave the worker healthy and reusable.
	if got := p.reused.Load(); got != 1 {
		t.Fatalf("reused %d times, want 1", got)
	}
}

// The worker's own timeout, not the node's kill, must stop a spinning task,
// and promptly. A regression here once came from interrupting through the
// gas counter, which the guest's writes kept overwriting.
func TestPoolTimeoutIsPrecise(t *testing.T) {
	p := newPool(t, Config{Isolation: StrictIsolation()})
	lim := testLimits()
	lim.Timeout = 100 * time.Millisecond
	for range 5 {
		start := time.Now()
		_, err := p.Run(t.Context(), "alice", sandbox.Task{Module: w.Spin(), Limits: lim})
		if !errors.Is(err, sandbox.ErrTimeout) || strings.Contains(err.Error(), "unresponsive") {
			t.Fatalf("got %v, want in-worker timeout", err)
		}
		if el := time.Since(start); el > 300*time.Millisecond {
			t.Fatalf("timeout after %s", el)
		}
	}
}

// A module within structural limits can still be slow and memory hungry to
// compile. Neither may hold a worker beyond the task's limits, and the pool
// must recover.
func TestCompileBombIsContained(t *testing.T) {
	defer func(g time.Duration) { workerGrace = g }(workerGrace)
	workerGrace = 300 * time.Millisecond
	cases := map[string]struct {
		memory  int64
		timeout time.Duration
		want    []error
	}{
		// The default worker memory is far below what compilation wants.
		// The timeout is long so that only memory stops it: compilation
		// reaches the limit in 0.5-1 s, which a slow machine can stretch
		// past a short timeout.
		"memory": {0, 10 * time.Second, []error{ErrWorkerOOM}},
		// With ample memory, the timeout stops it (or memory, if it grows
		// faster than the timeout on this machine).
		"time": {4 << 30, time.Second, []error{sandbox.ErrTimeout, ErrWorkerOOM}},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			lim := testLimits()
			lim.Timeout = c.timeout
			if raceEnabled && name == "memory" {
				t.Skip("address space limits are meaningless under ThreadSanitizer")
			}
			iso := StrictIsolation()
			iso.MemoryBytes = c.memory
			p := newPool(t, Config{Isolation: iso})
			start := time.Now()
			_, err := p.Run(t.Context(), "", sandbox.Task{Module: w.CompileBomb(16_000), Limits: lim})
			matched := false
			for _, want := range c.want {
				matched = matched || errors.Is(err, want)
			}
			if !matched {
				t.Fatalf("got %v, want one of %v", err, c.want)
			}
			if el := time.Since(start); el > lim.Timeout+workerGrace+500*time.Millisecond {
				t.Fatalf("compile bomb held the pool for %s", el)
			}
			if _, err := p.Run(t.Context(), "", sandbox.Task{Module: w.Echo(), Limits: testLimits()}); err != nil {
				t.Fatalf("pool did not recover: %v", err)
			}
		})
	}
}

func TestCgroupLimits(t *testing.T) {
	if os.Getenv("RAKKAN_TEST_CGROUP") == "" {
		t.Skip("set RAKKAN_TEST_CGROUP=1 and run inside a delegated cgroup, see docs/ARCHITECTURE.md")
	}
	iso := StrictIsolation()
	iso.Cgroup = "self"
	iso.MemoryBytes = 96 << 20
	iso.CPUs = 0.5
	p := newPool(t, Config{Isolation: iso, Workers: 2})
	if _, err := p.Run(t.Context(), "", sandbox.Task{Module: w.Echo(), Limits: testLimits()}); err != nil {
		t.Fatal(err)
	}

	// A worker exceeding memory.max is OOM-killed and reported as such.
	cg, err := p.cg.newWorker(iso)
	if err != nil {
		t.Fatal(err)
	}
	iso.MemoryBytes = 0 // no address space limit: let the cgroup act
	cmd := exec.Command(os.Args[0], "probe=alloc")
	cmd.Env = []string{workerEnv + "=" + iso.encode(iso.startStage())}
	cmd.SysProcAttr = iso.sysProcAttr()
	setCgroupFD(cmd.SysProcAttr, cg)
	err = cmd.Run()
	if err == nil || !cg.oomKilled() {
		t.Fatalf("1 GiB allocation under a 96 MiB cgroup: err=%v oom=%v", err, cg.oomKilled())
	}
	cg.remove()

	p.Close()
	ents, _ := os.ReadDir(p.cg.dir)
	for _, e := range ents {
		if strings.HasPrefix(e.Name(), "worker-") {
			t.Errorf("worker cgroup %s left behind", e.Name())
		}
	}
}

func TestPoolFreshProcessPerTask(t *testing.T) {
	p := newPool(t, Config{Isolation: StrictIsolation()})
	for range 3 {
		if _, err := p.Run(t.Context(), "", sandbox.Task{Module: w.Echo(), Limits: testLimits()}); err != nil {
			t.Fatal(err)
		}
	}
	if got := p.spawned.Load(); got < 3 {
		t.Fatalf("spawned %d workers for 3 tasks", got)
	}
	if got := p.reused.Load(); got != 0 {
		t.Fatalf("anonymous tasks reused workers %d times", got)
	}
}

func TestPoolAffinity(t *testing.T) {
	p := newPool(t, Config{Isolation: StrictIsolation(), Workers: 4, TasksPerWorker: 3})
	run := func(owner string, mod []byte) {
		t.Helper()
		if _, err := p.Run(t.Context(), owner, sandbox.Task{Module: mod, Limits: testLimits()}); err != nil {
			t.Fatal(err)
		}
	}
	echo, spin := w.Echo(), w.CompileBomb(10)
	run("alice", echo)
	run("alice", echo)
	run("alice", echo)
	if got := p.reused.Load(); got != 2 {
		t.Fatalf("reused %d times for one owner and module, want 2", got)
	}
	run("alice", echo) // TasksPerWorker reached: fresh worker
	run("bob", echo)   // other owner: fresh worker
	run("alice", spin) // other module: fresh worker
	if got := p.reused.Load(); got != 2 {
		t.Fatalf("reused %d times across owners or modules, want 2", got)
	}
}

func TestPoolAffinitySkipsCompilation(t *testing.T) {
	mod := w.CompileBomb(500)
	lim := sandbox.DefaultLimits()
	lim.Timeout = time.Minute // compilation is slow under -race on a loaded machine
	p := newPool(t, Config{Isolation: StrictIsolation(), Ceiling: lim})
	first, err := p.Run(t.Context(), "alice", sandbox.Task{Module: mod, Limits: lim})
	if err != nil {
		t.Fatal(err)
	}
	second, err := p.Run(t.Context(), "alice", sandbox.Task{Module: mod, Limits: lim})
	if err != nil {
		t.Fatal(err)
	}
	if second.Elapsed*5 > first.Elapsed {
		t.Fatalf("second run took %s, first %s", second.Elapsed, first.Elapsed)
	}
	t.Logf("first %s, second %s", first.Elapsed, second.Elapsed)
}

func TestPoolRetiresIdleWorkers(t *testing.T) {
	p := newPool(t, Config{Isolation: StrictIsolation(), IdleTimeout: 100 * time.Millisecond})
	if _, err := p.Run(t.Context(), "alice", sandbox.Task{Module: w.Echo(), Limits: testLimits()}); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for {
		p.mu.Lock()
		n := len(p.parked)
		p.mu.Unlock()
		if n == 0 {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("%d idle workers still parked", n)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func TestPoolConcurrent(t *testing.T) {
	p := newPool(t, Config{Isolation: StrictIsolation(), Workers: 4})
	errs := make(chan error, 16)
	for i := range 16 {
		go func() {
			in := []byte{byte(i)}
			res, err := p.Run(t.Context(), "", sandbox.Task{Module: w.Echo(), Input: in, Limits: testLimits()})
			if err == nil && !bytes.Equal(res.Output, in) {
				err = errors.New("output mismatch")
			}
			errs <- err
		}()
	}
	for range 16 {
		if err := <-errs; err != nil {
			t.Fatal(err)
		}
	}
}

func TestPoolCancellationKillsWorker(t *testing.T) {
	p := newPool(t, Config{Isolation: StrictIsolation()})
	ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer cancel()
	start := time.Now()
	if _, err := p.Run(ctx, "", sandbox.Task{Module: w.Spin(), Limits: testLimits()}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("got %v, want context.DeadlineExceeded", err)
	}
	if el := time.Since(start); el > time.Second {
		t.Fatalf("cancellation took %s", el)
	}
}

func TestPoolMisbehavingWorkers(t *testing.T) {
	defer func(g time.Duration) { workerGrace = g }(workerGrace)
	workerGrace = 200 * time.Millisecond
	lim := testLimits()
	lim.Timeout = 100 * time.Millisecond

	cases := map[string]error{
		"crash":   ErrWorkerCrashed,
		"hang":    sandbox.ErrTimeout,
		"lie":     ErrWorkerProtocol,
		"badread": ErrWorkerProtocol,
	}
	for mode, want := range cases {
		t.Run(mode, func(t *testing.T) {
			p := newPool(t, Config{Isolation: StrictIsolation(), args: []string{mode}})
			_, err := p.Run(t.Context(), "", sandbox.Task{Module: w.Echo(), Limits: lim})
			if !errors.Is(err, want) {
				t.Fatalf("got %v, want %v", err, want)
			}
		})
	}
}

func TestPoolRejectsOverPolicy(t *testing.T) {
	p := newPool(t, Config{Isolation: StrictIsolation()})
	lim := sandbox.DefaultLimits()
	lim.MaxGas *= 2
	if _, err := p.Run(t.Context(), "", sandbox.Task{Module: w.Echo(), Limits: lim}); !errors.Is(err, sandbox.ErrLimitPolicy) {
		t.Fatalf("got %v, want ErrLimitPolicy", err)
	}
}

// runProbe starts a hardened worker in probe mode and returns its report,
// or "killed: <signal>" if the kernel killed it.
func runProbe(t *testing.T, iso Isolation, name string) string {
	t.Helper()
	requireIsolationSupport(t)
	cmd := exec.Command(os.Args[0], "probe="+name)
	cmd.Env = []string{workerEnv + "=" + iso.encode(iso.startStage())}
	cmd.SysProcAttr = iso.sysProcAttr()
	cmd.Dir = "/"
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	var exit *exec.ExitError
	if errors.As(err, &exit) {
		if ws := exit.Sys().(syscall.WaitStatus); ws.Signaled() {
			return "killed: " + ws.Signal().String()
		}
		return fmt.Sprintf("exited %d: %s", exit.ExitCode(), lastLine(stderr.String()))
	}
	if err != nil {
		t.Fatal(err)
	}
	return strings.TrimSpace(string(out))
}

func lastLine(s string) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	return lines[0]
}

// onlyWorkerFiles accepts a root containing the worker binary and, for a
// dynamically linked binary, the loader and the shared objects it maps.
func onlyWorkerFiles(report string) bool {
	files, ok := strings.CutPrefix(report, "files: ")
	if !ok {
		return false
	}
	for _, f := range strings.Fields(files) {
		if f != "/w" && !strings.Contains(filepath.Base(f), ".so") {
			return false
		}
	}
	return true
}

func TestHardenedWorkerConfinement(t *testing.T) {
	strict := StrictIsolation()
	nsOnly := Isolation{Namespaces: true}

	type expect func(string) bool
	denied := func(s string) bool { return strings.HasPrefix(s, "denied") }
	deniedOrKilled := func(s string) bool { return denied(s) || s == "killed: bad system call" }
	cases := []struct {
		iso   Isolation
		probe string
		ok    expect
	}{
		// Namespaces alone: empty read-only root, no capabilities, no network.
		{nsOnly, "read-file", denied},
		{nsOnly, "list-root", onlyWorkerFiles},
		{nsOnly, "write-file", denied},
		{nsOnly, "mount", denied},
		{nsOnly, "caps", func(s string) bool { return s == "caps: 0 0 0" }},
		{nsOnly, "connect", denied},
		// With seccomp on top: dangerous calls are refused or fatal.
		{strict, "read-file", denied},
		{strict, "list-root", deniedOrKilled},
		{strict, "socket", deniedOrKilled},
		{strict, "fork", deniedOrKilled},
		{strict, "mount", deniedOrKilled},
		{strict, "prctl-vma", func(s string) bool { return s == "allowed" }},
		{strict, "prctl-other", func(s string) bool { return s == "killed: bad system call" }},
	}
	for _, c := range cases {
		name := c.probe
		if c.iso.Seccomp {
			name += "/seccomp"
		}
		t.Run(name, func(t *testing.T) {
			got := runProbe(t, c.iso, c.probe)
			if !c.ok(got) {
				t.Fatalf("probe %s: %s", c.probe, got)
			}
			t.Log(got)
		})
	}
}

func TestAddressSpaceLimit(t *testing.T) {
	if raceEnabled {
		t.Skip("address space limits are meaningless under ThreadSanitizer")
	}
	iso := StrictIsolation()
	iso.MemoryBytes = 128 << 20
	if got := runProbe(t, iso, "alloc"); !strings.HasPrefix(got, "exited") {
		t.Fatalf("1 GiB allocation under a 128 MiB limit: %s", got)
	}
}

type memBlob struct {
	data []byte
	fail bool
}

func (m memBlob) Size() int64 { return int64(len(m.data)) }

func (m memBlob) ReadAt(p []byte, off int64) (int, error) {
	if m.fail {
		return 0, errors.New("provider gone")
	}
	return bytes.NewReader(m.data).ReadAt(p, off)
}

func TestPoolBlobReads(t *testing.T) {
	p := newPool(t, Config{Isolation: StrictIsolation(), TasksPerWorker: 10})
	data := bytes.Repeat([]byte("rakkan blob "), 250_000) // 3 MB: several pipe round trips
	res, err := p.Run(t.Context(), "alice", sandbox.Task{Module: w.BlobEcho(), Blobs: []sandbox.BlobSource{memBlob{data: data}}, Limits: testLimits()})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(res.Output, data) {
		t.Fatalf("read back %d of %d bytes", len(res.Output), len(data))
	}
	// A failing source on the node's side reaches the task as such.
	_, err = p.Run(t.Context(), "alice", sandbox.Task{Module: w.BlobEcho(), Blobs: []sandbox.BlobSource{memBlob{data: data, fail: true}}, Limits: testLimits()})
	if !errors.Is(err, sandbox.ErrBlobUnavailable) {
		t.Fatalf("got %v, want ErrBlobUnavailable", err)
	}
}
