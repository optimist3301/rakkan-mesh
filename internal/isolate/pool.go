// Package isolate runs sandbox tasks in separate, hardened worker processes.
//
// The WebAssembly sandbox is the first line of defence; this package is the
// second, for when the first fails: a bug in the engine, the compiler or the
// metering must not give a task access to the node. Each worker is a re-exec
// of the node binary (see IsWorker and RunWorker) with an empty filesystem,
// no network, no capabilities, a seccomp allow list and its own memory, CPU
// and process limits (see Isolation). The node treats every worker response
// as untrusted and kills workers that crash, hang or misbehave.
package isolate

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/sandbox"
)

var (
	ErrWorkerCrashed  = errors.New("sandbox worker crashed")
	ErrWorkerOOM      = errors.New("sandbox worker exceeded its memory limit")
	ErrWorkerProtocol = errors.New("sandbox worker violated the protocol")
	ErrWorkerInternal = errors.New("sandbox worker internal error")
	ErrPoolClosed     = errors.New("worker pool closed")
)

var (
	// workerGrace is how long past a task's timeout the node waits for the
	// worker's own interruption before killing it.
	workerGrace = 2 * time.Second
	// workerStartTimeout bounds worker start-up, including hardening.
	workerStartTimeout = 10 * time.Second
)

// Config configures a Pool.
type Config struct {
	// Ceiling is the node's policy; tasks requesting more are rejected.
	Ceiling sandbox.Limits
	// Workers is the number of tasks run concurrently. Default: GOMAXPROCS.
	Workers int
	// TasksPerWorker is how many tasks a worker process serves before it is
	// replaced. The default of 1 means no state, compromised or not, carries
	// over between tasks; larger values reuse compiled modules and save
	// process start-up.
	TasksPerWorker int
	Isolation      Isolation
	// Executable is the worker binary. Default: the running executable,
	// which must call RunWorker when IsWorker reports true.
	Executable string

	args []string // extra worker arguments, for tests
}

// Pool runs tasks in worker processes. It is safe for concurrent use.
type Pool struct {
	cfg    Config
	cg     *cgroupRoot
	slots  chan struct{}
	idle   chan *worker
	closed atomic.Bool
	bg     sync.WaitGroup

	spawned atomic.Int64
}

// NewPool validates the configuration and starts Workers warm workers, so
// that isolation problems surface here rather than on the first task.
func NewPool(cfg Config) (*Pool, error) {
	if cfg.Workers <= 0 {
		cfg.Workers = runtime.GOMAXPROCS(0)
	}
	if cfg.TasksPerWorker <= 0 {
		cfg.TasksPerWorker = 1
	}
	if cfg.Isolation.MemoryBytes <= 0 {
		// Linear memory may be copied when it grows, plus runtime and
		// compiled code.
		cfg.Isolation.MemoryBytes = 2*int64(cfg.Ceiling.MaxMemoryPages)*sandbox.PageSize + 256<<20
	}
	if cfg.Executable == "" {
		exe, err := os.Executable()
		if err != nil {
			return nil, err
		}
		cfg.Executable = exe
	}
	if err := checkPlatform(cfg.Isolation); err != nil {
		return nil, err
	}

	p := &Pool{
		cfg:   cfg,
		slots: make(chan struct{}, cfg.Workers),
		idle:  make(chan *worker, cfg.Workers),
	}
	if cfg.Isolation.Cgroup != "" {
		cg, err := openCgroupRoot(cfg.Isolation.Cgroup)
		if err != nil {
			return nil, err
		}
		p.cg = cg
	}
	for range cfg.Workers {
		w, err := p.spawn()
		if err != nil {
			p.Close()
			return nil, err
		}
		p.idle <- w
	}
	return p, nil
}

// Close stops all workers. Tasks in flight are not interrupted.
func (p *Pool) Close() error {
	p.closed.Store(true)
	for {
		select {
		case w := <-p.idle:
			w.retire()
		default:
			p.bg.Wait()
			return nil
		}
	}
}

// Run executes a task in a worker process.
func (p *Pool) Run(ctx context.Context, t sandbox.Task) (*sandbox.Result, error) {
	if err := t.Limits.Within(p.cfg.Ceiling); err != nil {
		return nil, err
	}
	// Cheap checks the worker would repeat, done here to avoid shipping
	// oversized frames.
	if len(t.Module) > t.Limits.MaxModuleBytes {
		return nil, fmt.Errorf("%w: %d bytes, limit %d", sandbox.ErrModuleTooLarge, len(t.Module), t.Limits.MaxModuleBytes)
	}
	if len(t.Input) > t.Limits.MaxInputBytes {
		return nil, fmt.Errorf("%w: %d bytes, limit %d", sandbox.ErrInputTooLarge, len(t.Input), t.Limits.MaxInputBytes)
	}

	select {
	case p.slots <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	defer func() { <-p.slots }()
	if p.closed.Load() {
		return nil, ErrPoolClosed
	}

	w, err := p.acquire()
	if err != nil {
		return nil, err
	}
	res, healthy, err := w.run(ctx, t)
	p.release(w, healthy)
	return res, err
}

func (p *Pool) acquire() (*worker, error) {
	select {
	case w := <-p.idle:
		return w, nil
	default:
		return p.spawn()
	}
}

// release returns a worker to the idle set or replaces it.
func (p *Pool) release(w *worker, healthy bool) {
	if healthy && w.tasks < p.cfg.TasksPerWorker && !p.closed.Load() {
		select {
		case p.idle <- w:
			return
		default:
		}
	}
	p.bg.Go(func() {
		w.retire()
		if p.closed.Load() {
			return
		}
		// Keep a warm replacement so the next task skips start-up.
		nw, err := p.spawn()
		if err != nil {
			return // the next Run spawns synchronously and reports the error
		}
		select {
		case p.idle <- nw:
		default:
			nw.retire()
		}
	})
}

type worker struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
	stderr *tailBuffer
	cg     *workerCgroup
	exited chan struct{}
	tasks  int
}

func (p *Pool) spawn() (*worker, error) {
	iso := p.cfg.Isolation
	cmd := exec.Command(p.cfg.Executable)
	cmd.Args = append([]string{"rakkan-worker"}, p.cfg.args...)
	cmd.Env = []string{workerEnv + "=" + iso.encode(iso.startStage())}
	cmd.Dir = "/"
	cmd.SysProcAttr = iso.sysProcAttr()

	w := &worker{cmd: cmd, stderr: &tailBuffer{}, exited: make(chan struct{})}
	cmd.Stderr = w.stderr
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	w.stdin, w.stdout = stdin, bufio.NewReader(stdout)

	if p.cg != nil {
		if w.cg, err = p.cg.newWorker(iso); err != nil {
			return nil, err
		}
		setCgroupFD(cmd.SysProcAttr, w.cg)
	}
	if err := cmd.Start(); err != nil {
		w.cleanup()
		return nil, fmt.Errorf("start worker: %w", err)
	}
	p.spawned.Add(1)
	go func() {
		cmd.Wait()
		w.cleanup()
		close(w.exited)
	}()

	// The worker reports readiness once hardened.
	ready := make(chan error, 1)
	go func() {
		frame, err := readFrame(w.stdout, 16)
		if err == nil && (len(frame) != 1 || frame[0] != protoVersion) {
			err = errors.New("bad ready message")
		}
		ready <- err
	}()
	select {
	case err = <-ready:
	case <-time.After(workerStartTimeout):
		err = errors.New("timed out")
	}
	if err != nil {
		w.kill()
		return nil, fmt.Errorf("worker start-up failed: %w%s", err, w.diagnostics())
	}
	return w, nil
}

// run sends a task and waits for its result. healthy reports whether the
// worker may serve further tasks.
func (w *worker) run(ctx context.Context, t sandbox.Task) (res *sandbox.Result, healthy bool, err error) {
	type reply struct {
		res     *sandbox.Result
		taskErr error
		err     error
	}
	done := make(chan reply, 1)
	go func() {
		if err := writeFrame(w.stdin, encodeRequest(t)); err != nil {
			done <- reply{err: err}
			return
		}
		limit := t.Limits.MaxOutputBytes + t.Limits.MaxLogBytes + maxErrorMessage + 64
		frame, err := readFrame(w.stdout, limit)
		if err != nil {
			done <- reply{err: err}
			return
		}
		res, taskErr, err := decodeResponse(frame, t.Limits)
		if err != nil {
			err = fmt.Errorf("%w: %v", ErrWorkerProtocol, err)
		}
		done <- reply{res, taskErr, err}
	}()

	timer := time.NewTimer(t.Limits.Timeout + workerGrace)
	defer timer.Stop()
	select {
	case r := <-done:
		if r.err != nil {
			w.kill()
			return nil, false, w.failure(r.err)
		}
		w.tasks++
		return r.res, !errors.Is(r.taskErr, ErrWorkerInternal), r.taskErr
	case <-timer.C:
		w.kill()
		return nil, false, fmt.Errorf("%w: worker unresponsive after %s, killed", sandbox.ErrTimeout, t.Limits.Timeout+workerGrace)
	case <-ctx.Done():
		w.kill()
		return nil, false, ctx.Err()
	}
}

// failure classifies an I/O or protocol error once the worker has exited.
func (w *worker) failure(err error) error {
	<-w.exited
	switch {
	case errors.Is(err, ErrWorkerProtocol):
		return err
	case w.cg != nil && w.cg.oomKilled():
		return ErrWorkerOOM
	}
	return fmt.Errorf("%w: %s%s", ErrWorkerCrashed, exitDescription(w.cmd.ProcessState), w.diagnostics())
}

func exitDescription(ps *os.ProcessState) string {
	if ps == nil {
		return "unknown exit"
	}
	if ws, ok := ps.Sys().(syscall.WaitStatus); ok && ws.Signaled() {
		if ws.Signal() == syscall.SIGSYS {
			return "killed by seccomp (SIGSYS)"
		}
		return "killed by " + ws.Signal().String()
	}
	return ps.String()
}

func (w *worker) diagnostics() string {
	if tail := w.stderr.String(); tail != "" {
		return "\nworker stderr:\n" + tail
	}
	return ""
}

func (w *worker) kill() {
	if w.cmd.Process != nil {
		w.cmd.Process.Kill()
	}
}

// retire asks the worker to exit by closing its input, killing it if it
// does not comply promptly.
func (w *worker) retire() {
	w.stdin.Close()
	select {
	case <-w.exited:
	case <-time.After(time.Second):
		w.kill()
		<-w.exited
	}
}

func (w *worker) cleanup() {
	if w.cg != nil {
		w.cg.remove()
	}
}

// tailBuffer keeps the last bytes written to it.
type tailBuffer struct {
	mu  sync.Mutex
	buf []byte
}

const tailSize = 4 << 10

func (t *tailBuffer) Write(p []byte) (int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.buf = append(t.buf, p...)
	if len(t.buf) > tailSize {
		t.buf = append(t.buf[:0], t.buf[len(t.buf)-tailSize:]...)
	}
	return len(p), nil
}

func (t *tailBuffer) String() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return string(t.buf)
}
