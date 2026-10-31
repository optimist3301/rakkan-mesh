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
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
	"slices"
	"strings"
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
	// TasksPerWorker bounds how many tasks of one owner and module a worker
	// serves before it is replaced (see Run). Default: 100.
	TasksPerWorker int
	// IdleTimeout retires workers kept for an owner and module after this
	// long without use. Default: 30s.
	IdleTimeout time.Duration
	Isolation   Isolation
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
	idle   chan *worker // fresh workers that have run nothing yet
	closed atomic.Bool
	bg     sync.WaitGroup
	stop   chan struct{}

	mu     sync.Mutex
	parked []*worker // workers bound to an owner and module, oldest first

	spawned atomic.Int64
	reused  atomic.Int64
}

// NewPool validates the configuration and starts Workers warm workers, so
// that isolation problems surface here rather than on the first task.
func NewPool(cfg Config) (*Pool, error) {
	if cfg.Workers <= 0 {
		cfg.Workers = runtime.GOMAXPROCS(0)
	}
	if cfg.TasksPerWorker <= 0 {
		cfg.TasksPerWorker = 100
	}
	if cfg.IdleTimeout <= 0 {
		cfg.IdleTimeout = 30 * time.Second
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
		stop:  make(chan struct{}),
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
	p.bg.Go(p.reapIdle)
	return p, nil
}

// Close stops all workers. Tasks in flight are not interrupted.
func (p *Pool) Close() error {
	if p.closed.Swap(true) {
		return nil
	}
	close(p.stop)
	p.mu.Lock()
	parked := p.parked
	p.parked = nil
	p.mu.Unlock()
	for _, w := range parked {
		w.retire()
	}
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
//
// A task with an empty owner runs in a process that has never run anything
// else. Tasks naming an owner may reuse a process that already ran the same
// module for the same owner, which then skips start-up, instrumentation and
// compilation. Processes are never shared between owners or modules, so a
// task that compromises its worker can reach only data its owner already
// gave to the same code.
func (p *Pool) Run(ctx context.Context, owner string, t sandbox.Task) (*sandbox.Result, error) {
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

	var key string
	if owner != "" {
		h := sha256.Sum256(t.Module)
		key = owner + "\x00" + string(h[:])
	}
	w, err := p.acquire(key)
	if err != nil {
		return nil, err
	}
	res, healthy, err := w.run(ctx, t)
	p.release(w, key, healthy)
	return res, err
}

func (p *Pool) acquire(key string) (*worker, error) {
	if key != "" {
		p.mu.Lock()
		for i, w := range p.parked {
			if w.key == key {
				p.parked = slices.Delete(p.parked, i, i+1)
				p.mu.Unlock()
				p.reused.Add(1)
				return w, nil
			}
		}
		p.mu.Unlock()
	}
	select {
	case w := <-p.idle:
		// Keep a warm replacement so the next task skips start-up.
		p.bg.Go(p.replenish)
		w.key = key
		return w, nil
	default:
		w, err := p.spawn()
		if w != nil {
			w.key = key
		}
		return w, err
	}
}

func (p *Pool) replenish() {
	if p.closed.Load() {
		return
	}
	w, err := p.spawn()
	if err != nil {
		return // the next Run spawns synchronously and reports the error
	}
	select {
	case p.idle <- w:
	default:
		w.retire()
	}
}

// release keeps a worker for its owner and module, or retires it.
func (p *Pool) release(w *worker, key string, healthy bool) {
	if healthy && key != "" && w.tasks < p.cfg.TasksPerWorker && !p.closed.Load() {
		w.parkedAt = time.Now()
		p.mu.Lock()
		p.parked = append(p.parked, w)
		var evicted *worker
		if len(p.parked) > p.cfg.Workers {
			evicted = p.parked[0]
			p.parked = p.parked[1:]
		}
		p.mu.Unlock()
		if evicted != nil {
			p.bg.Go(evicted.retire)
		}
		return
	}
	p.bg.Go(w.retire)
}

// reapIdle retires parked workers unused for IdleTimeout.
func (p *Pool) reapIdle() {
	tick := time.NewTicker(max(p.cfg.IdleTimeout/4, 10*time.Millisecond))
	defer tick.Stop()
	for {
		select {
		case <-p.stop:
			return
		case <-tick.C:
		}
		deadline := time.Now().Add(-p.cfg.IdleTimeout)
		p.mu.Lock()
		var stale []*worker
		p.parked = slices.DeleteFunc(p.parked, func(w *worker) bool {
			if w.parkedAt.Before(deadline) {
				stale = append(stale, w)
				return true
			}
			return false
		})
		p.mu.Unlock()
		for _, w := range stale {
			w.retire()
		}
	}
}

type worker struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
	stderr *tailBuffer
	cg     *workerCgroup
	exited chan struct{}
	tasks  int

	key      string // owner and module the worker is bound to, if any
	parkedAt time.Time
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
	case strings.Contains(w.stderr.String(), "fatal error: out of memory"):
		// The Go runtime hit the address space limit. The report comes
		// from the worker, but misreporting only affects its own task.
		return fmt.Errorf("%w (address space limit)", ErrWorkerOOM)
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

// tailBuffer keeps the first and last bytes written to it: a Go runtime
// crash states its cause at the start and goroutine dumps follow.
type tailBuffer struct {
	mu   sync.Mutex
	head []byte
	tail []byte
	cut  bool
}

const (
	headSize = 1 << 10
	tailSize = 3 << 10
)

func (t *tailBuffer) Write(p []byte) (int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	n := len(p)
	if room := headSize - len(t.head); room > 0 {
		k := min(room, len(p))
		t.head = append(t.head, p[:k]...)
		p = p[k:]
	}
	t.tail = append(t.tail, p...)
	if len(t.tail) > tailSize {
		t.tail = append(t.tail[:0], t.tail[len(t.tail)-tailSize:]...)
		t.cut = true
	}
	return n, nil
}

func (t *tailBuffer) String() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.cut {
		return string(t.head) + "\n[...]\n" + string(t.tail)
	}
	return string(t.head) + string(t.tail)
}
