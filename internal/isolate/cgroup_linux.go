package isolate

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"golang.org/x/sys/unix"
)

const cgroupFS = "/sys/fs/cgroup"

// workerPids bounds threads per worker; the Go runtime needs a handful.
const workerPids = 64

// cgroupRoot manages per-worker cgroups below a directory delegated to the
// node. cgroup v2 forbids processes in a cgroup whose controllers are
// delegated to children, so the node moves itself into a "node" child first.
type cgroupRoot struct {
	dir  string
	next atomic.Uint64
}

func openCgroupRoot(dir string) (*cgroupRoot, error) {
	if dir == "self" {
		own, err := ownCgroup()
		if err != nil {
			return nil, err
		}
		dir = own
	}
	controllers, err := os.ReadFile(filepath.Join(dir, "cgroup.controllers"))
	if err != nil {
		return nil, fmt.Errorf("cgroup %s: %w", dir, err)
	}
	for _, c := range []string{"memory", "pids", "cpu"} {
		if !bytes.Contains(controllers, []byte(c)) {
			return nil, fmt.Errorf("cgroup %s: controller %q not delegated", dir, c)
		}
	}

	procs, err := os.ReadFile(filepath.Join(dir, "cgroup.procs"))
	if err != nil {
		return nil, err
	}
	if pids := strings.Fields(string(procs)); len(pids) > 0 {
		self := strconv.Itoa(os.Getpid())
		for _, p := range pids {
			if p != self {
				return nil, fmt.Errorf("cgroup %s: shared with process %s; delegate a cgroup to the node alone", dir, p)
			}
		}
		node := filepath.Join(dir, "node")
		if err := os.Mkdir(node, 0o755); err != nil && !errors.Is(err, os.ErrExist) {
			return nil, err
		}
		if err := os.WriteFile(filepath.Join(node, "cgroup.procs"), []byte(self), 0); err != nil {
			return nil, fmt.Errorf("move node into %s: %w", node, err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "cgroup.subtree_control"), []byte("+memory +pids +cpu"), 0); err != nil {
		return nil, fmt.Errorf("enable controllers in %s: %w", dir, err)
	}
	return &cgroupRoot{dir: dir}, nil
}

func ownCgroup() (string, error) {
	data, err := os.ReadFile("/proc/self/cgroup")
	if err != nil {
		return "", err
	}
	sc := bufio.NewScanner(bytes.NewReader(data))
	for sc.Scan() {
		if path, ok := strings.CutPrefix(sc.Text(), "0::"); ok {
			return filepath.Join(cgroupFS, path), nil
		}
	}
	return "", errors.New("not running under cgroup v2")
}

// workerCgroup is a per-worker cgroup. Its descriptor places the worker in
// it atomically at clone time (CLONE_INTO_CGROUP).
type workerCgroup struct {
	path string
	fd   int
}

func (c *cgroupRoot) newWorker(iso Isolation) (*workerCgroup, error) {
	path := filepath.Join(c.dir, fmt.Sprintf("worker-%d", c.next.Add(1)))
	if err := os.Mkdir(path, 0o755); err != nil {
		return nil, err
	}
	settings := [][2]string{
		{"memory.max", strconv.FormatInt(iso.MemoryBytes, 10)},
		{"memory.swap.max", "0"},
		{"memory.oom.group", "1"},
		{"pids.max", strconv.Itoa(workerPids)},
	}
	if iso.CPUs > 0 {
		const period = 100_000
		settings = append(settings, [2]string{"cpu.max", fmt.Sprintf("%d %d", int64(iso.CPUs*period), period)})
	}
	for _, s := range settings {
		if err := os.WriteFile(filepath.Join(path, s[0]), []byte(s[1]), 0); err != nil {
			os.Remove(path)
			return nil, fmt.Errorf("cgroup %s: %w", s[0], err)
		}
	}
	fd, err := unix.Open(path, unix.O_RDONLY|unix.O_DIRECTORY|unix.O_CLOEXEC, 0)
	if err != nil {
		os.Remove(path)
		return nil, err
	}
	return &workerCgroup{path: path, fd: fd}, nil
}

func setCgroupFD(attr *syscall.SysProcAttr, cg *workerCgroup) {
	attr.UseCgroupFD = true
	attr.CgroupFD = cg.fd
}

// oomKilled reports whether the kernel killed the worker for exceeding its
// memory limit.
func (w *workerCgroup) oomKilled() bool {
	data, err := os.ReadFile(filepath.Join(w.path, "memory.events"))
	if err != nil {
		return false
	}
	for _, line := range strings.Split(string(data), "\n") {
		if v, ok := strings.CutPrefix(line, "oom_kill "); ok {
			return v != "0"
		}
	}
	return false
}

// remove deletes the cgroup once its processes have exited.
func (w *workerCgroup) remove() {
	unix.Close(w.fd)
	for range 50 {
		if err := os.Remove(w.path); err == nil || errors.Is(err, os.ErrNotExist) {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
}
