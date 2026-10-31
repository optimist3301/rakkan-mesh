package isolate

import (
	"bufio"
	"bytes"
	"debug/elf"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"

	"golang.org/x/sys/unix"
)

// Worker start-up on Linux happens in two stages.
//
// Stage 1 runs as root of a fresh user namespace, holding every capability
// inside it. It replaces the filesystem with an empty read-only tmpfs that
// contains only a bind mount of the worker binary, sets securebits so that
// root no longer gains capabilities on execve, and re-executes the binary.
//
// Stage 2 therefore starts with no capabilities in any thread, which Go
// cannot otherwise guarantee once the runtime has started threads (capset
// is per thread, and AllThreadsSyscall is unavailable when cgo is linked,
// as it is through wazero's import of net). It then applies resource limits,
// no_new_privs and the seccomp filter before serving tasks.
//
// Without namespaces only stage 2 runs.

// selfPath is where the worker binary is visible inside the empty root.
const selfPath = "/w"

// startStage is the stage the node starts a worker at.
func (iso Isolation) startStage() int {
	if iso.Namespaces {
		return 1
	}
	return 2
}

// sysProcAttr configures how the node spawns workers.
func (iso Isolation) sysProcAttr() *syscall.SysProcAttr {
	attr := &syscall.SysProcAttr{
		// Workers must not outlive the node.
		Pdeathsig: syscall.SIGKILL,
		Setpgid:   true,
	}
	if iso.Namespaces {
		attr.Cloneflags = syscall.CLONE_NEWUSER | syscall.CLONE_NEWNS | syscall.CLONE_NEWPID |
			syscall.CLONE_NEWNET | syscall.CLONE_NEWIPC | syscall.CLONE_NEWUTS
		attr.UidMappings = []syscall.SysProcIDMap{{ContainerID: 0, HostID: os.Getuid(), Size: 1}}
		attr.GidMappings = []syscall.SysProcIDMap{{ContainerID: 0, HostID: os.Getgid(), Size: 1}}
		attr.GidMappingsEnableSetgroups = false
	}
	return attr
}

func checkPlatform(iso Isolation) error {
	if !iso.Namespaces && os.Geteuid() == 0 {
		return fmt.Errorf("refusing to run workers as root without namespaces")
	}
	if iso.Seccomp && auditArch == 0 {
		return fmt.Errorf("seccomp filter not available on %s", runtime.GOARCH)
	}
	return nil
}

func harden(iso Isolation, stage int) error {
	switch {
	case stage == 1 && iso.Namespaces:
		if err := limitAddressSpace(iso); err != nil {
			return err
		}
		return enterEmptyRootAndReexec(iso)
	case stage == 2:
		if !iso.Namespaces {
			if err := limitAddressSpace(iso); err != nil {
				return err
			}
		}
		prepareRuntime(iso)
		if err := setRlimits(); err != nil {
			return err
		}
		// no_new_privs is per thread and seccomp checks it on the calling
		// thread, so both must run on the same one; TSYNC then propagates
		// the filter and the flag to every other thread.
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		if err := unix.Prctl(unix.PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0); err != nil {
			return fmt.Errorf("no_new_privs: %w", err)
		}
		if iso.Seccomp {
			return installSeccomp()
		}
		return nil
	}
	return fmt.Errorf("invalid worker stage %d", stage)
}

func enterEmptyRootAndReexec(iso Isolation) error {
	// Hold the binary open: the tmpfs may hide the directory it lives in.
	exe, err := unix.Open("/proc/self/exe", unix.O_PATH|unix.O_CLOEXEC, 0)
	if err != nil {
		return fmt.Errorf("open worker binary: %w", err)
	}
	defer unix.Close(exe)
	exePath := fmt.Sprintf("/proc/self/fd/%d", exe)
	libs, err := loaderFiles(exePath)
	if err != nil {
		return err
	}

	// Mount events must not propagate back to the parent namespace.
	if err := unix.Mount("", "/", "", unix.MS_PRIVATE|unix.MS_REC, ""); err != nil {
		return fmt.Errorf("make mounts private: %w", err)
	}
	const newRoot = "/tmp"
	if err := unix.Mount("tmpfs", newRoot, "tmpfs", unix.MS_NOSUID|unix.MS_NODEV, "size=1m,mode=0755"); err != nil {
		return fmt.Errorf("mount tmpfs: %w", err)
	}
	if err := bindReadOnly(exePath, newRoot+selfPath); err != nil {
		return fmt.Errorf("bind worker binary: %w", err)
	}
	for _, lib := range libs {
		if err := bindReadOnly(lib.source, newRoot+lib.target); err != nil {
			return fmt.Errorf("bind %s: %w", lib.target, err)
		}
	}
	if err := os.Mkdir(newRoot+"/old", 0o700); err != nil {
		return fmt.Errorf("create old root mount point: %w", err)
	}
	if err := unix.PivotRoot(newRoot, newRoot+"/old"); err != nil {
		return fmt.Errorf("pivot_root: %w", err)
	}
	if err := os.Chdir("/"); err != nil {
		return fmt.Errorf("chdir to new root: %w", err)
	}
	if err := unix.Unmount("/old", unix.MNT_DETACH); err != nil {
		return fmt.Errorf("detach old root: %w", err)
	}
	if err := os.Remove("/old"); err != nil {
		return fmt.Errorf("remove old root mount point: %w", err)
	}
	if err := unix.Mount("", "/", "", unix.MS_REMOUNT|unix.MS_RDONLY|unix.MS_NOSUID|unix.MS_NODEV, ""); err != nil {
		return fmt.Errorf("remount root read-only: %w", err)
	}

	// securebits and execve act on the calling thread.
	runtime.LockOSThread()
	const securebits = 1<<0 | 1<<1 | // SECBIT_NOROOT, locked
		1<<2 | 1<<3 | // SECBIT_NO_SETUID_FIXUP, locked
		1<<5 | // SECBIT_KEEP_CAPS_LOCKED (KEEP_CAPS off)
		1<<6 | 1<<7 // SECBIT_NO_CAP_AMBIENT_RAISE, locked
	if err := unix.Prctl(unix.PR_SET_SECUREBITS, securebits, 0, 0, 0); err != nil {
		return fmt.Errorf("securebits: %w", err)
	}
	env := []string{workerEnv + "=" + iso.encode(2)}
	if err := syscall.Exec(selfPath, os.Args, env); err != nil {
		return fmt.Errorf("re-exec worker: %w", err)
	}
	return nil
}

// bindReadOnly makes the file source visible read-only at target.
func bindReadOnly(source, target string) error {
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(target, nil, 0o500); err != nil {
		return err
	}
	if err := unix.Mount(source, target, "", unix.MS_BIND, ""); err != nil {
		return err
	}
	return unix.Mount("", target, "", unix.MS_REMOUNT|unix.MS_BIND|unix.MS_RDONLY|unix.MS_NOSUID|unix.MS_NODEV, "")
}

type loaderFile struct{ source, target string }

// loaderFiles lists what the dynamic loader needs to execute the worker
// binary inside the empty root: its ELF interpreter and the shared objects
// currently mapped into this process, at the paths they are found under.
// A statically linked binary (CGO_ENABLED=0, recommended) needs none.
func loaderFiles(exePath string) ([]loaderFile, error) {
	f, err := elf.Open(exePath)
	if err != nil {
		return nil, fmt.Errorf("read worker binary: %w", err)
	}
	defer f.Close()
	var interp string
	for _, p := range f.Progs {
		if p.Type == elf.PT_INTERP {
			b, err := io.ReadAll(p.Open())
			if err != nil {
				return nil, err
			}
			interp = strings.TrimRight(string(b), "\x00")
		}
	}
	if interp == "" {
		return nil, nil
	}
	real, err := filepath.EvalSymlinks(interp)
	if err != nil {
		return nil, fmt.Errorf("resolve ELF interpreter: %w", err)
	}
	files := []loaderFile{{source: real, target: interp}}

	maps, err := os.ReadFile("/proc/self/maps")
	if err != nil {
		return nil, err
	}
	seen := map[string]bool{interp: true, real: true}
	for _, line := range strings.Split(string(maps), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 6 {
			continue
		}
		path := fields[5]
		if !strings.HasPrefix(path, "/") || !strings.Contains(path, ".so") || seen[path] {
			continue
		}
		seen[path] = true
		files = append(files, loaderFile{source: path, target: path})
	}
	return files, nil
}

func setRlimits() error {
	limits := []struct {
		res int
		val uint64
	}{
		{unix.RLIMIT_CORE, 0},
		{unix.RLIMIT_FSIZE, 0},
		{unix.RLIMIT_NOFILE, 64},
		{unix.RLIMIT_MEMLOCK, 0},
		{unix.RLIMIT_MSGQUEUE, 0},
		{unix.RLIMIT_LOCKS, 0},
	}
	for _, l := range limits {
		if err := unix.Setrlimit(l.res, &unix.Rlimit{Cur: l.val, Max: l.val}); err != nil {
			return fmt.Errorf("setrlimit %d: %w", l.res, err)
		}
	}
	return nil
}

// limitAddressSpace caps virtual memory when no cgroup enforces a limit.
// The Go runtime reserves address space up front, so the cap is relative
// to the current size; the limit is inherited across the stage 2 execve.
func limitAddressSpace(iso Isolation) error {
	if iso.Cgroup != "" || iso.MemoryBytes <= 0 {
		return nil
	}
	vm, err := currentVirtualSize()
	if err != nil {
		return err
	}
	lim := uint64(vm) + uint64(iso.MemoryBytes)
	return unix.Setrlimit(unix.RLIMIT_AS, &unix.Rlimit{Cur: lim, Max: lim})
}

func currentVirtualSize() (int64, error) {
	data, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return 0, err
	}
	sc := bufio.NewScanner(bytes.NewReader(data))
	for sc.Scan() {
		if v, ok := strings.CutPrefix(sc.Text(), "VmSize:"); ok {
			kb, err := strconv.ParseInt(strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(v), "kB")), 10, 64)
			return kb << 10, err
		}
	}
	return 0, fmt.Errorf("VmSize not found in /proc/self/status")
}
