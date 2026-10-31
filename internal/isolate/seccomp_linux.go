package isolate

import (
	"fmt"
	"unsafe"

	"golang.org/x/sys/unix"
)

// The seccomp filter allows the system calls a worker needs after start-up:
// memory management (the JIT maps executable code), threads, signals,
// futexes, clocks, and I/O on the descriptors it already holds. Opening
// files is refused with EPERM; creating processes, sockets, mounts or
// namespaces, ptrace, BPF and everything else kills the process.
//
// Per-architecture lists live in seccomp_<arch>.go.

const (
	seccompRetKillProcess = 0x80000000
	seccompRetErrno       = 0x00050000
	seccompRetAllow       = 0x7fff0000

	// Offsets into struct seccomp_data.
	offNr    = 0
	offArch  = 4
	offArg0  = 16 // low 32 bits on little-endian architectures
	bpfLd    = unix.BPF_LD | unix.BPF_W | unix.BPF_ABS
	bpfJeq   = unix.BPF_JMP | unix.BPF_JEQ | unix.BPF_K
	bpfJge   = unix.BPF_JMP | unix.BPF_JGE | unix.BPF_K
	bpfJset  = unix.BPF_JMP | unix.BPF_JSET | unix.BPF_K
	bpfRet   = unix.BPF_RET | unix.BPF_K
	x32Bit   = 0x40000000
	cloneThr = unix.CLONE_THREAD
)

// seccompDefaultAction applies to system calls not listed. Set it to
// SECCOMP_RET_TRAP (0x00030000) when extending the allow list: the Go
// runtime then crashes with SIGSYS and a stack trace naming the call.
var seccompDefaultAction uint32 = seccompRetKillProcess

type errnoRule struct {
	nr    uintptr
	errno unix.Errno
}

func stmt(code uint16, k uint32) unix.SockFilter {
	return unix.SockFilter{Code: code, K: k}
}

func jump(code uint16, k uint32, jt, jf uint8) unix.SockFilter {
	return unix.SockFilter{Code: code, Jt: jt, Jf: jf, K: k}
}

func seccompProgram() []unix.SockFilter {
	kill := stmt(bpfRet, seccompRetKillProcess)
	allow := stmt(bpfRet, seccompRetAllow)

	prog := []unix.SockFilter{
		stmt(bpfLd, offArch),
		jump(bpfJeq, auditArch, 1, 0),
		kill,
		stmt(bpfLd, offNr),
	}
	if hasX32 {
		prog = append(prog, jump(bpfJge, x32Bit, 0, 1), kill)
	}
	// clone is allowed for threads only; fork-like clones are killed.
	// clone3 passes flags in memory the filter cannot inspect, so it gets
	// ENOSYS and the Go runtime keeps using clone.
	prog = append(prog,
		jump(bpfJeq, uint32(unix.SYS_CLONE), 0, 4),
		stmt(bpfLd, offArg0),
		jump(bpfJset, cloneThr, 0, 1),
		allow,
		kill,
	)
	for _, r := range errnoSyscalls {
		prog = append(prog, jump(bpfJeq, uint32(r.nr), 0, 1), stmt(bpfRet, seccompRetErrno|uint32(r.errno)))
	}
	for _, nr := range allowedSyscalls {
		prog = append(prog, jump(bpfJeq, uint32(nr), 0, 1), allow)
	}
	return append(prog, stmt(bpfRet, seccompDefaultAction))
}

// installSeccomp applies the filter to every thread of the process.
// no_new_privs must already be set.
func installSeccomp() error {
	prog := seccompProgram()
	fprog := unix.SockFprog{Len: uint16(len(prog)), Filter: &prog[0]}
	r, _, errno := unix.Syscall(unix.SYS_SECCOMP, unix.SECCOMP_SET_MODE_FILTER,
		unix.SECCOMP_FILTER_FLAG_TSYNC, uintptr(unsafe.Pointer(&fprog)))
	if errno != 0 {
		return fmt.Errorf("seccomp: %w", errno)
	}
	if r != 0 {
		return fmt.Errorf("seccomp: thread %d could not be synchronized", r)
	}
	return nil
}
