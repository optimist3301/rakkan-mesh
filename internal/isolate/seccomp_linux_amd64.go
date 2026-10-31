package isolate

import "golang.org/x/sys/unix"

const (
	auditArch = unix.AUDIT_ARCH_X86_64
	hasX32    = true
)

var allowedSyscalls = []uintptr{
	// I/O on inherited descriptors and the runtime's poller.
	unix.SYS_READ, unix.SYS_WRITE, unix.SYS_READV, unix.SYS_WRITEV,
	unix.SYS_PREAD64, unix.SYS_PWRITE64, unix.SYS_CLOSE, unix.SYS_FSTAT,
	unix.SYS_LSEEK, unix.SYS_FCNTL, unix.SYS_PIPE2, unix.SYS_EVENTFD2,
	unix.SYS_EPOLL_CREATE1, unix.SYS_EPOLL_CTL, unix.SYS_EPOLL_WAIT, unix.SYS_EPOLL_PWAIT,
	// Memory, including executable mappings for compiled WebAssembly.
	unix.SYS_MMAP, unix.SYS_MUNMAP, unix.SYS_MPROTECT, unix.SYS_MREMAP,
	unix.SYS_MADVISE, unix.SYS_MINCORE, unix.SYS_BRK,
	// Threads, scheduling and synchronisation.
	unix.SYS_FUTEX, unix.SYS_SCHED_YIELD, unix.SYS_SCHED_GETAFFINITY,
	unix.SYS_SET_ROBUST_LIST, unix.SYS_RSEQ, unix.SYS_ARCH_PRCTL,
	unix.SYS_GETPID, unix.SYS_GETTID, unix.SYS_TGKILL,
	unix.SYS_EXIT, unix.SYS_EXIT_GROUP,
	// Signals.
	unix.SYS_RT_SIGACTION, unix.SYS_RT_SIGPROCMASK, unix.SYS_RT_SIGRETURN,
	unix.SYS_SIGALTSTACK, unix.SYS_RESTART_SYSCALL,
	// Time.
	unix.SYS_CLOCK_GETTIME, unix.SYS_GETTIMEOFDAY, unix.SYS_NANOSLEEP, unix.SYS_CLOCK_NANOSLEEP,
	unix.SYS_GETRANDOM, unix.SYS_GETRLIMIT,
}

var errnoSyscalls = []errnoRule{
	{unix.SYS_OPEN, unix.EPERM},
	{unix.SYS_OPENAT, unix.EPERM},
	{unix.SYS_CLONE3, unix.ENOSYS},
}
