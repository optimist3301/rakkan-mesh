//go:build linux && !amd64 && !arm64

package isolate

// No seccomp filter is defined for this architecture; checkPlatform rejects
// configurations that request one.

const (
	auditArch = 0
	hasX32    = false
)

var (
	allowedSyscalls []uintptr
	errnoSyscalls   []errnoRule
)
