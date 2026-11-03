package sandbox

import "errors"

var (
	ErrLimitPolicy     = errors.New("requested limits exceed node policy")
	ErrModuleTooLarge  = errors.New("module too large")
	ErrInputTooLarge   = errors.New("input too large")
	ErrInvalidModule   = errors.New("invalid module")
	ErrForbiddenImport = errors.New("forbidden import")
	ErrMissingEntry    = errors.New("missing or malformed entry point")
	ErrOutputLimit     = errors.New("output limit exceeded")
	ErrMemoryAccess    = errors.New("out of bounds memory access in host call")
	ErrOutOfGas        = errors.New("out of gas")
	ErrTimeout         = errors.New("task timed out")
	ErrTrap            = errors.New("task trapped")
	// ErrBlobUnavailable means an attached blob could not be read, for
	// example because no provider could be reached: not the task's fault.
	ErrBlobUnavailable = errors.New("attached blob unavailable")
)

// gasError is a task error raised once the task was running, carrying the
// gas it had consumed by then. Like a result's GasUsed, it is deterministic
// for traps and host call faults; for interruptions it is where the task
// happened to stop.
type gasError struct {
	err error
	gas int64
}

func (e *gasError) Error() string      { return e.err.Error() }
func (e *gasError) Unwrap() error      { return e.err }
func (e *gasError) TaskGasUsed() int64 { return e.gas }

// WithGasUsed attaches to err the gas a task consumed before failing with
// it. A nil err stays nil.
func WithGasUsed(err error, gas int64) error {
	if err == nil {
		return nil
	}
	return &gasError{err: err, gas: gas}
}

// GasUsed returns the gas a failed task consumed, when known: for errors
// raised after the task started running. Errors before that, such as an
// invalid module, report false.
func GasUsed(err error) (int64, bool) {
	var g interface{ TaskGasUsed() int64 }
	if errors.As(err, &g) {
		return g.TaskGasUsed(), true
	}
	return 0, false
}
