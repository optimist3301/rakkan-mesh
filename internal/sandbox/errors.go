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
)
