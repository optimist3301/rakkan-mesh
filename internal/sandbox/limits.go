package sandbox

import (
	"fmt"
	"time"
)

// PageSize is the size of one WebAssembly linear memory page.
const PageSize = 64 * 1024

// MaxPages is the most a 32-bit linear memory can address (4 GiB).
const MaxPages = 65536

// MaxGas keeps the signed gas counter far from overflow: a balance may dip
// below zero between checks, but by much less than this headroom.
const MaxGas = 1 << 60

// Limits bounds the resources a single task may consume.
type Limits struct {
	MaxModuleBytes int           // size of the encoded module
	MaxMemoryPages uint32        // linear memory, in 64 KiB pages
	MaxInputBytes  int           // task input visible through input_read
	MaxOutputBytes int           // total bytes accepted by output_write
	MaxLogBytes    int           // log bytes kept; the rest is dropped
	MaxGas         int64         // deterministic instruction budget, see package meter
	Timeout        time.Duration // wall clock, including instantiation; a backstop for gas
}

// DefaultLimits is a conservative profile for untrusted tasks.
func DefaultLimits() Limits {
	return Limits{
		MaxModuleBytes: 8 << 20,
		MaxMemoryPages: 1024, // 64 MiB
		MaxInputBytes:  16 << 20,
		MaxOutputBytes: 16 << 20,
		MaxLogBytes:    64 << 10,
		MaxGas:         5_000_000_000,
		Timeout:        10 * time.Second,
	}
}

// Within reports whether every limit in l is at most the corresponding limit
// in ceiling. Nodes use it to reject requests exceeding their local policy.
func (l Limits) Within(ceiling Limits) error {
	switch {
	case l.MaxModuleBytes <= 0 || l.MaxModuleBytes > ceiling.MaxModuleBytes:
		return fmt.Errorf("%w: module size %d (ceiling %d)", ErrLimitPolicy, l.MaxModuleBytes, ceiling.MaxModuleBytes)
	case l.MaxMemoryPages == 0 || l.MaxMemoryPages > ceiling.MaxMemoryPages || l.MaxMemoryPages > MaxPages:
		return fmt.Errorf("%w: memory pages %d (ceiling %d)", ErrLimitPolicy, l.MaxMemoryPages, min(ceiling.MaxMemoryPages, MaxPages))
	case l.MaxInputBytes < 0 || l.MaxInputBytes > ceiling.MaxInputBytes:
		return fmt.Errorf("%w: input size %d (ceiling %d)", ErrLimitPolicy, l.MaxInputBytes, ceiling.MaxInputBytes)
	case l.MaxOutputBytes < 0 || l.MaxOutputBytes > ceiling.MaxOutputBytes:
		return fmt.Errorf("%w: output size %d (ceiling %d)", ErrLimitPolicy, l.MaxOutputBytes, ceiling.MaxOutputBytes)
	case l.MaxLogBytes < 0 || l.MaxLogBytes > ceiling.MaxLogBytes:
		return fmt.Errorf("%w: log size %d (ceiling %d)", ErrLimitPolicy, l.MaxLogBytes, ceiling.MaxLogBytes)
	case l.MaxGas <= 0 || l.MaxGas > ceiling.MaxGas || l.MaxGas > MaxGas:
		return fmt.Errorf("%w: gas %d (ceiling %d)", ErrLimitPolicy, l.MaxGas, min(ceiling.MaxGas, MaxGas))
	case l.Timeout <= 0 || l.Timeout > ceiling.Timeout:
		return fmt.Errorf("%w: timeout %s (ceiling %s)", ErrLimitPolicy, l.Timeout, ceiling.Timeout)
	}
	return nil
}
