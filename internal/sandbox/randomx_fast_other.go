//go:build !linux && !windows

package sandbox

import (
	"errors"
	"os"
)

func newSharedFile(int64, bool) (*os.File, error) {
	return nil, errors.New("shared RandomX dataset needs Linux or Windows")
}

func mapShared(int, int64, int64, bool) ([]byte, error) {
	return nil, errors.New("shared RandomX dataset needs Linux or Windows")
}

func inheritedShared() (int, bool) { return 0, false }

func unmapShared([]byte) {}
