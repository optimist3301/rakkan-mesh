//go:build !linux

package sandbox

import (
	"errors"
	"os"
)

func newSharedFile(int64, bool) (*os.File, error) {
	return nil, errors.New("shared RandomX dataset needs Linux")
}

func mapShared(int, int64, int64, bool) ([]byte, error) {
	return nil, errors.New("shared RandomX dataset needs Linux")
}

func fdSize(int) (int64, bool) { return 0, false }
