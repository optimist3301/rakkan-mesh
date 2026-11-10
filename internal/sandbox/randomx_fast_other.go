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

// releasePages: a section cannot be emptied while workers hold views of it,
// so on this platform the dataset stays once built.
func releasePages(*os.File, int64, int64) error { return errors.New("not supported on this platform") }
