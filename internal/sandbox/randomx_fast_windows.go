package sandbox

import (
	"errors"
	"os"
	"strconv"
	"strings"
	"unsafe"

	"golang.org/x/sys/windows"
)

// The shared dataset on Windows is a pagefile-backed section. The node
// creates it, maps it read-write and passes its handle to every worker
// (isolate.inheritFiles); the workers map it read-only.

func newSharedFile(size int64, huge bool) (*os.File, error) {
	if huge {
		return nil, errors.New("huge pages are not used on Windows")
	}
	h, err := windows.CreateFileMapping(windows.InvalidHandle, nil, windows.PAGE_READWRITE, uint32(size>>32), uint32(size), nil)
	if err != nil {
		return nil, err
	}
	return os.NewFile(uintptr(h), "rakkan-randomx-dataset"), nil
}

func mapShared(fd int, off, length int64, write bool) ([]byte, error) {
	access := uint32(windows.FILE_MAP_READ)
	if write {
		access = windows.FILE_MAP_WRITE
	}
	addr, err := windows.MapViewOfFile(windows.Handle(fd), access, uint32(off>>32), uint32(off), uintptr(length))
	if err != nil {
		return nil, err
	}
	// addr is memory the OS mapped, not the Go heap.
	return unsafe.Slice(*(**byte)(unsafe.Pointer(&addr)), length), nil
}

func unmapShared(b []byte) {
	if len(b) > 0 {
		windows.UnmapViewOfFile(uintptr(unsafe.Pointer(&b[0])))
	}
}

// inheritedShared is the handle of the dataset the node passed, if any.
func inheritedShared() (int, bool) {
	first, _, _ := strings.Cut(os.Getenv(InheritedHandlesEnv), ",")
	n, err := strconv.ParseUint(first, 10, 64)
	return int(n), err == nil && n != 0
}
