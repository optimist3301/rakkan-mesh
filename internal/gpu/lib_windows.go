//go:build windows

package gpu

import (
	"syscall"

	"github.com/ebitengine/purego"
)

func openLibrary() (uintptr, error) {
	h, err := syscall.LoadLibrary("vulkan-1.dll")
	return uintptr(h), err
}

func registerFunc(fn any, lib uintptr, name string) { purego.RegisterLibFunc(fn, lib, name) }
