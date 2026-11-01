//go:build !linux && !darwin && !windows

package gpu

import "errors"

func openLibrary() (uintptr, error) { return 0, errors.New("no Vulkan on this platform") }

func registerFunc(any, uintptr, string) {}
