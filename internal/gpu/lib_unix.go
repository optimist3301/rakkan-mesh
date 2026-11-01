//go:build linux || darwin

package gpu

import (
	"runtime"

	"github.com/ebitengine/purego"
)

func openLibrary() (uintptr, error) {
	names := []string{"libvulkan.so.1", "libvulkan.so"}
	if runtime.GOOS == "darwin" {
		// MoltenVK, directly or behind the loader of the Vulkan SDK.
		names = []string{"libvulkan.1.dylib", "libvulkan.dylib", "libMoltenVK.dylib"}
	}
	var err error
	for _, n := range names {
		var h uintptr
		if h, err = purego.Dlopen(n, purego.RTLD_NOW|purego.RTLD_GLOBAL); err == nil {
			return h, nil
		}
	}
	return 0, err
}

func registerFunc(fn any, lib uintptr, name string) { purego.RegisterLibFunc(fn, lib, name) }
