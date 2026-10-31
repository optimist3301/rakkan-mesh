package num

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

var useAVX2 = cpu.X86.HasAVX2

//go:noescape
func dot32AVX2(a, b *float32, n int, s *[32]float32)

//go:noescape
func dotBF16AVX2(w *uint16, x *float32, n int, s *[32]float32)

func dot32(a, b *float32, n int, s *[32]float32) {
	if useAVX2 {
		dot32AVX2(a, b, n, s)
		return
	}
	dot32Generic(unsafe.Slice(a, n), unsafe.Slice(b, n), s)
}

func dotBF16(w *uint16, x *float32, n int, s *[32]float32) {
	if useAVX2 {
		dotBF16AVX2(w, x, n, s)
		return
	}
	dotBF16Generic(unsafe.Slice(w, n), unsafe.Slice(x, n), s)
}
