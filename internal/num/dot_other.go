//go:build !amd64

package num

import "unsafe"

const useAVX2 = false

func dot32(a, b *float32, n int, s *[32]float32) {
	dot32Generic(unsafe.Slice(a, n), unsafe.Slice(b, n), s)
}

func dotBF16(w *uint16, x *float32, n int, s *[32]float32) {
	dotBF16Generic(unsafe.Slice(w, n), unsafe.Slice(x, n), s)
}

func blockSumsQ8AVX2(w, x *int8, nb int, sums *int32) { panic("no AVX2") }

func blockSumsQ4AVX2(w *byte, x *int8, nb int, sums *int32) { panic("no AVX2") }
