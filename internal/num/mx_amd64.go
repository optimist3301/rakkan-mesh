package num

import "golang.org/x/sys/cpu"

// The FP8 code multiplies with VFMADD231PD: its products are exact, so a
// fused multiply-add rounds nothing the portable code would not.
var useFMA = cpu.X86.HasAVX2 && cpu.X86.HasFMA

//go:noescape
func blockSumsFP4AVX2(w *byte, x *int32, nb int, sums *int32)

//go:noescape
func blockSumsFP8AVX2(w *byte, x *float64, nb int, sums *float64)
