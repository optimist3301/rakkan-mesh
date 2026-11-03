//go:build !amd64

package num

var useFMA = false

func blockSumsFP4AVX2(w *byte, x *int32, nb int, sums *int32) { panic("no AVX2") }

func blockSumsFP8AVX2(w *byte, x *float64, nb int, sums *float64) { panic("no AVX2") }
