package num

import (
	"math"
	"unsafe"
)

// Quantized weights, in blocks of QBlock consecutive weights of a row, each
// block with its own float32 scale d: Q8_0 stores q in [-127, 127], Q4_0
// stores q in [-8, 7] packed two to a byte (byte j of a block holds weight
// j in its low nibble and weight j+16 in its high nibble, offset by 8);
// the weight is q·d.
//
// A product with a quantized matrix quantizes the input vector to Q8_0 as
// well, so each block's dot product is an exact integer: the same on every
// processor whatever instructions compute it. Only the combination is
// floating point, in a fixed order: block b of a row contributes
// float32(sum_b) · float32(dw_b · dx_b) to running sum b mod 8, and the
// eight sums combine as ((0+1)+(2+3))+((4+5)+(6+7)).

// QBlock is the number of weights sharing a scale.
const QBlock = 32

// QuantizeQ8 quantizes weights, a whole number of blocks, to Q8_0.
func QuantizeQ8(w []float32) (q []int8, scales []float32) {
	q = make([]int8, len(w))
	scales = make([]float32, len(w)/QBlock)
	quantize8(w, q, scales)
	return q, scales
}

// quantize8 rounds each block to the nearest multiple of amax/127.
// Rounding to an integer is exact, as is |v|.
func quantize8(w []float32, q []int8, scales []float32) {
	for b := range scales {
		blk := w[b*QBlock : (b+1)*QBlock]
		var amax float32
		for _, v := range blk {
			amax = max(amax, float32(math.Abs(float64(v))))
		}
		// No division: GPUs quantize inputs too and must match (recip.go).
		d := amax * float32(1.0/127)
		scales[b] = d
		if d == 0 || d != d {
			clear(q[b*QBlock : (b+1)*QBlock])
			continue
		}
		id := Recip32(d)
		for j, v := range blk {
			q[b*QBlock+j] = int8(max(-127, min(127, math.RoundToEven(float64(float32(v*id))))))
		}
	}
}

// QuantizeQ4 quantizes weights, a whole number of blocks, to Q4_0. Each
// block's scale minimizes the squared error: starting from scales that put
// the largest weight at -8 or at ±7 and a few in between, the quantized
// values and the least-squares scale for them are alternated, and the best
// result kept. Every product is rounded explicitly, so the same weights
// quantize identically everywhere and a model imported on any node gets
// the same ID.
func QuantizeQ4(w []float32) (q []byte, scales []float32) {
	q = make([]byte, len(w)/2)
	scales = make([]float32, len(w)/QBlock)
	// Blocks are independent: split them among goroutines.
	rows(len(scales), QBlock*256, func(lo, hi int) { quantize4(w, q, scales, lo, hi) })
	return q, scales
}

func quantize4(w []float32, q []byte, scales []float32, lo, hi int) {
	var vals, best [QBlock]int8
	for b := lo; b < hi; b++ {
		blk := w[b*QBlock : (b+1)*QBlock]
		var amax, peak float32
		for _, v := range blk {
			if a := float32(math.Abs(float64(v))); a > amax {
				amax, peak = a, v
			}
		}
		bestErr, bestD := math.Inf(1), float32(0)
		if amax > 0 && amax == amax {
			for _, div := range []float32{-8, -7.5, -7, 7, 7.5, 6.5} {
				d := peak / div
				for range 3 {
					quant4(blk, d, &vals)
					// The least-squares scale for these values.
					var wq, qq float64
					for j, v := range vals {
						wq += float64(float64(blk[j]) * float64(v))
						qq += float64(float64(v) * float64(v))
					}
					if qq == 0 {
						break
					}
					d = float32(wq / qq)
				}
				quant4(blk, d, &vals)
				var e float64
				for j, v := range vals {
					diff := float64(blk[j]) - float64(float32(float32(v)*d))
					e += float64(diff * diff)
				}
				if e < bestErr {
					bestErr, bestD, best = e, d, vals
				}
			}
		} else {
			best = [QBlock]int8{}
		}
		scales[b] = bestD
		out := q[b*QBlock/2 : (b+1)*QBlock/2]
		for j := range QBlock / 2 {
			out[j] = byte(best[j]+8) | byte(best[j+16]+8)<<4
		}
	}
}

// quant4 rounds a block to multiples of d within [-8, 7].
func quant4(blk []float32, d float32, vals *[QBlock]int8) {
	var id float32
	if d != 0 {
		id = 1 / d
	}
	for j, v := range blk {
		vals[j] = int8(max(-8, min(7, math.RoundToEven(float64(float32(v*id))))))
	}
}

// DequantQ8 expands blocks of Q8_0 into dst.
func DequantQ8(dst []float32, q []int8, scales []float32) {
	for i := range dst {
		dst[i] = float32(float32(q[i]) * scales[i/QBlock])
	}
}

// DequantQ4 expands blocks of Q4_0 into dst.
func DequantQ4(dst []float32, q []byte, scales []float32) {
	for b := range len(dst) / QBlock {
		d := scales[b]
		for j := range QBlock / 2 {
			v := q[b*QBlock/2+j]
			dst[b*QBlock+j] = float32(float32(int(v&15)-8) * d)
			dst[b*QBlock+j+16] = float32(float32(int(v>>4)-8) * d)
		}
	}
}

// QuantizeActs quantizes an input vector, a whole number of blocks, to
// Q8_0 as the quantized products do.
func QuantizeActs(x []float32) (q []int8, scales []float32) {
	q = make([]int8, len(x))
	scales = make([]float32, len(x)/QBlock)
	quantize8(x, q, scales)
	return q, scales
}

// qrow combines a row's integer block sums with the scales.
func qrow(sums []int32, ws, xs []float32) float32 {
	var s [8]float32
	for b, si := range sums {
		s[b&7] += float32(float32(si) * float32(ws[b]*xs[b]))
	}
	return ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]))
}

// MatVecQ8 sets out = w·x for a Q8_0 matrix of len(out) rows of len(x).
func MatVecQ8(out []float32, w []int8, ws []float32, x []float32) {
	n := len(x)
	nb := n / QBlock
	qx := make([]int8, n)
	xs := make([]float32, nb)
	quantize8(x, qx, xs)
	rows(len(out), n, func(lo, hi int) {
		sums := make([]int32, nb)
		for i := lo; i < hi; i++ {
			blockSumsQ8(w[i*n:(i+1)*n], qx, sums)
			out[i] = qrow(sums, ws[i*nb:(i+1)*nb], xs)
		}
	})
}

// MatVecQ4 sets out = w·x for a Q4_0 matrix of len(out) rows of len(x).
func MatVecQ4(out []float32, w []byte, ws []float32, x []float32) {
	n := len(x)
	nb := n / QBlock
	qx := make([]int8, n)
	xs := make([]float32, nb)
	quantize8(x, qx, xs)
	rows(len(out), n, func(lo, hi int) {
		sums := make([]int32, nb)
		for i := lo; i < hi; i++ {
			blockSumsQ4(w[i*n/2:(i+1)*n/2], qx, sums)
			out[i] = qrow(sums, ws[i*nb:(i+1)*nb], xs)
		}
	})
}

func blockSumsQ8Generic(w, x []int8, sums []int32) {
	for b := range sums {
		w, x := w[b*QBlock:(b+1)*QBlock:(b+1)*QBlock], x[b*QBlock:(b+1)*QBlock:(b+1)*QBlock]
		var s int32
		for j := range QBlock {
			s += int32(w[j]) * int32(x[j])
		}
		sums[b] = s
	}
}

func blockSumsQ4Generic(w []byte, x []int8, sums []int32) {
	for b := range sums {
		w, x := w[b*QBlock/2:(b+1)*QBlock/2:(b+1)*QBlock/2], x[b*QBlock:(b+1)*QBlock:(b+1)*QBlock]
		var s int32
		for j := range QBlock / 2 {
			s += (int32(w[j]&15)-8)*int32(x[j]) + (int32(w[j]>>4)-8)*int32(x[j+16])
		}
		sums[b] = s
	}
}

// The vector code takes blocks four at a time; the rest are portable.

func blockSumsQ8(w, x []int8, sums []int32) {
	done := 0
	if useAVX2 {
		if done = len(sums) &^ 3; done > 0 {
			blockSumsQ8AVX2(unsafe.SliceData(w), unsafe.SliceData(x), done, unsafe.SliceData(sums))
		}
	}
	blockSumsQ8Generic(w[done*QBlock:], x[done*QBlock:], sums[done:])
}

func blockSumsQ4(w []byte, x []int8, sums []int32) {
	done := 0
	if useAVX2 {
		if done = len(sums) &^ 3; done > 0 {
			blockSumsQ4AVX2(unsafe.SliceData(w), unsafe.SliceData(x), done, unsafe.SliceData(sums))
		}
	}
	blockSumsQ4Generic(w[done*QBlock/2:], x[done*QBlock:], sums[done:])
}
