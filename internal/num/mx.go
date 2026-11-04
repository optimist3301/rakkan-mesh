package num

import (
	"math"
	"unsafe"
)

// Block-scaled ("microscaling") arithmetic of DeepSeek-V4.1 checkpoints, as
// the reference's kernels compute it (docs/dev/deepseek-v41.md, section 4).
//
// Formats:
//   - E4M3 ("float8_e4m3fn"): sign, 4 exponent bits (bias 7), 3 mantissa
//     bits; subnormals down to 2^-9, largest 448, no infinities, 0x7f and
//     0xff are NaN.
//   - E2M1 (FP4): sign, 2 exponent bits, 1 mantissa bit; magnitudes 0, 0.5,
//     1, 1.5, 2, 3, 4, 6. Packed two to a byte, the even element in the low
//     nibble.
//   - E8M0: an exponent alone, byte e meaning 2^(e-127); 255 is NaN.
//
// Weights: E4M3 with one E8M0 scale per 32x32 block, or E2M1 with one E8M0
// scale per 32 consecutive weights of a row. Before a product the input is
// quantized to E4M3 with one power-of-two scale per 32 values (ActFP8).
//
// Every E4M3 value is a multiple of 2^-9 below 2^9 and every E2M1 value a
// multiple of 2^-1, so the 32 products of a block, and any partial sum of
// them, are exact in float64 (41 significant bits at most). A block's dot
// product therefore does not depend on how it is summed, nor on fused
// multiply-adds: any implementation gives the same bits. It is rounded to
// float32 once, scaled, and the blocks are added in order in float32, as
// the kernels add them; the result is rounded to bfloat16.

var (
	e4m3Val [256]float32 // value of each E4M3 code (NaN codes: NaN)
	e4m3F64 [256]float64
	fp4Val  [16]float32 // value of each E2M1 code
	fp4Int  [16]int32   // twice the value of each E2M1 code, an integer
	// The kernels' constants: Python floats (doubles) rounded to float32.
	inv448   = f32div(1, 448)
	inv6     = f32div(1, 6)
	actFloor = f32of(1e-4)
)

func f32div(a, b float64) float32 { return float32(a / b) }
func f32of(v float64) float32     { return float32(v) }

var fp4Grid = [8]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6}

func init() {
	for c := range 256 {
		sign, e, m := c>>7, (c>>3)&15, c&7
		var v float64
		switch {
		case e == 15 && m == 7:
			v = math.NaN()
		case e == 0:
			v = math.Ldexp(float64(m), -9)
		default:
			v = math.Ldexp(float64(8+m), e-10)
		}
		if sign == 1 {
			v = -v
		}
		e4m3Val[c], e4m3F64[c] = float32(v), v
	}
	for c := range 16 {
		v := fp4Grid[c&7]
		if c&8 != 0 {
			v = -v
		}
		fp4Val[c], fp4Int[c] = v, int32(2*v)
	}
}

// RoundBF16 rounds a float32 to the nearest bfloat16, ties to even.
func RoundBF16(f float32) float32 { return BF16(ToBF16(f)) }

// ValidE4M3 reports whether w holds no NaN codes. Weights must be checked
// on import: the vector code reads the NaN codes as 480.
func ValidE4M3(w []byte) bool {
	for _, c := range w {
		if c&0x7f == 0x7f {
			return false
		}
	}
	return true
}

// DecodeE4M3 returns the value of an E4M3 code.
func DecodeE4M3(c uint8) float32 { return e4m3Val[c] }

// EncodeE4M3 rounds v to the nearest E4M3 value, ties to even, and returns
// its code. Magnitudes beyond 448 saturate (the kernels clamp first); NaN
// becomes 0x7f.
func EncodeE4M3(v float32) uint8 {
	if v != v {
		return 0x7f
	}
	var sign uint8
	if math.Signbit(float64(v)) {
		sign = 0x80
	}
	a := math.Abs(float64(v))
	if a < 1.0/64 { // subnormal range: multiples of 2^-9
		return sign | uint8(math.RoundToEven(a*512)) // 8 rounds up to the smallest normal, code 8
	}
	_, x := math.Frexp(a) // a = f·2^x, f in [0.5, 1)
	e := x - 1            // a in [2^e, 2^(e+1))
	n := math.RoundToEven(math.Ldexp(a, 3-e))
	if n == 16 {
		e, n = e+1, 8
	}
	code := (e+7)<<3 | (int(n) - 8)
	return sign | uint8(min(code, 0x7e))
}

// DecodeE8M0 returns the value of an E8M0 scale, 2^(c-127).
func DecodeE8M0(c uint8) float32 {
	switch c {
	case 255:
		return float32(math.NaN())
	case 0:
		return math.Float32frombits(1 << 22) // 2^-127, subnormal
	}
	return math.Float32frombits(uint32(c) << 23)
}

// DecodeE2M1 returns the value of a 4-bit E2M1 code.
func DecodeE2M1(c uint8) float32 { return fp4Val[c&15] }

// EncodeE2M1 rounds v, within ±6, to the nearest E2M1 value, ties to the
// even code, and returns its 4-bit code (the sign of v, zero included, is
// kept).
func EncodeE2M1(v float32) uint8 {
	a := float32(math.Abs(float64(v)))
	var code uint8
	for i := range 7 {
		mid := (fp4Grid[i] + fp4Grid[i+1]) / 2
		if a > mid || a == mid && (i+1)%2 == 0 {
			code = uint8(i + 1)
		}
	}
	if math.Signbit(float64(v)) {
		code |= 8
	}
	return code
}

// pow2Ceil is 2^ceil(log2(t)) for positive normal t, computed as the
// kernels do: the exponent field, plus one if any mantissa bit is set.
func pow2Ceil(t float32) float32 {
	b := math.Float32bits(t)
	e := int32(b>>23&0xff) - 127
	if b&0x7fffff != 0 {
		e++
	}
	return math.Float32frombits(uint32(e+127) << 23)
}

func absMax(x []float32) float32 {
	var m float32
	for _, v := range x {
		m = max(m, float32(math.Abs(float64(v))))
	}
	return m
}

// ActFP8 is a vector quantized as the reference quantizes the input of an
// FP8 or FP4 product (act_quant with power-of-two scales): blocks of
// QBlock values, each with a scale 2^ceil(log2(max(amax, 1e-4)/448)) and
// the values rounded to E4M3 after division by it.
type ActFP8 struct {
	// Vals are the E4M3 values, unscaled, as float64 for exact block sums.
	Vals   []float64
	Scales []float32 // one per block, a power of two
	// ints are the values times 2^9, integers below 2^18, for products with
	// FP4 weights; within each block the even elements come first, then
	// the odd ones, the order of a packed block's low and high nibbles.
	ints []int32
}

// QuantizeActFP8 quantizes x, a whole number of blocks, into a, reusing its
// storage.
func QuantizeActFP8(x []float32, a *ActFP8) {
	nb := len(x) / QBlock
	if len(x) != nb*QBlock {
		panic("num: activation length is not a multiple of 32")
	}
	if cap(a.Vals) < len(x) {
		a.Vals = make([]float64, len(x))
		a.ints = make([]int32, len(x))
		a.Scales = make([]float32, nb)
	}
	a.Vals, a.ints, a.Scales = a.Vals[:len(x)], a.ints[:len(x)], a.Scales[:nb]
	for b := range nb {
		blk := x[b*QBlock : (b+1)*QBlock]
		s := pow2Ceil(max(absMax(blk), actFloor) * inv448)
		a.Scales[b] = s
		for j, v := range blk {
			q := e4m3F64[EncodeE4M3(min(max(v/s, -448), 448))]
			a.Vals[b*QBlock+j] = q
			a.ints[b*QBlock+j/2+(j%2)*QBlock/2] = int32(q * 512)
		}
	}
}

// ActQuantInPlace replaces x by its quantized value, as act_quant with
// inplace=True does: each value becomes its E4M3 rounding times the block
// scale, in bfloat16 (exactly representable).
func ActQuantInPlace(x []float32) {
	var a ActFP8
	QuantizeActFP8(x, &a)
	for i, v := range a.Vals {
		x[i] = RoundBF16(float32(v) * a.Scales[i/QBlock])
	}
}

// FP4Scale selects the scales of FP4 quantization.
type FP4Scale int

const (
	// FP4ScaleE8M0: blocks of 32, power-of-two scales (indexer queries and keys).
	FP4ScaleE8M0 FP4Scale = iota
	// FP4ScaleE4M3: blocks of 16, scales rounded to E4M3 (compressed KV).
	FP4ScaleE4M3
)

// Block returns the number of values sharing a scale.
func (k FP4Scale) Block() int {
	if k == FP4ScaleE4M3 {
		return 16
	}
	return 32
}

// QuantizeFP4 quantizes x, a whole number of blocks, as fp4_act_quant does:
// packed holds len(x)/2 bytes of E2M1 codes, scales one byte per block
// (E8M0 or E4M3 code).
func QuantizeFP4(x []float32, kind FP4Scale, packed, scales []byte) {
	blk := kind.Block()
	for b := range len(x) / blk {
		vals := x[b*blk : (b+1)*blk]
		amax := absMax(vals)
		var s float32
		if kind == FP4ScaleE4M3 {
			amax = max(amax, 6*(1.0/512)) // an all-zero group keeps a nonzero scale
			scales[b] = EncodeE4M3(amax / 6)
			s = DecodeE4M3(scales[b])
		} else {
			amax = max(amax, float32(math.Ldexp(6, -126)))
			s = pow2Ceil(amax * inv6)
			scales[b] = uint8(math.Float32bits(s) >> 23)
		}
		for j, v := range vals {
			c := EncodeE2M1(min(max(v/s, -6), 6))
			i := b*blk + j
			if i%2 == 0 {
				packed[i/2] = c
			} else {
				packed[i/2] |= c << 4
			}
		}
	}
}

// DequantFP4 sets x from QuantizeFP4's output: each value its E2M1 code
// times its block's scale, exact in bfloat16.
func DequantFP4(x []float32, kind FP4Scale, packed, scales []byte) {
	blk := kind.Block()
	for i := range x {
		var s float32
		if kind == FP4ScaleE4M3 {
			s = DecodeE4M3(scales[i/blk])
		} else {
			s = DecodeE8M0(scales[i/blk])
		}
		x[i] = RoundBF16(fp4Val[packed[i/2]>>(4*(i%2))&15] * s)
	}
}

// FP4QuantInPlace replaces x by its FP4 quantization, as fp4_act_quant
// with inplace=True does.
func FP4QuantInPlace(x []float32, kind FP4Scale) {
	packed := make([]byte, len(x)/2)
	scales := make([]byte, len(x)/kind.Block())
	QuantizeFP4(x, kind, packed, scales)
	DequantFP4(x, kind, packed, scales)
}

// MatVecFP8 sets out[r] for each of the len(out) rows of an E4M3 weight
// matrix w (row-major, cols = len(x.Vals) columns) with E8M0 scales, one
// per 32x32 block, row-major [ceil(rows/32)][cols/32]:
//
//	out[r] = bf16( Σ_b float32(Σ_{j∈b} w[r,j]·x[j]) · xs[b] · ws[r/32, b] )
//
// the blocks added in order in float32.
func MatVecFP8(out []float32, w, scales []byte, x *ActFP8) {
	matVecFP8(out, w, scales, x, true)
}

// matVecFP8 is MatVecFP8, leaving the sums in float32 unless bf16.
func matVecFP8(out []float32, w, scales []byte, x *ActFP8, bf16 bool) {
	cols := len(x.Vals)
	nb := cols / QBlock
	rows(len(out), cols, func(lo, hi int) {
		sums := make([]float64, nb)
		for r := lo; r < hi; r++ {
			blockSumsFP8(w[r*cols:(r+1)*cols], x.Vals, sums)
			srow := scales[(r/QBlock)*nb : (r/QBlock+1)*nb]
			out[r] = combineBlocks(sums, x.Scales, srow, 1, bf16)
		}
	})
}

// MatVecFP8Unrounded is MatVecFP8 with the sums left in float32, not
// rounded to bfloat16: for checking other implementations, whose last
// bits the rounding would mostly hide.
func MatVecFP8Unrounded(out []float32, w, scales []byte, x *ActFP8) {
	matVecFP8(out, w, scales, x, false)
}

// MatVecFP4Unrounded is MatVecFP4 with the sums left in float32.
func MatVecFP4Unrounded(out []float32, w, scales []byte, x *ActFP8) {
	matVecFP4(out, w, scales, x, false)
}

// MatVecFP4 is MatVecFP8 for an E2M1 weight matrix, packed (rows ×
// cols/2 bytes), with one E8M0 scale per 32 weights of a row ([rows][cols/32]).
func MatVecFP4(out []float32, w, scales []byte, x *ActFP8) {
	matVecFP4(out, w, scales, x, true)
}

func matVecFP4(out []float32, w, scales []byte, x *ActFP8, bf16 bool) {
	cols := len(x.Vals)
	nb := cols / QBlock
	rows(len(out), cols, func(lo, hi int) {
		isums := make([]int32, nb)
		sums := make([]float64, nb)
		for r := lo; r < hi; r++ {
			blockSumsFP4(w[r*cols/2:(r+1)*cols/2], x.ints, isums)
			for b, v := range isums {
				sums[b] = float64(v) // exact
			}
			// The integers are 2·w times 2^9·x.
			out[r] = combineBlocks(sums, x.Scales, scales[r*nb:(r+1)*nb], 1.0/1024, bf16)
		}
	})
}

// combineBlocks adds the blocks' exact sums (times unit, a power of two) as the
// kernels do: each rounded to float32, times the activation scale, times
// the weight scale, added in order.
func combineBlocks(sums []float64, xs []float32, ws []byte, unit float64, bf16 bool) float32 {
	var acc float32
	for b, d := range sums {
		t := float32(float32(d*unit) * xs[b])
		t = float32(t * DecodeE8M0(ws[b]))
		acc = float32(acc + t)
	}
	// Processors make NaNs of different bits (x86 0xffc00000, arm64 and
	// NVIDIA's GPUs others): one for all, so that every node gives the same.
	acc = Canonical(acc)
	if bf16 {
		acc = RoundBF16(acc)
	}
	return acc
}

// blockSumsFP8 sets sums[b] to the exact dot product of block b of an E4M3
// row and of x.
func blockSumsFP8(w []byte, x []float64, sums []float64) {
	if useFMA && len(sums) > 0 {
		blockSumsFP8AVX2(unsafe.SliceData(w), unsafe.SliceData(x), len(sums), unsafe.SliceData(sums))
		return
	}
	for b := range sums {
		sums[b] = blockDotFP8(w[b*QBlock:(b+1)*QBlock], x[b*QBlock:(b+1)*QBlock])
	}
}

// blockSumsFP4 sets sums[b] to the integer dot product of block b of a
// packed E2M1 row (codes as twice their values) and of x (ActFP8.ints).
func blockSumsFP4(w []byte, x []int32, sums []int32) {
	done := 0
	if useFMA {
		if done = len(sums) &^ 3; done > 0 {
			blockSumsFP4AVX2(unsafe.SliceData(w), unsafe.SliceData(x), done, unsafe.SliceData(sums))
		}
	}
	for b := done; b < len(sums); b++ {
		w, x := w[b*QBlock/2:(b+1)*QBlock/2], x[b*QBlock:(b+1)*QBlock]
		var s int32
		for j, c := range w {
			s += fp4Int[c&15]*x[j] + fp4Int[c>>4]*x[QBlock/2+j]
		}
		sums[b] = s
	}
}

// blockDotFP8 is the exact dot product of 32 E4M3 codes and values.
func blockDotFP8(w []byte, x []float64) float64 {
	w, x = w[:QBlock], x[:QBlock]
	var s0, s1, s2, s3 float64
	for j := 0; j < QBlock; j += 4 {
		s0 += e4m3F64[w[j]] * x[j]
		s1 += e4m3F64[w[j+1]] * x[j+1]
		s2 += e4m3F64[w[j+2]] * x[j+2]
		s3 += e4m3F64[w[j+3]] * x[j+3]
	}
	return (s0 + s1) + (s2 + s3)
}
