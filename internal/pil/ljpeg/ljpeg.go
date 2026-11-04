// Package ljpeg decodes JPEG images to the samples libjpeg-turbo (3.2, as
// Pillow 13 uses it) gives, to the bit: Go's image/jpeg reads the file (its
// markers, Huffman decoding, progressive scans, restarts), and libjpeg-turbo's
// arithmetic, ported, makes the samples: the accurate integer inverse DCT
// (jidctint.c), the "fancy" triangle-filter upsampling of chroma
// (jdsample.c) with its context rows (jdmainct.c), and the YCbCr to RGB
// tables (jdcolor.c). libjpeg-turbo's SIMD code computes the same bits as
// its C, which is what is ported.
//
// This software is based in part on the work of the Independent JPEG
// Group: the ported parts follow libjpeg-turbo, under its IJG and BSD
// licenses (NOTICE). The rest is Go's, under Go's license (LICENSE-GO).
package ljpeg

import (
	"bytes"
	"errors"
)

const blockSize = 8 * 8

type block [blockSize]int32

// Image is a decoded image: W×H pixels of Mode's channels, row by row.
// Mode is "L" (grey), "RGB", or "CMYK" (libjpeg's, not inverted).
type Image struct {
	W, H int
	Mode string
	Pix  []byte
}

// MaxPixels bounds the images Decode accepts.
const MaxPixels = 64 << 20

// Decode decodes a baseline or progressive JPEG as Pillow's
// Image.open(...) does: libjpeg-turbo's default decompression (islow
// IDCT, fancy upsampling), grey to "L", three components to "RGB" (from
// YCbCr or RGB, which libjpeg tells apart by JFIF, Adobe and component
// identifiers), four to "CMYK" (from CMYK or YCCK).
func Decode(data []byte) (*Image, error) {
	d := &decoder{}
	if _, err := d.decode(bytes.NewReader(data), true); err != nil && d.nComp == 0 {
		return nil, err
	}
	if d.width <= 0 || d.height <= 0 || int64(d.width)*int64(d.height) > MaxPixels {
		return nil, FormatError("image size")
	}
	d = &decoder{}
	if _, err := d.decode(bytes.NewReader(data), false); err != nil {
		return nil, err
	}
	if d.progressive {
		if err := d.checkComplete(); err != nil {
			return nil, err
		}
	}
	return d.output()
}

// checkComplete refuses a progressive image whose first coefficients were
// not all sent to their last bit: libjpeg-turbo smooths its blocks
// (jdcoefct.c, smoothing_ok), which this does not.
func (d *decoder) checkComplete() error {
	for i := range d.nComp {
		for _, b := range d.coefBits[i] {
			if b != 0 {
				return UnsupportedError("progressive image with coefficients left out")
			}
		}
	}
	return nil
}

// idctIslow is jpeg_idct_islow: coefficients (natural order) times the
// quantization table (zig-zag order) to 8×8 samples at dst.
func idctIslow(coef *[blockSize]int16, qt *block, dst []byte, stride int) {
	const (
		constBits = 13
		pass1Bits = 2
		fix0298   = 2446
		fix0390   = 3196
		fix0541   = 4433
		fix0765   = 6270
		fix0899   = 7373
		fix1175   = 9633
		fix1501   = 12299
		fix1847   = 15137
		fix1961   = 16069
		fix2053   = 16819
		fix2562   = 20995
		fix3072   = 25172
	)
	var q [blockSize]int64 // the table in natural order
	for zig, v := range qt {
		q[unzig[zig]] = int64(v)
	}
	deq := func(i int) int64 { return int64(coef[i]) * q[i] }
	descale := func(x int64, n uint) int64 { return (x + 1<<(n-1)) >> n }
	var ws [blockSize]int32
	for c := range 8 {
		if coef[8+c] == 0 && coef[16+c] == 0 && coef[24+c] == 0 && coef[32+c] == 0 &&
			coef[40+c] == 0 && coef[48+c] == 0 && coef[56+c] == 0 {
			dc := int32(deq(c)) << pass1Bits
			for r := range 8 {
				ws[8*r+c] = dc
			}
			continue
		}
		z2, z3 := deq(16+c), deq(48+c)
		z1 := (z2 + z3) * fix0541
		tmp2 := z1 + z3*-fix1847
		tmp3 := z1 + z2*fix0765
		z2, z3 = deq(c), deq(32+c)
		tmp0 := (z2 + z3) << constBits
		tmp1 := (z2 - z3) << constBits
		tmp10, tmp13 := tmp0+tmp3, tmp0-tmp3
		tmp11, tmp12 := tmp1+tmp2, tmp1-tmp2
		tmp0, tmp1, tmp2, tmp3 = deq(56+c), deq(40+c), deq(24+c), deq(8+c)
		o0, o1, o2, o3 := odd(tmp0, tmp1, tmp2, tmp3)
		ws[c] = int32(descale(tmp10+o3, constBits-pass1Bits))
		ws[56+c] = int32(descale(tmp10-o3, constBits-pass1Bits))
		ws[8+c] = int32(descale(tmp11+o2, constBits-pass1Bits))
		ws[48+c] = int32(descale(tmp11-o2, constBits-pass1Bits))
		ws[16+c] = int32(descale(tmp12+o1, constBits-pass1Bits))
		ws[40+c] = int32(descale(tmp12-o1, constBits-pass1Bits))
		ws[24+c] = int32(descale(tmp13+o0, constBits-pass1Bits))
		ws[32+c] = int32(descale(tmp13-o0, constBits-pass1Bits))
	}
	for r := range 8 {
		w := ws[8*r : 8*r+8]
		out := dst[r*stride : r*stride+8]
		if w[1] == 0 && w[2] == 0 && w[3] == 0 && w[4] == 0 && w[5] == 0 && w[6] == 0 && w[7] == 0 {
			v := rangeLimit(descale(int64(w[0]), pass1Bits+3))
			for i := range out {
				out[i] = v
			}
			continue
		}
		z2, z3 := int64(w[2]), int64(w[6])
		z1 := (z2 + z3) * fix0541
		tmp2 := z1 + z3*-fix1847
		tmp3 := z1 + z2*fix0765
		tmp0 := (int64(w[0]) + int64(w[4])) << constBits
		tmp1 := (int64(w[0]) - int64(w[4])) << constBits
		tmp10, tmp13 := tmp0+tmp3, tmp0-tmp3
		tmp11, tmp12 := tmp1+tmp2, tmp1-tmp2
		o0, o1, o2, o3 := odd(int64(w[7]), int64(w[5]), int64(w[3]), int64(w[1]))
		const n = constBits + pass1Bits + 3
		out[0] = rangeLimit(descale(tmp10+o3, n))
		out[7] = rangeLimit(descale(tmp10-o3, n))
		out[1] = rangeLimit(descale(tmp11+o2, n))
		out[6] = rangeLimit(descale(tmp11-o2, n))
		out[2] = rangeLimit(descale(tmp12+o1, n))
		out[5] = rangeLimit(descale(tmp12-o1, n))
		out[3] = rangeLimit(descale(tmp13+o0, n))
		out[4] = rangeLimit(descale(tmp13-o0, n))
	}
}

// odd is the islow IDCT's odd part: y7, y5, y3, y1 in, tmp0..tmp3 out.
func odd(tmp0, tmp1, tmp2, tmp3 int64) (int64, int64, int64, int64) {
	z1, z2, z3, z4 := tmp0+tmp3, tmp1+tmp2, tmp0+tmp2, tmp1+tmp3
	z5 := (z3 + z4) * 9633
	tmp0 *= 2446
	tmp1 *= 16819
	tmp2 *= 25172
	tmp3 *= 12299
	z1 *= -7373
	z2 *= -20995
	z3 *= -16069
	z4 *= -3196
	z3 += z5
	z4 += z5
	tmp0 += z1 + z3
	tmp1 += z2 + z4
	tmp2 += z2 + z3
	tmp3 += z1 + z4
	return tmp0, tmp1, tmp2, tmp3
}

// rangeLimit is the IDCT's range limit table (jdmaster.c): the sample plus
// 128 clamped to 0..255, the int's low 10 bits only, as a signed value.
func rangeLimit(x int64) byte {
	v := int32(x) & 1023
	if v >= 512 {
		v -= 1024
	}
	v += 128
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return byte(v)
}

func clamp(v int) byte {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return byte(v)
}

// output upsamples the components to the image's size and converts them,
// as jdsample.c and jdcolor.c do.
func (d *decoder) output() (*Image, error) {
	w, h := d.width, d.height
	maxH, maxV := d.comp[0].h, d.comp[0].v // the reader keeps component 0 largest
	full := make([][]byte, d.nComp)
	for i := range d.nComp {
		c := d.comp[i]
		full[i] = upsample(d.planes[i], d.strides[i], ceilDiv(w*c.h, maxH), ceilDiv(h*c.v, maxV), maxH/c.h, maxV/c.v, w, h)
	}
	img := &Image{W: w, H: h}
	switch d.nComp {
	case 1:
		img.Mode, img.Pix = "L", full[0]
		return img, nil
	case 3:
		img.Mode, img.Pix = "RGB", make([]byte, 3*w*h)
		ycc := d.colorSpace3() == "YCbCr"
		for i := range w * h {
			y, cb, cr := full[0][i], full[1][i], full[2][i]
			if ycc {
				img.Pix[3*i], img.Pix[3*i+1], img.Pix[3*i+2] = yccRGB(int(y), int(cb), int(cr))
			} else {
				img.Pix[3*i], img.Pix[3*i+1], img.Pix[3*i+2] = y, cb, cr
			}
		}
		return img, nil
	case 4:
		img.Mode, img.Pix = "CMYK", make([]byte, 4*w*h)
		ycck := d.adobeTransformValid && d.adobeTransform != 0
		for i := range w * h {
			a, b, c, k := full[0][i], full[1][i], full[2][i], full[3][i]
			if ycck {
				r, g, bl := yccRGB(int(a), int(b), int(c))
				a, b, c = 255-r, 255-g, 255-bl // ycck_cmyk_convert
			}
			img.Pix[4*i], img.Pix[4*i+1], img.Pix[4*i+2], img.Pix[4*i+3] = a, b, c, k
		}
		return img, nil
	}
	return nil, errors.New("ljpeg: number of components")
}

// colorSpace3 is libjpeg's guess for three components
// (jdapimin.c, default_decompress_parms).
func (d *decoder) colorSpace3() string {
	switch {
	case d.jfif:
		return "YCbCr"
	case d.adobeTransformValid:
		if d.adobeTransform == 0 {
			return "RGB"
		}
		return "YCbCr"
	case d.comp[0].c == 'R' && d.comp[1].c == 'G' && d.comp[2].c == 'B':
		return "RGB"
	}
	return "YCbCr"
}

func ceilDiv(a, b int) int { return (a + b - 1) / b }

// The YCbCr to RGB tables (build_ycc_rgb_table), 16-bit fixed point.
var crR, cbB [256]int
var crG, cbG [256]int64

func init() {
	const scaleBits = 16
	fix := func(x float64) int64 { return int64(x*(1<<scaleBits) + 0.5) }
	for i := range 256 {
		x := int64(i - 128)
		crR[i] = int((fix(1.40200)*x + 1<<(scaleBits-1)) >> scaleBits)
		cbB[i] = int((fix(1.77200)*x + 1<<(scaleBits-1)) >> scaleBits)
		crG[i] = -fix(0.71414) * x
		cbG[i] = -fix(0.34414)*x + 1<<(scaleBits-1)
	}
}

// yccRGB is ycc_rgb_convert for one pixel.
func yccRGB(y, cb, cr int) (byte, byte, byte) {
	return clamp(y + crR[cr]), clamp(y + int((cbG[cb]+crG[cr])>>16)), clamp(y + cbB[cb])
}

// upsample brings a component's samples (dw×dh of them used, stride per
// row) to the image's w×h: by hx and vy, the methods jinit_upsampler
// chooses with fancy upsampling on.
func upsample(p []byte, stride, dw, dh, hx, vy, w, h int) []byte {
	out := make([]byte, w*h)
	// row is a component row as the main controller gives the upsampler:
	// above the first, the first; below the last real one, the last.
	row := func(r int) []byte {
		r = min(max(r, 0), dh-1)
		return p[r*stride : r*stride+stride]
	}
	switch {
	case hx == 1 && vy == 1:
		for y := range h {
			copy(out[y*w:(y+1)*w], row(y))
		}
	case hx == 2 && vy == 1 && dw > 2: // h2v1_fancy_upsample
		line := make([]byte, 2*dw)
		for y := range h {
			in := row(y)
			line[0] = in[0]
			line[1] = byte((int(in[0])*3 + int(in[1]) + 2) >> 2)
			for x := 1; x < dw-1; x++ {
				v := int(in[x]) * 3
				line[2*x] = byte((v + int(in[x-1]) + 1) >> 2)
				line[2*x+1] = byte((v + int(in[x+1]) + 2) >> 2)
			}
			v := int(in[dw-1])
			line[2*dw-2] = byte((v*3 + int(in[dw-2]) + 1) >> 2)
			line[2*dw-1] = byte(v)
			copy(out[y*w:(y+1)*w], line)
		}
	case hx == 1 && vy == 2: // h1v2_fancy_upsample
		for y := range h {
			in0 := row(y / 2)
			in1, bias := row(y/2-1), 1
			if y%2 == 1 {
				in1, bias = row(y/2+1), 2
			}
			for x := range w {
				out[y*w+x] = byte((int(in0[x])*3 + int(in1[x]) + bias) >> 2)
			}
		}
	case hx == 2 && vy == 2 && dw > 2: // h2v2_fancy_upsample
		line := make([]byte, 2*dw)
		for y := range h {
			in0 := row(y / 2)
			in1 := row(y/2 - 1)
			if y%2 == 1 {
				in1 = row(y/2 + 1)
			}
			col := func(x int) int { return int(in0[x])*3 + int(in1[x]) }
			this, next := col(0), col(1)
			line[0] = byte((this*4 + 8) >> 4)
			line[1] = byte((this*3 + next + 7) >> 4)
			last := this
			this = next
			for x := 1; x < dw-1; x++ {
				next = col(x + 1)
				line[2*x] = byte((this*3 + last + 8) >> 4)
				line[2*x+1] = byte((this*3 + next + 7) >> 4)
				last, this = this, next
			}
			line[2*dw-2] = byte((this*3 + last + 8) >> 4)
			line[2*dw-1] = byte((this*4 + 7) >> 4)
			copy(out[y*w:(y+1)*w], line)
		}
	default: // h2v1_upsample, h2v2_upsample, int_upsample: replication
		for y := range h {
			in := p[(y/vy)*stride:]
			for x := range w {
				out[y*w+x] = in[x/hx]
			}
		}
	}
	return out
}
