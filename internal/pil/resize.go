package pil

import "math"

// Resize scales the image to w×h with Pillow's BICUBIC filter
// (Image.resize, Resample.c): two separable passes, the horizontal first
// unless the height shrinks by more than twice what the width does, each
// with coefficients per output pixel summed in 22-bit fixed point.
func (m *RGB) Resize(w, h int) *RGB {
	if w == m.W && h == m.H {
		return &RGB{W: w, H: h, Pix: append([]byte(nil), m.Pix...)}
	}
	needH, needV := w != m.W, h != m.H
	horizontalFirst := !(m.H-h > 0 && m.H-h > (m.W-w)*2)
	var vert, horiz *coeffs
	if (needH && horizontalFirst) || needV {
		vert = bicubicCoeffs(m.H, h)
	}
	in := m
	if horizontalFirst {
		if needH {
			first := vert.bounds[0]
			last := vert.bounds[2*h-2] + vert.bounds[2*h-1]
			for i := range h {
				vert.bounds[2*i] -= first
			}
			horiz = bicubicCoeffs(m.W, w)
			in = resampleH(in, w, last-first, first, horiz)
		}
		if needV {
			in = resampleV(in, h, vert)
		}
	} else {
		if needV {
			in = resampleV(in, h, vert)
		}
		if needH {
			in = resampleH(in, w, h, 0, bicubicCoeffs(m.W, w))
		}
	}
	return in
}

// precisionBits is Resample.c's PRECISION_BITS: 8 bits of result, two for
// the filter's overshoot and the sign.
const precisionBits = 32 - 8 - 2

// coeffs are a pass's: for output pixel i, k[i·size:][:bounds[2i+1]]
// weigh the input pixels from bounds[2i].
type coeffs struct {
	size   int
	bounds []int
	k      []int32
}

// bicubic is Resample.c's bicubic_filter (a = −0.5), its products unfused.
func bicubic(x float64) float64 {
	const a = -0.5
	if x < 0 {
		x = -x
	}
	if x < 1 {
		return float64(float64(float64(float64((a+2)*x)-(a+3))*x)*x) + 1
	}
	if x < 2 {
		return float64(float64(float64(float64(float64(x-5)*x)+8)*x)-4) * a
	}
	return 0
}

// bicubicCoeffs is precompute_coeffs and normalize_coeffs_8bpc for a pass
// from inSize to outSize over the whole input (box 0..inSize).
func bicubicCoeffs(inSize, outSize int) *coeffs {
	const support0 = 2.0
	scale := float64(inSize) / float64(outSize)
	filterscale := max(scale, 1)
	support := support0 * filterscale
	size := int(math.Ceil(support))*2 + 1
	c := &coeffs{size: size, bounds: make([]int, 2*outSize), k: make([]int32, size*outSize)}
	kk := make([]float64, size)
	inv := 1 / filterscale
	for xx := range outSize {
		center := 0 + float64((float64(xx)+0.5)*scale)
		xmin := max(int(center-support+0.5), 0)
		xmax := min(int(center+support+0.5), inSize) - xmin
		ww := 0.0
		for x := range xmax {
			w := bicubic((float64(x+xmin) - center + 0.5) * inv)
			kk[x] = w
			ww += w
		}
		for x := range xmax {
			if ww != 0 {
				kk[x] /= ww
			}
			// (int)(±0.5 + w·2²²): truncation toward zero.
			f := kk[x] * (1 << precisionBits)
			if kk[x] < 0 {
				c.k[xx*size+x] = int32(-0.5 + f)
			} else {
				c.k[xx*size+x] = int32(0.5 + f)
			}
		}
		c.bounds[2*xx], c.bounds[2*xx+1] = xmin, xmax
	}
	return c
}

// clip8 is Resample.c's: the sum shifted down, clamped to 0..255.
func clip8(s int32) byte {
	v := s >> precisionBits
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return byte(v)
}

// resampleH is _ImagingResampleHorizontal_8bpc: an out×rows image from the
// input's rows offset.., each of its pixels a weighted sum along the row.
func resampleH(in *RGB, out, rows, offset int, c *coeffs) *RGB {
	m := &RGB{W: out, H: rows, Pix: make([]byte, 3*out*rows)}
	for y := range rows {
		line := in.Pix[3*(y+offset)*in.W:]
		o := m.Pix[3*y*out:]
		for xx := range out {
			xmin, n := c.bounds[2*xx], c.bounds[2*xx+1]
			k := c.k[xx*c.size:]
			s0, s1, s2 := int32(1)<<(precisionBits-1), int32(1)<<(precisionBits-1), int32(1)<<(precisionBits-1)
			for x := range n {
				p := line[3*(x+xmin):]
				s0 += int32(p[0]) * k[x]
				s1 += int32(p[1]) * k[x]
				s2 += int32(p[2]) * k[x]
			}
			o[3*xx], o[3*xx+1], o[3*xx+2] = clip8(s0), clip8(s1), clip8(s2)
		}
	}
	return m
}

// resampleV is _ImagingResampleVertical_8bpc: in.W×out, each pixel a
// weighted sum down its column.
func resampleV(in *RGB, out int, c *coeffs) *RGB {
	w := in.W
	m := &RGB{W: w, H: out, Pix: make([]byte, 3*w*out)}
	for yy := range out {
		ymin, n := c.bounds[2*yy], c.bounds[2*yy+1]
		k := c.k[yy*c.size:]
		o := m.Pix[3*yy*w:]
		for xx := range w {
			s0, s1, s2 := int32(1)<<(precisionBits-1), int32(1)<<(precisionBits-1), int32(1)<<(precisionBits-1)
			for y := range n {
				p := in.Pix[3*((y+ymin)*w+xx):]
				s0 += int32(p[0]) * k[y]
				s1 += int32(p[1]) * k[y]
				s2 += int32(p[2]) * k[y]
			}
			o[3*xx], o[3*xx+1], o[3*xx+2] = clip8(s0), clip8(s1), clip8(s2)
		}
	}
	return m
}

// Contain resizes the image to fit w×h, keeping its aspect ratio
// (ImageOps.contain with BICUBIC).
func (m *RGB) Contain(w, h int) *RGB {
	if m.H == 0 {
		return &RGB{W: w}
	}
	if h == 0 {
		return &RGB{}
	}
	im := float64(m.W) / float64(m.H)
	dest := float64(w) / float64(h)
	if im != dest {
		// Python's round: half to even.
		if im > dest {
			h = int(math.RoundToEven(float64(m.H) / float64(m.W) * float64(w)))
		} else {
			w = int(math.RoundToEven(float64(m.W) / float64(m.H) * float64(h)))
		}
	}
	if w == 0 || h == 0 {
		return &RGB{W: w, H: h}
	}
	return m.Resize(w, h)
}

// Pad resizes the image to fit w×h and centres it on a fill-coloured w×h
// image (ImageOps.pad with BICUBIC, centering (0.5, 0.5)).
func (m *RGB) Pad(w, h int, fill [3]byte) *RGB {
	r := m.Contain(w, h)
	if r.W == w && r.H == h {
		return r
	}
	out := NewRGB(w, h, fill)
	x, y := 0, 0
	if r.W != w {
		x = int(math.RoundToEven(float64(w-r.W) * 0.5))
	} else {
		y = int(math.RoundToEven(float64(h-r.H) * 0.5))
	}
	for row := range r.H {
		copy(out.Pix[3*((row+y)*w+x):], r.Pix[3*row*r.W:3*(row+1)*r.W])
	}
	return out
}
