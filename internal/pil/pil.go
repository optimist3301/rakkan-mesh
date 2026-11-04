// Package pil does, to the bit, what models' image preparation asks of
// Pillow (13.0.0): open an image, convert it to RGB, resize it with the
// BICUBIC filter and pad it to a size. The reference code of a model
// (DeepSeek-V4.1's image_processor.py) prepares its images with Pillow, and
// a node without Python has to give the model the same pixels.
//
// The arithmetic is Pillow's own: its resampler (libImaging/Resample.c)
// computes coefficients in float64 and sums 8-bit pixels in 22-bit fixed
// point; its mode conversions are integer. Go evaluates the float64
// expressions in the same order, and every product that C does not fuse is
// kept unfused here (an explicit conversion stops Go fusing a*b+c on
// arm64), so the results do not depend on the processor.
//
// Formats: PNG, all of its colour types and depths, as Pillow reads them.
// Others are refused.
package pil

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
)

// RGB is an 8-bit RGB image, Pillow's mode "RGB": Pix holds W·H pixels,
// row by row, three bytes each.
type RGB struct {
	W, H int
	Pix  []byte
}

// NewRGB is a W×H image filled with one colour (Image.new).
func NewRGB(w, h int, fill [3]byte) *RGB {
	m := &RGB{W: w, H: h, Pix: make([]byte, 3*w*h)}
	for i := 0; i < len(m.Pix); i += 3 {
		copy(m.Pix[i:i+3], fill[:])
	}
	return m
}

// MaxPixels bounds the images Decode accepts (Pillow's decompression bomb
// limit is twice its MAX_IMAGE_PIXELS, 89478485; a node needs far less).
const MaxPixels = 64 << 20

// Decode opens an image and converts it to RGB, as Image.open(...).convert
// ("RGB") does: alpha is dropped, not composed; grey is repeated in the
// three channels; 16-bit samples keep their high byte, but for 16-bit
// grey, which Pillow opens as mode I;16 and clips to 255.
func Decode(data []byte) (*RGB, error) {
	if !bytes.HasPrefix(data, []byte("\x89PNG\r\n\x1a\n")) {
		return nil, errors.New("pil: not a PNG image")
	}
	cfg, err := png.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("pil: %w", err)
	}
	if cfg.Width <= 0 || cfg.Height <= 0 || int64(cfg.Width)*int64(cfg.Height) > MaxPixels {
		return nil, fmt.Errorf("pil: a %d×%d image", cfg.Width, cfg.Height)
	}
	img, err := png.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("pil: %w", err)
	}
	// IHDR, the first chunk: bit depth and colour type at bytes 24, 25.
	grey16 := len(data) > 25 && data[24] == 16 && data[25] == 0
	return toRGB(img, grey16)
}

// toRGB converts what image/png gives. Its colours are not premultiplied
// (NRGBA, NRGBA64, a palette of NRGBA), so the channels are the file's.
// grey16 tells 16-bit grey, which a tRNS chunk makes NRGBA64.
func toRGB(img image.Image, grey16 bool) (*RGB, error) {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	m := &RGB{W: w, H: h, Pix: make([]byte, 3*w*h)}
	set := func(x, y int, r, g, bl byte) {
		i := 3 * (y*w + x)
		m.Pix[i], m.Pix[i+1], m.Pix[i+2] = r, g, bl
	}
	switch im := img.(type) {
	case *image.Gray:
		for y := range h {
			for x := range w {
				v := im.Pix[y*im.Stride+x]
				set(x, y, v, v, v)
			}
		}
	case *image.Gray16:
		// Mode I;16 to RGB (Convert.c, I16_RGB): values above 255 are 255.
		for y := range h {
			for x := range w {
				i := y*im.Stride + 2*x
				v := uint16(im.Pix[i])<<8 | uint16(im.Pix[i+1])
				g := byte(255)
				if v < 256 {
					g = byte(v)
				}
				set(x, y, g, g, g)
			}
		}
	case *image.RGBA:
		// Opaque truecolour: image/png gives RGBA with alpha 255.
		for y := range h {
			for x := range w {
				i := y*im.Stride + 4*x
				set(x, y, im.Pix[i], im.Pix[i+1], im.Pix[i+2])
			}
		}
	case *image.NRGBA:
		for y := range h {
			for x := range w {
				i := y*im.Stride + 4*x
				set(x, y, im.Pix[i], im.Pix[i+1], im.Pix[i+2])
			}
		}
	case *image.RGBA64:
		for y := range h {
			for x := range w {
				i := y*im.Stride + 8*x
				set(x, y, im.Pix[i], im.Pix[i+2], im.Pix[i+4])
			}
		}
	case *image.NRGBA64:
		// 16-bit with alpha (RGBA;16B, LA;16B) or with tRNS: high bytes,
		// but grey with tRNS, which Pillow still opens as I;16.
		if grey16 {
			for y := range h {
				for x := range w {
					i := y*im.Stride + 8*x
					v := uint16(im.Pix[i])<<8 | uint16(im.Pix[i+1])
					g := byte(255)
					if v < 256 {
						g = byte(v)
					}
					set(x, y, g, g, g)
				}
			}
			break
		}
		for y := range h {
			for x := range w {
				i := y*im.Stride + 8*x
				set(x, y, im.Pix[i], im.Pix[i+2], im.Pix[i+4])
			}
		}
	case *image.Paletted:
		pal := make([][3]byte, len(im.Palette))
		for i, c := range im.Palette {
			n := color.NRGBAModel.Convert(c).(color.NRGBA)
			pal[i] = [3]byte{n.R, n.G, n.B}
		}
		for y := range h {
			for x := range w {
				p := pal[im.Pix[y*im.Stride+x]]
				set(x, y, p[0], p[1], p[2])
			}
		}
	default:
		return nil, fmt.Errorf("pil: unexpected decoded image %T", img)
	}
	return m, nil
}
