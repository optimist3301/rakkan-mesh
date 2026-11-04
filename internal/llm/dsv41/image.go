package dsv41

import (
	"errors"
	"fmt"
	"math"

	"github.com/optimist3301/rakkan-mesh/internal/num"
	"github.com/optimist3301/rakkan-mesh/internal/pil"
)

// Images (image_processor.py). An image is cut into Patch×Patch pixel
// patches for the image encoder, which sees a VitH×VitW grid of them; the
// aligner merges Downsample×Downsample patches into one of the model's
// positions, an LLMH×LLMW grid, which the model reads as the span
//
//	IMAGE_START, (IMAGE × LLMW, IMAGE_NEW_LINE) × LLMH, IMAGE_END
//
// every position of it carrying the image token. The preparation is
// Pillow's (package pil), so the pixels are the reference's.

// Token types of an image span's positions; text is TokenText.
const (
	TokenText = -1 + iota
	TokenImageStart
	TokenImage
	TokenImageNewLine
	TokenImageEnd
)

// ImageGrid is an image's plan: the pixel size it is brought to, and its
// patch and token grids.
type ImageGrid struct {
	Height, Width int // pixels
	VitH, VitW    int // patches
	LLMH, LLMW    int // positions
}

// Tokens is how many positions the image's span takes.
func (g ImageGrid) Tokens() int { return imageTokens(g.LLMH, g.LLMW) }

// Types are the token types of the image's span, in order.
func (g ImageGrid) Types() []int {
	t := make([]int, 0, g.Tokens())
	t = append(t, TokenImageStart)
	for range g.LLMH {
		for range g.LLMW {
			t = append(t, TokenImage)
		}
		t = append(t, TokenImageNewLine)
	}
	return append(t, TokenImageEnd)
}

func imageTokens(h, w int) int { return h*(w+1) + 2 }

func ceilDiv(a, b int) int { return (a + b - 1) / b }

// PlanImage is plan_image_grid: the grid of an image of the given size.
// Its float64 arithmetic is Python's, in its order.
func (c *Config) PlanImage(width, height int) (ImageGrid, error) {
	v := &c.Vision
	if v.Layers == 0 {
		return ImageGrid{}, errors.New("dsv41: the model has no image encoder")
	}
	if width <= 0 || height <= 0 || width > pil.MaxPixels || height > pil.MaxPixels {
		return ImageGrid{}, fmt.Errorf("dsv41: a %d×%d image", width, height)
	}
	p, r := v.Patch, v.Downsample
	if v.MaxWHRatio != 0 && width > height*v.MaxWHRatio {
		width = height * v.MaxWHRatio
	}
	if area := width * height; area < v.MinPixels {
		// (min_pixels / area) ** 0.5: C's pow, which is sqrt for a half
		// power wherever it is correctly rounded.
		ratio := math.Sqrt(float64(v.MinPixels) / float64(area))
		width = int(float64(width) * ratio)
		height = int(float64(height) * ratio)
	}
	g := ImageGrid{Height: ceilDiv(height, p) * p, Width: ceilDiv(width, p) * p}
	g.LLMH, g.LLMW = ceilDiv(g.Height/p, r), ceilDiv(g.Width/p, r)
	if g.Tokens() > v.MaxTokens {
		// solve_resize_ratio: the largest size keeping the aspect ratio.
		rr := float64(height) / float64(width)
		maxW := math.Sqrt(float64(v.MaxTokens-2)/rr+0.25) - 0.5
		maxH := maxW * rr
		cell := p * r
		switch {
		case maxW < 1: // very tall: one column
			g.Height, g.Width = (v.MaxTokens-2)/2*cell, cell
		case maxH < 1: // very wide: one row
			g.Height, g.Width = cell, (v.MaxTokens-3)*cell
		default:
			beta := min(float64(int(math.Floor(maxW))*cell)/float64(width), float64(int(math.Floor(maxH))*cell)/float64(height))
			g.Height = int(math.Floor(float64(height)*beta/float64(p))) * p
			g.Width = int(math.Floor(float64(width)*beta/float64(p))) * p
		}
		g.LLMH, g.LLMW = ceilDiv(g.Height/p, r), ceilDiv(g.Width/p, r)
		if g.Tokens() > v.MaxTokens || g.Height == 0 || g.Width == 0 {
			return ImageGrid{}, fmt.Errorf("dsv41: no grid of at most %d tokens for a %d×%d image", v.MaxTokens, width, height)
		}
	}
	g.VitH, g.VitW = g.Height/p, g.Width/p
	return g, nil
}

// Image is a prepared image: its grid and its patches, VitH·VitW of them
// in reading order, each 3·Patch·Patch bf16 values (channel, row, column)
// in [−1, 1].
type Image struct {
	ImageGrid
	Patches []float32
}

// PrepareImage is load_image: an encoded image (PNG) to the patches the
// encoder takes.
func (c *Config) PrepareImage(data []byte) (*Image, error) {
	src, err := pil.Decode(data)
	if err != nil {
		return nil, err
	}
	g, err := c.PlanImage(src.W, src.H)
	if err != nil {
		return nil, err
	}
	var px *pil.RGB
	if r := c.Vision.MaxWHRatio; r != 0 && src.W >= r*src.H {
		px = src.Resize(g.Width, g.Height)
	} else {
		px = src.Pad(g.Width, g.Height, [3]byte{127, 127, 127})
	}
	p := c.Vision.Patch
	img := &Image{ImageGrid: g, Patches: make([]float32, g.VitH*g.VitW*3*p*p)}
	// ((x / 255) − 0.5) / 0.5 in float32, then bf16: one value per byte.
	var level [256]float32
	for b := range level {
		x := float32(b) / 255
		level[b] = num.RoundBF16((x - 0.5) / 0.5)
	}
	o := 0
	for i := range g.VitH {
		for j := range g.VitW {
			for ch := range 3 {
				for y := range p {
					row := px.Pix[3*((i*p+y)*g.Width+j*p):]
					for x := range p {
						img.Patches[o] = level[row[3*x+ch]]
						o++
					}
				}
			}
		}
	}
	return img, nil
}
