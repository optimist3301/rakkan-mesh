package sandbox

import (
	"bytes"
	"context"
	"image"
	"image/draw"
	"image/jpeg"
	"image/png"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
)

// Image kernels: native image operations a task drives from WebAssembly.
// Decoding, resampling and encoding run as host code, many times faster
// than the same code compiled to WebAssembly, yet inside the sandboxed
// worker, since image decoders handle untrusted input.
//
// Images live in a per-task table and are referred to by handle. Their
// pixels count against a budget equal to the task's memory limit, and
// dimensions are checked before decoding, so a small file claiming huge
// dimensions is refused rather than decoded. Every operation is integer
// arithmetic or a standard-library codec, so results are bit-identical on
// every architecture (for a given Go version) and replicas can be compared.
// Gas is charged per pixel.
//
// Functions return a handle or a byte count, or a negative error code.
const (
	ImageModule     = "rakkan_image_v1"
	ImageCapability = "image/1"
)

// Error codes returned by image functions.
const (
	imgErrArgs    = -1 // bad handle or arguments
	imgErrFormat  = -2 // not a decodable image
	imgErrTooBig  = -3 // over the pixel budget
	imgErrEncode  = -4 // encoding failed
	maxDimension  = 1 << 15
	formatPNG     = 0
	formatJPEG    = 1
	pixelGas      = 4 // per pixel processed, per pass
	codecPixelGas = 16
)

var imageABI = map[string]signature{
	// decode_input() / decode_blob(index) / decode(src, len) -> handle
	"decode_input": sig(nil, i32),
	"decode_blob":  sig([]api.ValueType{i32}, i32),
	"decode":       sig([]api.ValueType{i32, i32}, i32),
	// create(width, height) -> handle of a transparent image
	"create": sig([]api.ValueType{i32, i32}, i32),
	"width":  sig([]api.ValueType{i32}, i32),
	"height": sig([]api.ValueType{i32}, i32),
	// resize(h, width, height), fit(h, max_width, max_height) keeping the
	// aspect ratio, crop(h, x, y, width, height) -> new handle
	"resize": sig([]api.ValueType{i32, i32, i32}, i32),
	"fit":    sig([]api.ValueType{i32, i32, i32}, i32),
	"crop":   sig([]api.ValueType{i32, i32, i32, i32, i32}, i32),
	// grayscale(h), blur(h, radius) -> new handle
	"grayscale": sig([]api.ValueType{i32}, i32),
	"blur":      sig([]api.ValueType{i32, i32}, i32),
	// pixels_read(h, dst, len) / pixels_write(h, src, len): RGBA, 8 bits
	// per channel, non-premultiplied, rows top to bottom -> bytes copied
	"pixels_read":  sig([]api.ValueType{i32, i32, i32}, i32),
	"pixels_write": sig([]api.ValueType{i32, i32, i32}, i32),
	// encode_output(h, format, quality) appends the encoded image to the
	// task output -> bytes written. format: 0 PNG, 1 JPEG (quality 1-100).
	"encode_output": sig([]api.ValueType{i32, i32, i32}, i32),
	// free(h) releases an image and its share of the budget.
	"free": sig([]api.ValueType{i32}, i32),
}

type imageTable struct {
	images map[int32]*image.NRGBA
	next   int32
	bytes  int64 // pixel bytes held
}

func (s *execState) budget() int64 { return int64(s.lim.MaxMemoryPages) * PageSize }

// add stores an image, refusing it if it would exceed the budget.
func (s *execState) addImage(img *image.NRGBA) int32 {
	t := &s.images
	if t.images == nil {
		t.images = map[int32]*image.NRGBA{}
	}
	t.next++
	t.images[t.next] = img
	t.bytes += int64(len(img.Pix))
	return t.next
}

// reserve checks that an image of w x h fits the budget before it is made.
func (s *execState) reserve(w, h int) bool {
	if w <= 0 || h <= 0 || w > maxDimension || h > maxDimension {
		return false
	}
	return s.images.bytes+int64(w)*int64(h)*4 <= s.budget()
}

func (s *execState) image(h int32) *image.NRGBA { return s.images.images[h] }

func (s *execState) instantiateImage(ctx context.Context, rt wazero.Runtime) error {
	b := rt.NewHostModuleBuilder(ImageModule)
	fns := map[string]any{
		"decode_input":  s.imgDecodeInput,
		"decode_blob":   s.imgDecodeBlob,
		"decode":        s.imgDecode,
		"create":        s.imgCreate,
		"width":         s.imgWidth,
		"height":        s.imgHeight,
		"resize":        s.imgResize,
		"fit":           s.imgFit,
		"crop":          s.imgCrop,
		"grayscale":     s.imgGrayscale,
		"blur":          s.imgBlur,
		"pixels_read":   s.imgPixelsRead,
		"pixels_write":  s.imgPixelsWrite,
		"encode_output": s.imgEncodeOutput,
		"free":          s.imgFree,
	}
	for name, fn := range fns {
		b = b.NewFunctionBuilder().WithFunc(fn).Export(name)
	}
	_, err := b.Instantiate(ctx)
	return err
}

// decodeBytes decodes PNG or JPEG after checking the declared dimensions.
func (s *execState) decodeBytes(data []byte) int32 {
	s.chargeUnits(int64(len(data)) >> 4)
	cfg, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return imgErrFormat
	}
	if !s.reserve(cfg.Width, cfg.Height) {
		return imgErrTooBig
	}
	s.chargeUnits(int64(cfg.Width) * int64(cfg.Height) * codecPixelGas)
	src, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return imgErrFormat
	}
	b := src.Bounds()
	dst := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(dst, dst.Rect, src, b.Min, draw.Src)
	return s.addImage(dst)
}

func (s *execState) imgDecodeInput() int32 {
	return s.decodeBytes(s.input)
}

func (s *execState) imgDecodeBlob(index uint32) int32 {
	if index >= uint32(len(s.blobs)) {
		return imgErrArgs
	}
	b := s.blobs[index]
	// Encoded images are far smaller than their pixels; bound the read by
	// the pixel budget too.
	if b.Size() > s.budget() {
		return imgErrTooBig
	}
	data := make([]byte, b.Size())
	if n, err := b.ReadAt(data, 0); int64(n) < b.Size() {
		s.trap(errBlob(index, err))
	}
	return s.decodeBytes(data)
}

func (s *execState) imgDecode(_ context.Context, m api.Module, src, n uint32) int32 {
	data, ok := m.Memory().Read(src, n)
	if !ok {
		return imgErrArgs
	}
	return s.decodeBytes(data)
}

func (s *execState) imgCreate(w, h int32) int32 {
	s.charge(0)
	if !s.reserve(int(w), int(h)) {
		return imgErrTooBig
	}
	return s.addImage(image.NewNRGBA(image.Rect(0, 0, int(w), int(h))))
}

func (s *execState) imgWidth(h int32) int32 {
	s.charge(0)
	if img := s.image(h); img != nil {
		return int32(img.Rect.Dx())
	}
	return imgErrArgs
}

func (s *execState) imgHeight(h int32) int32 {
	s.charge(0)
	if img := s.image(h); img != nil {
		return int32(img.Rect.Dy())
	}
	return imgErrArgs
}

func (s *execState) imgResize(h, w, ht int32) int32 {
	s.charge(0)
	src := s.image(h)
	if src == nil {
		return imgErrArgs
	}
	if !s.reserve(int(w), int(ht)) {
		return imgErrTooBig
	}
	sw, sh := src.Rect.Dx(), src.Rect.Dy()
	s.chargeUnits((int64(sw)*int64(ht) + int64(w)*int64(ht) + int64(sw)*int64(sh)) * pixelGas)
	return s.addImage(resizeBox(src, int(w), int(ht)))
}

func (s *execState) imgFit(h, maxW, maxH int32) int32 {
	src := s.image(h)
	if src == nil || maxW <= 0 || maxH <= 0 {
		s.charge(0)
		return imgErrArgs
	}
	sw, sh := int64(src.Rect.Dx()), int64(src.Rect.Dy())
	w, ht := sw, sh
	if w > int64(maxW) {
		w, ht = int64(maxW), max(1, sh*int64(maxW)/sw)
	}
	if ht > int64(maxH) {
		w, ht = max(1, sw*int64(maxH)/sh), int64(maxH)
	}
	return s.imgResize(h, int32(w), int32(ht))
}

func (s *execState) imgCrop(h, x, y, w, ht int32) int32 {
	s.charge(0)
	src := s.image(h)
	r := image.Rect(int(x), int(y), int(x)+int(w), int(y)+int(ht))
	if src == nil || w <= 0 || ht <= 0 || !r.In(src.Rect) {
		return imgErrArgs
	}
	if !s.reserve(int(w), int(ht)) {
		return imgErrTooBig
	}
	s.chargeUnits(int64(w) * int64(ht) * pixelGas)
	dst := image.NewNRGBA(image.Rect(0, 0, int(w), int(ht)))
	for row := 0; row < int(ht); row++ {
		copy(dst.Pix[row*dst.Stride:row*dst.Stride+int(w)*4], src.Pix[src.PixOffset(int(x), int(y)+row):])
	}
	return s.addImage(dst)
}

func (s *execState) imgGrayscale(h int32) int32 {
	s.charge(0)
	src := s.image(h)
	if src == nil {
		return imgErrArgs
	}
	if !s.reserve(src.Rect.Dx(), src.Rect.Dy()) {
		return imgErrTooBig
	}
	s.chargeUnits(int64(len(src.Pix)/4) * pixelGas)
	dst := image.NewNRGBA(src.Rect)
	for i := 0; i < len(src.Pix); i += 4 {
		// Rec. 601 luma in 16-bit fixed point, as image/color does.
		y := uint8((19595*uint32(src.Pix[i]) + 38470*uint32(src.Pix[i+1]) + 7471*uint32(src.Pix[i+2]) + 1<<15) >> 16)
		dst.Pix[i], dst.Pix[i+1], dst.Pix[i+2], dst.Pix[i+3] = y, y, y, src.Pix[i+3]
	}
	return s.addImage(dst)
}

func (s *execState) imgBlur(h, radius int32) int32 {
	s.charge(0)
	src := s.image(h)
	if src == nil || radius < 0 || radius > 1024 {
		return imgErrArgs
	}
	if !s.reserve(src.Rect.Dx(), src.Rect.Dy()) {
		return imgErrTooBig
	}
	s.chargeUnits(int64(len(src.Pix)/4) * 2 * pixelGas)
	return s.addImage(boxBlur(src, int(radius)))
}

func (s *execState) imgPixelsRead(_ context.Context, m api.Module, h int32, dst, n uint32) int32 {
	img := s.image(h)
	if img == nil || uint64(n) < uint64(len(img.Pix)) {
		s.charge(0)
		return imgErrArgs
	}
	s.charge(uint32(len(img.Pix)))
	if !m.Memory().Write(dst, img.Pix) {
		s.trap(errMemory("pixels_read", dst, uint32(len(img.Pix))))
	}
	return int32(len(img.Pix))
}

func (s *execState) imgPixelsWrite(_ context.Context, m api.Module, h int32, src, n uint32) int32 {
	img := s.image(h)
	if img == nil || uint64(n) < uint64(len(img.Pix)) {
		s.charge(0)
		return imgErrArgs
	}
	s.charge(uint32(len(img.Pix)))
	buf, ok := m.Memory().Read(src, uint32(len(img.Pix)))
	if !ok {
		s.trap(errMemory("pixels_write", src, uint32(len(img.Pix))))
	}
	copy(img.Pix, buf)
	return int32(len(img.Pix))
}

func (s *execState) imgEncodeOutput(h, format, quality int32) int32 {
	s.charge(0)
	img := s.image(h)
	if img == nil || format != formatPNG && format != formatJPEG || format == formatJPEG && (quality < 1 || quality > 100) {
		return imgErrArgs
	}
	s.chargeUnits(int64(len(img.Pix)/4) * codecPixelGas)
	var buf bytes.Buffer
	var err error
	if format == formatPNG {
		err = (&png.Encoder{CompressionLevel: png.DefaultCompression}).Encode(&buf, img)
	} else {
		err = jpeg.Encode(&buf, img, &jpeg.Options{Quality: int(quality)})
	}
	if err != nil {
		return imgErrEncode
	}
	if int64(len(s.output))+int64(buf.Len()) > int64(s.lim.MaxOutputBytes) {
		s.trap(errOutput(s.lim.MaxOutputBytes))
	}
	s.chargeUnits(int64(buf.Len()) >> 4)
	s.output = append(s.output, buf.Bytes()...)
	return int32(buf.Len())
}

func (s *execState) imgFree(h int32) int32 {
	s.charge(0)
	img := s.image(h)
	if img == nil {
		return imgErrArgs
	}
	s.images.bytes -= int64(len(img.Pix))
	delete(s.images.images, h)
	return 0
}

// resizeBox resamples with a box filter in exact integer arithmetic: each
// output pixel is the area-weighted average of the source pixels it covers.
// Separable: rows first, then columns.
func resizeBox(src *image.NRGBA, w, h int) *image.NRGBA {
	sw, sh := src.Rect.Dx(), src.Rect.Dy()
	tmp := make([]uint8, w*sh*4) // horizontally resampled, w x sh
	for y := 0; y < sh; y++ {
		row := src.Pix[y*src.Stride : y*src.Stride+sw*4]
		resample1D(row, 4, sw, tmp[y*w*4:], 4, w)
	}
	dst := image.NewNRGBA(image.Rect(0, 0, w, h))
	for x := 0; x < w; x++ {
		resample1D(tmp[x*4:], w*4, sh, dst.Pix[x*4:], dst.Stride, h)
	}
	return dst
}

// resample1D maps n source pixels (stride apart, 4 channels) onto m output
// pixels. In units where the line is n*m long, source pixel j spans
// [j*m, (j+1)*m) and output pixel i spans [i*n, (i+1)*n); each source pixel
// contributes in proportion to the overlap.
func resample1D(src []uint8, srcStride, n int, dst []uint8, dstStride, m int) {
	for i := 0; i < m; i++ {
		lo, hi := i*n, (i+1)*n
		var acc [4]uint64
		for j := lo / m; j*m < hi; j++ {
			overlap := uint64(min(hi, (j+1)*m) - max(lo, j*m))
			p := src[j*srcStride:]
			for c := range 4 {
				acc[c] += uint64(p[c]) * overlap
			}
		}
		q := dst[i*dstStride:]
		for c := range 4 {
			q[c] = uint8((acc[c] + uint64(n)/2) / uint64(n))
		}
	}
}

// boxBlur averages each pixel's (2r+1)-square neighbourhood, clamping at
// the edges, with running sums: cost independent of the radius.
func boxBlur(src *image.NRGBA, r int) *image.NRGBA {
	w, h := src.Rect.Dx(), src.Rect.Dy()
	tmp := make([]uint8, len(src.Pix))
	for y := 0; y < h; y++ {
		blur1D(src.Pix[y*src.Stride:], 4, w, tmp[y*w*4:], 4, r)
	}
	dst := image.NewNRGBA(src.Rect)
	for x := 0; x < w; x++ {
		blur1D(tmp[x*4:], w*4, h, dst.Pix[x*4:], dst.Stride, r)
	}
	return dst
}

func blur1D(src []uint8, srcStride, n int, dst []uint8, dstStride, r int) {
	clamp := func(i int) []uint8 { return src[min(max(i, 0), n-1)*srcStride:] }
	var acc [4]uint32
	for k := -r; k <= r; k++ {
		p := clamp(k)
		for c := range 4 {
			acc[c] += uint32(p[c])
		}
	}
	width := uint32(2*r + 1)
	for i := 0; i < n; i++ {
		q := dst[i*dstStride:]
		for c := range 4 {
			q[c] = uint8((acc[c] + width/2) / width)
		}
		out, in := clamp(i-r), clamp(i+r+1)
		for c := range 4 {
			acc[c] += uint32(in[c]) - uint32(out[c])
		}
	}
}
