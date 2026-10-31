package sandbox

import (
	"bytes"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"io"
	"testing"

	w "github.com/optimist3301/rakkan-mesh/internal/wasmtest"
)

func gradientPNG(t *testing.T, width, height int) []byte {
	t.Helper()
	img := image.NewNRGBA(image.Rect(0, 0, width, height))
	for y := range height {
		for x := range width {
			img.Set(x, y, color.NRGBA{uint8(x), uint8(y), uint8(x ^ y), 255})
		}
	}
	var buf bytes.Buffer
	png.Encode(&buf, img)
	return buf.Bytes()
}

func imageLimits() Limits {
	l := DefaultLimits()
	l.MaxMemoryPages = 256 // 16 MiB pixel budget
	return l
}

func TestThumbnail(t *testing.T) {
	src := gradientPNG(t, 1000, 600)
	for _, c := range []struct {
		format int32
		decode func(io.Reader) (image.Image, error)
	}{{0, png.Decode}, {1, jpeg.Decode}} {
		res, err := run(t, w.Thumbnail(256, 256, c.format, 85), src, imageLimits())
		if err != nil {
			t.Fatal(err)
		}
		if res.ExitCode != 0 {
			t.Fatalf("format %d: exit %d", c.format, res.ExitCode)
		}
		img, err := c.decode(bytes.NewReader(res.Output))
		if err != nil {
			t.Fatal(err)
		}
		if b := img.Bounds(); b.Dx() != 256 || b.Dy() != 153 {
			t.Fatalf("format %d: thumbnail %dx%d, want 256x153", c.format, b.Dx(), b.Dy())
		}
	}
}

func TestImageKernelsAreDeterministic(t *testing.T) {
	src := gradientPNG(t, 640, 480)
	a, err := run(t, w.Thumbnail(200, 200, 1, 90), src, imageLimits())
	if err != nil {
		t.Fatal(err)
	}
	b, _ := run(t, w.Thumbnail(200, 200, 1, 90), src, imageLimits())
	if !bytes.Equal(a.Output, b.Output) || a.GasUsed != b.GasUsed {
		t.Fatalf("outputs equal: %v, gas %d vs %d", bytes.Equal(a.Output, b.Output), a.GasUsed, b.GasUsed)
	}
	// Gas reflects the pixels processed, not just instructions.
	small, _ := run(t, w.Thumbnail(200, 200, 1, 90), gradientPNG(t, 64, 48), imageLimits())
	if small.GasUsed*10 > a.GasUsed {
		t.Fatalf("gas %d for 64x48, %d for 640x480", small.GasUsed, a.GasUsed)
	}
}

// pngClaiming rewrites a tiny PNG's header to claim other dimensions.
func pngClaiming(t *testing.T, width, height uint32) []byte {
	b := gradientPNG(t, 1, 1)
	// IHDR data starts after the 8-byte signature, length and type.
	binary.BigEndian.PutUint32(b[16:], width)
	binary.BigEndian.PutUint32(b[20:], height)
	binary.BigEndian.PutUint32(b[29:], crc32.ChecksumIEEE(b[12:29]))
	return b
}

func TestImageInputsAreBounded(t *testing.T) {
	cases := map[string][]byte{
		"decompression bomb": pngClaiming(t, 30000, 30000),
		"over the budget":    gradientPNG(t, 4000, 2000), // 32 MB of pixels
		"not an image":       []byte("definitely not a PNG"),
	}
	for name, input := range cases {
		t.Run(name, func(t *testing.T) {
			res, err := run(t, w.Thumbnail(100, 100, 0, 0), input, imageLimits())
			if err != nil {
				t.Fatal(err)
			}
			if res.ExitCode != 1 {
				t.Fatalf("exit %d, want 1 (decode refused)", res.ExitCode)
			}
		})
	}
}

func TestResizeBoxIsExact(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 2, 1))
	copy(src.Pix, []uint8{0, 0, 0, 255, 255, 255, 255, 255})
	if got := resizeBox(src, 1, 1).Pix; !bytes.Equal(got, []uint8{128, 128, 128, 255}) {
		t.Fatalf("2->1 average: %v", got)
	}
	if got := resizeBox(src, 4, 1).Pix; !bytes.Equal(got, []uint8{0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255}) {
		t.Fatalf("2->4 replicate: %v", got)
	}
	// A uniform image stays uniform under blur and any resize.
	u := image.NewNRGBA(image.Rect(0, 0, 37, 23))
	for i := range u.Pix {
		u.Pix[i] = 200
	}
	for _, img := range []*image.NRGBA{boxBlur(u, 5), resizeBox(u, 13, 41)} {
		for _, p := range img.Pix {
			if p != 200 {
				t.Fatalf("uniform image changed to %d", p)
			}
		}
	}
}

type memBlob struct {
	data []byte
	fail bool
}

func (m memBlob) Size() int64 { return int64(len(m.data)) }

func (m memBlob) ReadAt(p []byte, off int64) (int, error) {
	if m.fail {
		return 0, errors.New("provider gone")
	}
	return bytes.NewReader(m.data).ReadAt(p, off)
}

func TestBlobRead(t *testing.T) {
	data := bytes.Repeat([]byte("0123456789"), 30_000) // 300 kB, several reads
	r := NewRunner(DefaultLimits())
	defer r.Close(t.Context())
	res, err := r.Run(t.Context(), Task{Module: w.BlobEcho(), Blobs: []BlobSource{memBlob{data: data}}, Limits: DefaultLimits()})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(res.Output, data) {
		t.Fatalf("read back %d bytes", len(res.Output))
	}
	_, err = r.Run(t.Context(), Task{Module: w.BlobEcho(), Blobs: []BlobSource{memBlob{data: data, fail: true}}, Limits: DefaultLimits()})
	if !errors.Is(err, ErrBlobUnavailable) {
		t.Fatalf("got %v, want ErrBlobUnavailable", err)
	}
}

func TestCapabilities(t *testing.T) {
	if got := CapabilitiesFor([]string{HostModule, ImageModule, ImageModule}); len(got) != 1 || got[0] != ImageCapability {
		t.Fatalf("CapabilitiesFor: %v", got)
	}
	if got := CapabilitiesFor([]string{"wasi_snapshot_preview1"}); got[0] != "import:wasi_snapshot_preview1" {
		t.Fatalf("unknown namespace: %v", got)
	}
}
