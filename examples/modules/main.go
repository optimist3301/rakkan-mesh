// Command modules writes the example task modules used by the SDK tests
// and examples. Real tasks would be compiled from AssemblyScript, Rust or
// TinyGo; these are assembled directly to avoid a toolchain dependency.
//
//	go run ./examples/modules -out sdk/js/test/fixtures
package main

import (
	"bytes"
	"flag"
	"image"
	"image/color"
	"image/png"
	"log"
	"os"
	"path/filepath"

	"github.com/optimist3301/rakkan-mesh/internal/llm/testmodel"
	w "github.com/optimist3301/rakkan-mesh/internal/wasmtest"
)

// samplePNG is a test picture: a colour gradient.
func samplePNG(width, height int) []byte {
	img := image.NewNRGBA(image.Rect(0, 0, width, height))
	for y := range height {
		for x := range width {
			img.Set(x, y, color.NRGBA{uint8(255 * x / width), uint8(255 * y / height), 160, 255})
		}
	}
	var buf bytes.Buffer
	png.Encode(&buf, img)
	return buf.Bytes()
}

func main() {
	out := flag.String("out", ".", "output directory")
	flag.Parse()
	modules := map[string][]byte{
		"echo.wasm":           w.Echo(),
		"echo-big.wasm":       w.EchoPages(256), // 16 MiB of memory
		"search.wasm":         w.Search(),
		"spin.wasm":           w.Spin(),
		"blob-echo.wasm":      w.BlobEcho(),
		"thumbnail-png.wasm":  w.Thumbnail(256, 256, 0, 0),
		"thumbnail-jpeg.wasm": w.Thumbnail(320, 320, 1, 85),
		"sample.png":          samplePNG(800, 500),
		"tiny-model.bin":      testmodel.Checkpoint(1),
		"tiny-tokenizer.bin":  testmodel.Tokenizer(),
	}
	for name, m := range modules {
		if err := os.WriteFile(filepath.Join(*out, name), m, 0o644); err != nil {
			log.Fatal(err)
		}
	}
}
