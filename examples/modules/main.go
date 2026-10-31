// Command modules writes the example task modules used by the SDK tests
// and examples. Real tasks would be compiled from AssemblyScript, Rust or
// TinyGo; these are assembled directly to avoid a toolchain dependency.
//
//	go run ./examples/modules -out sdk/js/test/fixtures
package main

import (
	"flag"
	"log"
	"os"
	"path/filepath"

	w "github.com/optimist3301/rakkan-mesh/internal/wasmtest"
)

func main() {
	out := flag.String("out", ".", "output directory")
	flag.Parse()
	modules := map[string][]byte{
		"echo.wasm":     w.Echo(),
		"echo-big.wasm": w.EchoPages(256), // 16 MiB of memory
		"search.wasm":   w.Search(),
		"spin.wasm":     w.Spin(),
	}
	for name, m := range modules {
		if err := os.WriteFile(filepath.Join(*out, name), m, 0o644); err != nil {
			log.Fatal(err)
		}
	}
}
