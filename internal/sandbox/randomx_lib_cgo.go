//go:build cgo && ((linux && (amd64 || arm64)) || (windows && amd64))

package sandbox

/*
// `make randomx` installs the library under build/randomx/<os>-<arch>; other
// places come from CGO_CFLAGS and CGO_LDFLAGS.
#cgo linux,amd64 CFLAGS: -I${SRCDIR}/../../build/randomx/linux-amd64/include
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/../../build/randomx/linux-amd64/lib
#cgo linux,arm64 CFLAGS: -I${SRCDIR}/../../build/randomx/linux-arm64/include
#cgo linux,arm64 LDFLAGS: -L${SRCDIR}/../../build/randomx/linux-arm64/lib
#cgo windows,amd64 CFLAGS: -I${SRCDIR}/../../build/randomx/windows-amd64/include
#cgo windows,amd64 LDFLAGS: -L${SRCDIR}/../../build/randomx/windows-amd64/lib
// The library is C++: link its runtime in, so the binary needs no libstdc++
// (or libgcc_s, or the MinGW DLLs) on the machine it runs on.
#cgo linux LDFLAGS: -lrandomx -Wl,-Bstatic -lstdc++ -Wl,-Bdynamic -lm -static-libgcc
#cgo windows LDFLAGS: -lrandomx -lstdc++ -static
#include <randomx.h>
#include <stdint.h>
#include <stdlib.h>

// randomx_dataset is opaque in randomx.h and there is no constructor for a
// dataset over memory somebody else owns, which the shared memfd needs. Its
// first member has always been the pointer to the items (src/dataset.hpp:
// struct randomx_dataset { uint8_t *memory; dealloc }), so a head of the
// same shape is enough for randomx_init_dataset and randomx_create_vm, which
// read nothing else. rx_wrap_dataset proves that through the library itself:
// randomx_get_dataset_memory must hand the pointer back, or the layout is not
// what we assume and the wrapper is refused.
struct rx_dataset_head {
	uint8_t *memory;
	void *dealloc;
};

static randomx_dataset *rx_wrap_dataset(void *mem) {
	struct rx_dataset_head *h = calloc(1, sizeof *h);
	if (h == NULL) return NULL;
	h->memory = mem;
	if (randomx_get_dataset_memory((randomx_dataset *)h) != mem) {
		free(h);
		return NULL;
	}
	return (randomx_dataset *)h;
}

static void rx_free_wrapper(randomx_dataset *d) { free(d); }

// randomx_get_flags may offer large pages never; take only what it chose for
// speed (JIT, hardware AES, the Argon2 variant) so the build also works on
// CPUs without them.
static randomx_flags rx_flags(void) {
	return randomx_get_flags() & ~RANDOMX_FLAG_LARGE_PAGES;
}
static randomx_flags rx_full(void) { return RANDOMX_FLAG_FULL_MEM; }
*/
import "C"

import (
	"errors"
	"sync"
	"unsafe"
)

const (
	rxlibName      = "librandomx"
	rxlibAvailable = true
)

type rxlibCache struct{ c *C.randomx_cache }

func newRxlibCache() (*rxlibCache, error) {
	c := C.randomx_alloc_cache(C.rx_flags())
	if c == nil {
		return nil, errors.New("randomx: cannot allocate the cache")
	}
	return &rxlibCache{c}, nil
}

// init (re)builds the cache for key, in place.
func (c *rxlibCache) init(key []byte) {
	C.randomx_init_cache(c.c, unsafe.Pointer(unsafe.SliceData(key)), C.size_t(len(key)))
}

func (c *rxlibCache) close() { C.randomx_release_cache(c.c); c.c = nil }

type rxlibVM struct{ vm *C.randomx_vm }

// newRxlibVM makes a VM that hashes against ds.
func newRxlibVM(ds *rxlibDataset) (*rxlibVM, error) {
	vm := C.randomx_create_vm(C.rx_flags()|C.rx_full(), nil, ds.d)
	if vm == nil {
		return nil, errors.New("randomx: cannot create a VM")
	}
	return &rxlibVM{vm}, nil
}

func (v *rxlibVM) hash(in []byte, out *[rxHashSize]byte) {
	var p unsafe.Pointer
	if len(in) > 0 {
		p = unsafe.Pointer(&in[0])
	}
	C.randomx_calculate_hash(v.vm, p, C.size_t(len(in)), unsafe.Pointer(&out[0]))
}

func (v *rxlibVM) close() { C.randomx_destroy_vm(v.vm); v.vm = nil }

type rxlibDataset struct {
	d *C.randomx_dataset
	// mem keeps the memory reachable for the Go collector for as long as the
	// library holds a pointer into it (it only matters for memory from the Go
	// heap; the node's is a mapping).
	mem []byte
}

// newRxlibDataset makes a dataset over mem (rxDatasetSize bytes, 64-byte
// aligned), which the caller keeps mapped for as long as it is used.
func newRxlibDataset(mem []byte) (*rxlibDataset, error) {
	if len(mem) < rxDatasetSize {
		return nil, errors.New("randomx dataset memory too small")
	}
	if uintptr(unsafe.Pointer(&mem[0]))%rxDatasetItemSize != 0 {
		return nil, errors.New("randomx dataset memory not aligned")
	}
	if uint64(C.randomx_dataset_item_count())*rxDatasetItemSize != rxDatasetSize {
		return nil, errors.New("randomx: librandomx has a dataset of another size")
	}
	d := C.rx_wrap_dataset(unsafe.Pointer(&mem[0]))
	if d == nil {
		return nil, errors.New("randomx: unsupported librandomx dataset layout")
	}
	return &rxlibDataset{d: d, mem: mem}, nil
}

func (d *rxlibDataset) fill(c *rxlibCache, threads int) {
	total := uint64(C.randomx_dataset_item_count())
	threads = max(threads, 1)
	var wg sync.WaitGroup
	for i := range threads {
		// Whole multiples of four items take the library's fast path.
		a := total * uint64(i) / uint64(threads) &^ 3
		b := total * uint64(i+1) / uint64(threads) &^ 3
		if i == threads-1 {
			b = total
		}
		if b <= a {
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			C.randomx_init_dataset(d.d, c.c, C.ulong(a), C.ulong(b-a))
		}()
	}
	wg.Wait()
}
