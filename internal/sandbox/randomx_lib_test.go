package sandbox

import (
	"bytes"
	"encoding/hex"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// The tests here need librandomx (a CGO_ENABLED=1 build, see `make randomx`)
// and about 2 GiB; each dataset is built over a plain Go slice, as the node
// builds its own over shared memory.

// Vectors of the RandomX reference implementation (tests.cpp): key, input,
// hash.
var rxVectors = []struct{ key, in, hash string }{
	{"RandomX example key\x00", "RandomX example input\x00", "8a48e5f9db45ab79d9080574c4d81954fe6ac63842214aff73c244b26330b7c9"},
	{"test key 000", "This is a test", "639183aae1bf4c9a35884cb46b09cad9175f04efd7684e7262a0ac1c2f0b4e3f"},
	{"test key 000", "Lorem ipsum dolor sit amet", "300a0adb47603dedb42228ccb2b211104f4da45af709cd7547cd049e9489c969"},
	{"test key 000", "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua", "c36d4ed4191e617309867ed66a443be4075014e2b061bcdaf9ce7b721d2b77a8"},
	{"test key 001", "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua", "e9ff4503201c0c2cca26d285c93ae883f9b1d30c9eb240b820756f2d5a7905fc"},
}

func availableMiB() int {
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 1 << 30 // cannot tell
	}
	for _, l := range strings.Split(string(b), "\n") {
		if f := strings.Fields(l); len(f) >= 2 && f[0] == "MemAvailable:" {
			kb, _ := strconv.Atoi(f[1])
			return kb / 1024
		}
	}
	return 1 << 30
}

// needRxlib skips a test that cannot run here: no library, or not the memory
// for gib datasets (with headroom).
func needRxlib(t testing.TB, datasets int) {
	t.Helper()
	if !rxlibAvailable {
		t.Skip("this build has no RandomX")
	}
	if m := availableMiB(); m < datasets*2048+4096 {
		t.Skipf("only %d MiB available; %d dataset(s) need %d MiB and headroom", m, datasets, datasets*2048)
	}
}

func rxlibCacheFor(t testing.TB) *rxlibCache {
	t.Helper()
	c, err := newRxlibCache()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(c.close)
	return c
}

func rxlibDatasetOver(t testing.TB) (*rxlibDataset, []byte) {
	t.Helper()
	mem := make([]byte, rxDatasetSize)
	ds, err := newRxlibDataset(mem)
	if err != nil {
		t.Fatal(err)
	}
	return ds, mem
}

func rxlibHash(t testing.TB, vm *rxlibVM, in []byte) string {
	t.Helper()
	var out [rxHashSize]byte
	vm.hash(in, &out)
	return hex.EncodeToString(out[:])
}

func TestRxlibVectors(t *testing.T) {
	needRxlib(t, 1)
	t.Logf("backend: %s", rxlibName)
	cache := rxlibCacheFor(t)
	ds, _ := rxlibDatasetOver(t)
	built := ""
	for _, v := range rxVectors {
		if built != v.key {
			cache.init([]byte(v.key)) // re-keying in place, as the node does
			ds.fill(cache, runtime.NumCPU())
			built = v.key
		}
		vm, err := newRxlibVM(ds)
		if err != nil {
			t.Fatal(err)
		}
		got := rxlibHash(t, vm, []byte(v.in))
		vm.close()
		if got != v.hash {
			t.Errorf("key %q input %q: %s, want %s", v.key, v.in, got, v.hash)
		}
	}
}

// An odd thread count splits the build differently and must not change it.
func TestRxlibFillThreads(t *testing.T) {
	needRxlib(t, 2)
	cache := rxlibCacheFor(t)
	cache.init([]byte("test key 000"))
	ds7, mem7 := rxlibDatasetOver(t)
	ds7.fill(cache, 7)
	ds1, mem1 := rxlibDatasetOver(t)
	ds1.fill(cache, 1)
	if !bytes.Equal(mem7, mem1) {
		t.Fatal("datasets built on 7 and 1 threads differ")
	}
}

// Many VMs over one dataset at once, as concurrent tasks in a worker do.
func TestRxlibConcurrentVMs(t *testing.T) {
	needRxlib(t, 1)
	cache := rxlibCacheFor(t)
	cache.init([]byte("test key 000"))
	ds, _ := rxlibDatasetOver(t)
	ds.fill(cache, runtime.NumCPU())
	var wg sync.WaitGroup
	for g := range runtime.NumCPU() {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 3 {
				vm, err := newRxlibVM(ds)
				if err != nil {
					t.Error(err)
					return
				}
				got := rxlibHash(t, vm, []byte("This is a test"))
				vm.close()
				if got != rxVectors[1].hash {
					t.Errorf("goroutine %d: %s", g, got)
				}
			}
		}()
	}
	wg.Wait()
}

func TestRxlibRefusesBadMemory(t *testing.T) {
	needRxlib(t, 0)
	if _, err := newRxlibDataset(make([]byte, 1<<20)); err == nil {
		t.Fatal("a dataset over too little memory was accepted")
	}
}

func BenchmarkRxlibCacheInit(b *testing.B) {
	needRxlib(b, 0)
	c := rxlibCacheFor(b)
	for i := 0; b.Loop(); i++ {
		c.init([]byte{byte(i), 1, 2, 3})
	}
}

func BenchmarkRxlibDatasetFill(b *testing.B) {
	needRxlib(b, 1)
	cache := rxlibCacheFor(b)
	cache.init([]byte("test key 000"))
	ds, _ := rxlibDatasetOver(b)
	for b.Loop() {
		ds.fill(cache, runtime.NumCPU())
	}
}

func rxlibBenchDataset(b *testing.B) *rxlibDataset {
	needRxlib(b, 1)
	cache := rxlibCacheFor(b)
	cache.init([]byte("test key 000"))
	ds, _ := rxlibDatasetOver(b)
	ds.fill(cache, runtime.NumCPU())
	b.ResetTimer()
	return ds
}

// Hashes in a loop on one VM, the way a search runs.
func BenchmarkRxlibSearch(b *testing.B) {
	ds := rxlibBenchDataset(b)
	vm, err := newRxlibVM(ds)
	if err != nil {
		b.Fatal(err)
	}
	defer vm.close()
	blob := bytes.Repeat([]byte{0xAB}, 76)
	var h [rxHashSize]byte
	for i := 0; b.Loop(); i++ {
		blob[39] = byte(i)
		vm.hash(blob, &h)
	}
}

// All cores hashing at once: the node's real load.
func BenchmarkRxlibParallel(b *testing.B) {
	ds := rxlibBenchDataset(b)
	b.RunParallel(func(pb *testing.PB) {
		vm, err := newRxlibVM(ds)
		if err != nil {
			b.Error(err)
			return
		}
		defer vm.close()
		blob := bytes.Repeat([]byte{0xAB}, 76)
		var h [rxHashSize]byte
		for i := 0; pb.Next(); i++ {
			blob[39] = byte(i)
			vm.hash(blob, &h)
		}
	})
}
