package sandbox

// RandomX comes from the reference C++ library, librandomx v1.2.1 (the
// algorithm Monero runs), linked through cgo; `make randomx` builds it (see
// scripts/build-randomx.sh). Two files implement the few calls the kernels
// need, with the same names:
//
//	randomx_lib_cgo.go    cgo (the build the node ships)
//	                      (linux/amd64, linux/arm64, windows/amd64)
//	randomx_lib_nocgo.go  stubs that report the library missing, so that
//	                      CGO_ENABLED=0 and the other platforms still compile
//	                      and a node built that way simply has no RandomX
//
//	newRxlibCache() (*rxlibCache, error)
//	(*rxlibCache).init(key), .close()   the 256 MiB cache the dataset is built from
//	newRxlibDataset(mem)                a dataset living in memory the caller
//	                                    owns (the node's shared mapping)
//	(*rxlibDataset).fill(cache, n)      builds it on n threads
//	newRxlibVM(ds)                      a VM hashing against the dataset
//	(*rxlibVM).hash(in, out), .close()
const (
	rxHashSize = 32
	// rxDatasetSize is the 2 GiB dataset: RANDOMX_DATASET_BASE_SIZE plus
	// RANDOMX_DATASET_EXTRA_SIZE. The cgo build checks it against the library.
	rxDatasetSize     = 2181038016
	rxDatasetItemSize = 64
)

// RandomXAvailable reports whether this build links librandomx.
func RandomXAvailable() bool { return rxlibAvailable }
