package sandbox

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	randomx "git.gammaspectra.live/P2Pool/go-randomx/v4"
)

// Fast RandomX: hashing against the 2 GiB dataset instead of the 256 MiB
// cache, about ten times quicker. Workers are separate processes, so the
// dataset cannot live in each of them. The node (the parent of the workers)
// owns one copy in a memfd, builds it, and hands the file to every worker as
// descriptor 3; the workers map it read-only.
//
// The memfd starts with a header page:
//
//	0   magic
//	8   generation, u64: odd while the dataset is being rebuilt
//	16  seed length, u32; 24 the seed it was built for (up to 64 bytes)
//	96  wanted seed length, u32; 104 the seed a worker asked for
//
// A worker that is asked to hash under another seed writes it as wanted and
// answers "not ready"; the node notices, rebuilds in place and bumps the
// generation. A search remembers the generation when it starts and discards
// its results if it changed, so a rebuild under a running search cannot
// produce a wrong hash. Tasks answer "not ready" with exit code 3 and the
// caller retries.
//
// Gas per hash is fixed here and does not depend on the node, as replicas
// compare gas; nodes without a dataset do not offer the capability.
const (
	RandomXFastModule     = "rakkan_randomx_fast_v1"
	RandomXFastCapability = "randomx-fast/1"
)

// randomxFastHashGas bills one fast hash: about 2 ms of a core.
const randomxFastHashGas = 14_000_000

const (
	// The header takes a whole huge page so that the dataset after it is
	// aligned for a hugetlb mapping; the dataset is rounded up to one too.
	hugePage         = 2 << 20
	sharedHeaderSize = hugePage
	datasetMapBytes  = (randomx.DatasetSize + hugePage - 1) / hugePage * hugePage
	// RandomXSharedBytes is the size of the shared dataset file.
	RandomXSharedBytes = sharedHeaderSize + datasetMapBytes
	// RandomXFastHostBytes is the address space a worker needs to map it.
	RandomXFastHostBytes = RandomXSharedBytes + 64<<20

	rxErrNoFast = -4 // this node has no shared dataset

	sharedFD = 3 // where a worker finds the file
)

const (
	offMagic   = 0
	offGen     = 8
	offSeedLen = 16
	offSeed    = 24
	offWantLen = 96
	offWant    = 104
	seedMax    = 64
)

var sharedMagic = [8]byte{'R', 'X', 'D', 'S', 'E', 'T', '0', '1'}

var randomxFastABI = randomxABI

type rxView struct {
	hdr  []byte // header page, read-write
	data []byte // the dataset
	ds   *randomx.Dataset
}

func newView(hdr, data []byte) *rxView {
	// Dataset has no constructor over foreign memory; its first field is the
	// slice of lines it reads. TestFastMatchesLight guards the layout.
	ds := &randomx.Dataset{}
	*(*[]randomx.RegisterLine)(unsafe.Pointer(ds)) = unsafe.Slice((*randomx.RegisterLine)(unsafe.Pointer(&data[0])), randomx.DatasetItemCount)
	return &rxView{hdr: hdr, data: data, ds: ds}
}

func (v *rxView) gen() *uint64 { return (*uint64)(unsafe.Pointer(&v.hdr[offGen])) }

func (v *rxView) field(lenOff, off int) []byte {
	n := binary.LittleEndian.Uint32(v.hdr[lenOff:])
	if n == 0 || n > seedMax {
		return nil
	}
	return v.hdr[off : off+int(n)]
}

func (v *rxView) setField(lenOff, off int, b []byte) {
	binary.LittleEndian.PutUint32(v.hdr[lenOff:], 0)
	copy(v.hdr[off:off+seedMax], b)
	binary.LittleEndian.PutUint32(v.hdr[lenOff:], uint32(len(b)))
}

func (v *rxView) seed() []byte   { return v.field(offSeedLen, offSeed) }
func (v *rxView) wanted() []byte { return v.field(offWantLen, offWant) }

var rxShared struct {
	mu      sync.Mutex
	v       *rxView
	tried   bool
	enabled bool // this process owns a dataset: advertise the capability
}

// sharedView is the dataset this process hashes against, if any: the node's
// own, or in a worker the one inherited as descriptor 3.
func sharedView() *rxView {
	rxShared.mu.Lock()
	defer rxShared.mu.Unlock()
	if rxShared.v == nil && !rxShared.tried {
		rxShared.tried = true
		rxShared.v = attachShared()
	}
	return rxShared.v
}

var rxFast = &rxBackend{gas: randomxFastHashGas, open: func(s *execState, seed []byte) (*randomx.VM, func() bool, int32) {
	v := sharedView()
	if v == nil {
		return nil, nil, rxErrNoFast
	}
	g := atomic.LoadUint64(v.gen())
	if g&1 == 1 || !bytes.Equal(v.seed(), seed) {
		if !bytes.Equal(v.wanted(), seed) {
			v.setField(offWantLen, offWant, seed)
		}
		return nil, nil, rxErrNotReady
	}
	vm, err := randomx.NewVM(randomxFastFlags(), nil, v.ds)
	if err != nil {
		s.trap(fmt.Errorf("%w: randomx vm: %v", ErrTrap, err))
	}
	return vm, func() bool {
		vm.Close()
		return atomic.LoadUint64(v.gen()) == g
	}, 0
}}

func randomxFastFlags() randomx.Flags { return randomxFlags() | randomx.RANDOMX_FLAG_FULL_MEM }

// RandomXShared is the node's copy of the dataset.
type RandomXShared struct {
	f *os.File
	// Huge reports that the dataset sits in huge pages.
	Huge bool
	v    *rxView
}

// NewRandomXShared allocates the shared file. Nothing is built until a worker
// asks for a seed; Serve does the building.
func NewRandomXShared() (*RandomXShared, error) {
	// Huge pages (vm.nr_hugepages) make the dataset's random reads about
	// twice as quick; without them ordinary pages do.
	for _, huge := range []bool{true, false} {
		f, err := newSharedFile(RandomXSharedBytes, huge)
		if err != nil {
			if huge {
				continue
			}
			return nil, err
		}
		mem, err := mapShared(int(f.Fd()), 0, RandomXSharedBytes, true)
		if err != nil {
			f.Close()
			if huge {
				continue
			}
			return nil, err
		}
		copy(mem[offMagic:], sharedMagic[:])
		return &RandomXShared{f: f, Huge: huge, v: newView(mem[:sharedHeaderSize:sharedHeaderSize], mem[sharedHeaderSize:])}, nil
	}
	panic("unreachable")
}

// File is the descriptor to give every worker as ExtraFiles[0].
func (s *RandomXShared) File() *os.File { return s.f }

// Enable makes this process offer randomx-fast/1 and, when it runs tasks
// itself, hash against the dataset.
func (s *RandomXShared) Enable() {
	rxShared.mu.Lock()
	rxShared.v, rxShared.enabled = s.v, true
	rxShared.mu.Unlock()
}

// Serve builds the dataset for whatever seed workers ask for until ctx ends.
// threads is how many cores a build uses.
func (s *RandomXShared) Serve(ctx context.Context, threads int, logf func(format string, args ...any)) {
	v := s.v
	tick := time.NewTicker(250 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
		want := bytes.Clone(v.wanted())
		if want == nil || bytes.Equal(want, v.seed()) {
			continue
		}
		start := time.Now()
		if err := s.build(want, threads); err != nil {
			logf("randomx dataset: %v", err)
			continue
		}
		logf("randomx dataset built for seed %x in %s", want, time.Since(start).Round(time.Millisecond))
	}
}

func (s *RandomXShared) build(seed []byte, threads int) error {
	if len(seed) == 0 || len(seed) > seedMax {
		return errors.New("bad seed")
	}
	v := s.v
	cache, err := randomx.NewCache(randomxFlags())
	if err != nil {
		return err
	}
	defer cache.Close()
	cache.Init(seed)

	atomic.AddUint64(v.gen(), 1) // odd: workers stop trusting the dataset
	binary.LittleEndian.PutUint32(v.hdr[offSeedLen:], 0)
	v.ds.InitDatasetParallel(cache, max(threads, 1))
	v.setField(offSeedLen, offSeed, seed)
	atomic.AddUint64(v.gen(), 1)
	return nil
}

// Close releases the file; workers keep their own mappings.
func (s *RandomXShared) Close() { s.f.Close() }

// attachShared maps the dataset inherited as descriptor 3, if there is one.
func attachShared() *rxView {
	size, ok := fdSize(sharedFD)
	if !ok || size != RandomXSharedBytes {
		return nil
	}
	hdr, err := mapShared(sharedFD, 0, sharedHeaderSize, true)
	if err != nil || !bytes.Equal(hdr[:8], sharedMagic[:]) {
		return nil
	}
	data, err := mapShared(sharedFD, sharedHeaderSize, datasetMapBytes, false)
	if err != nil {
		return nil
	}
	runtime.KeepAlive(hdr)
	return newView(hdr, data)
}
