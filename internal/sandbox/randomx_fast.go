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
)

// RandomX hashes against a 2 GiB dataset. Workers are separate processes, so
// the dataset cannot live in each of them. The node (the parent of the
// workers) owns one copy in shared memory (a memfd on Linux, a pagefile
// section on Windows), builds it, and hands it to every worker (descriptor 3
// on Linux, an inherited handle named by an environment variable on
// Windows); the workers map it read-only.
//
// The memfd starts with a header page:
//
//	0   magic
//	8   generation, u64: odd while the dataset is being rebuilt
//	16  seed length, u32; 24 the seed it was built for (up to 64 bytes)
//	96  wanted seed length, u32; 104 the seed a worker asked for
//	168 last use, u64: Unix nanoseconds of the latest hash request
//
// A worker that is asked to hash under another seed writes it as wanted and
// answers "not ready"; the node notices, rebuilds in place and bumps the
// generation. A search remembers the generation when it starts and discards
// its results if it changed, so a rebuild under a running search cannot
// produce a wrong hash. Tasks answer "not ready" with exit code 3 and the
// caller retries.
//
// The dataset takes memory only once it is built, and not for ever: when no
// hash has been asked for during RandomXShared.IdleRelease the node hands its
// pages back to the system (Linux; a Windows section cannot be emptied) and
// the next request builds it again, a few seconds later.
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
	datasetMapBytes  = (rxDatasetSize + hugePage - 1) / hugePage * hugePage
	// RandomXSharedBytes is the size of the shared dataset file.
	RandomXSharedBytes = sharedHeaderSize + datasetMapBytes
	// RandomXFastHostBytes is the address space a worker needs to map it.
	RandomXFastHostBytes = RandomXSharedBytes + 64<<20

	rxErrNoFast = -4 // this node has no shared dataset

)

// InheritedHandlesEnv names, on Windows, the handles a worker inherited (see
// isolate.inheritFiles); the shared dataset is the first. On Linux the
// dataset is descriptor 3.
const InheritedHandlesEnv = "RAKKAN_INHERITED_HANDLES"

// RandomXSharedSupported reports whether this build can offer RandomX: it
// links librandomx and has shared memory for the workers.
func RandomXSharedSupported() bool {
	return rxlibAvailable && (runtime.GOOS == "linux" || runtime.GOOS == "windows")
}

const (
	offMagic   = 0
	offGen     = 8
	offSeedLen = 16
	offSeed    = 24
	offWantLen = 96
	offWant    = 104
	offLastUse = 168
	seedMax    = 64
)

var sharedMagic = [8]byte{'R', 'X', 'D', 'S', 'E', 'T', '0', '1'}

type rxView struct {
	hdr  []byte // header page, read-write
	data []byte // the dataset
	ds   *rxlibDataset
}

func newView(hdr, data []byte) (*rxView, error) {
	ds, err := newRxlibDataset(data)
	if err != nil {
		return nil, err
	}
	return &rxView{hdr: hdr, data: data, ds: ds}, nil
}

func (v *rxView) gen() *uint64 { return (*uint64)(unsafe.Pointer(&v.hdr[offGen])) }

// touch records that a hash was just asked for or finished.
func (v *rxView) touch() {
	atomic.StoreInt64((*int64)(unsafe.Pointer(&v.hdr[offLastUse])), time.Now().UnixNano())
}

func (v *rxView) lastUse() time.Time {
	return time.Unix(0, atomic.LoadInt64((*int64)(unsafe.Pointer(&v.hdr[offLastUse]))))
}

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

// rxOpen returns a VM for the seed. done releases it and reports whether its
// results can be trusted (the dataset did not change under it). A negative
// code is returned to the task instead.
func rxOpen(s *execState, seed []byte) (vm *rxlibVM, done func() bool, code int32) {
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
	v.touch()
	vm, err := newRxlibVM(v.ds)
	if err != nil {
		s.trap(fmt.Errorf("%w: randomx vm: %v", ErrTrap, err))
	}
	return vm, func() bool {
		vm.close()
		v.touch()
		return atomic.LoadUint64(v.gen()) == g
	}, 0
}

// RandomXShared is the node's copy of the dataset.
type RandomXShared struct {
	f *os.File
	// Huge reports that the dataset sits in huge pages.
	Huge bool
	// IdleRelease is how long the built dataset is kept without a request
	// before its memory is returned to the system; zero keeps it for good.
	// Set it before Serve.
	IdleRelease time.Duration
	v           *rxView
	noRelease   bool // releasing failed once: do not try again
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
		v, err := newView(mem[:sharedHeaderSize:sharedHeaderSize], mem[sharedHeaderSize:])
		if err != nil {
			unmapShared(mem)
			f.Close()
			return nil, err
		}
		return &RandomXShared{f: f, Huge: huge, v: v}, nil
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
			s.releaseIfIdle(logf)
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
	cache, err := newRxlibCache()
	if err != nil {
		return err
	}
	defer cache.close()
	cache.init(seed)

	atomic.AddUint64(v.gen(), 1) // odd: workers stop trusting the dataset
	binary.LittleEndian.PutUint32(v.hdr[offSeedLen:], 0)
	v.ds.fill(cache, threads)
	v.setField(offSeedLen, offSeed, seed)
	v.touch()
	atomic.AddUint64(v.gen(), 1)
	return nil
}

// releaseIfIdle empties the dataset when nothing has asked for a hash for
// IdleRelease. Workers see the generation change and answer "not ready", the
// same as while a rebuild runs; the next request writes its seed as wanted and
// Serve builds the dataset again.
func (s *RandomXShared) releaseIfIdle(logf func(format string, args ...any)) {
	v := s.v
	if s.IdleRelease <= 0 || s.noRelease || v.seed() == nil || time.Since(v.lastUse()) < s.IdleRelease {
		return
	}
	binary.LittleEndian.PutUint32(v.hdr[offWantLen:], 0) // or Serve would rebuild at once
	atomic.AddUint64(v.gen(), 1)
	binary.LittleEndian.PutUint32(v.hdr[offSeedLen:], 0)
	err := releasePages(s.f, sharedHeaderSize, datasetMapBytes)
	atomic.AddUint64(v.gen(), 1)
	if err != nil {
		s.noRelease = true
		logf("randomx dataset: cannot return its memory, keeping it: %v", err)
		return
	}
	logf("randomx dataset released after %s without use", s.IdleRelease)
}

// Close releases the file; workers keep their own mappings.
func (s *RandomXShared) Close() { s.f.Close() }

// attachShared maps the dataset inherited as descriptor 3, if there is one.
func attachShared() *rxView {
	fd, ok := inheritedShared()
	if !ok {
		return nil
	}
	hdr, err := mapShared(fd, 0, sharedHeaderSize, true)
	if err != nil || !bytes.Equal(hdr[:8], sharedMagic[:]) {
		return nil
	}
	data, err := mapShared(fd, sharedHeaderSize, datasetMapBytes, false)
	if err != nil {
		return nil
	}
	runtime.KeepAlive(hdr)
	v, err := newView(hdr, data)
	if err != nil {
		return nil
	}
	return v
}
