package sandbox

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"sync"

	randomx "git.gammaspectra.live/P2Pool/go-randomx/v4"
	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
	"golang.org/x/sys/cpu"
)

// RandomX kernels: the proof-of-work hash of Monero, run as host code. A
// task cannot hold the 256 MiB cache RandomX needs in its linear memory (the
// ceiling is 64 MiB) and could not compute it within its gas, so the node
// keeps one cache per worker process, built on first use for a given seed
// and shared by later tasks. Hashing is light mode: the cache plus a 2 MiB
// scratchpad per call, no 2 GiB dataset.
//
// Hashes are deterministic whatever the CPU and whether or not the JIT is
// available, so replicated verification works. Building the cache is not
// billed in gas (gas must not depend on what an earlier task left behind);
// the wall-clock timeout covers it.
const (
	RandomXModule     = "rakkan_randomx_v1"
	RandomXCapability = "randomx/1"
)

// RandomXHostBytes is the memory a worker needs on top of the task's own for
// the cache, the scratchpad and the JIT.
const RandomXHostBytes = 320 << 20

// randomxHashGas bills one hash: about 25 ms of a modern core at the gas
// calibration of 0.15 ns per unit.
const randomxHashGas = 160_000_000

// Search results are records of a nonce (u32 LE) and its hash.
const randomxRecord = 4 + randomx.RANDOMX_HASH_SIZE

const (
	rxErrArgs = -1 // bad lengths or arguments
	rxErrSeed = -2 // seed empty or too long
	// rxErrNotReady: the fast kernel's dataset is not built for this seed
	// (it is being built now) or changed during the call. Not a task
	// failure: try again shortly.
	rxErrNotReady = -3
)

var randomxABI = map[string]signature{
	// hash(seed, seed_len, in, in_len, out) -> 0, or a negative error;
	// writes the 32-byte hash of in under the key seed to out.
	"hash": sig([]api.ValueType{i32, i32, i32, i32, i32}, i32),
	// search(seed, seed_len, blob, blob_len, nonce_off, start, count,
	//        target i64, out, out_max) -> records found, or a negative error.
	// Hashes blob with its u32 LE nonce at nonce_off set to start,
	// start+1, ... (count of them) and writes a record (nonce u32 LE, hash)
	// to out for each whose hash passes target: the u64 LE at hash[24:32]
	// is below target. Stops when out_max records are written.
	"search": sig([]api.ValueType{i32, i32, i32, i32, i32, i32, i32, i64, i32, i32}, i32),
}

// rxCache is the cache of the current seed, replaced when the seed changes.
var rxCache struct {
	mu  sync.RWMutex
	key []byte
	c   *randomx.Cache
	ok  bool
}

func randomxFlags() randomx.Flags {
	f := randomx.RANDOMX_FLAG_JIT | randomx.RANDOMX_FLAG_ARGON2
	if cpu.X86.HasAES || cpu.ARM64.HasAES {
		f |= randomx.RANDOMX_FLAG_HARD_AES
	}
	return f
}

// acquireRandomX returns the cache for key with a read lock held; the caller
// releases it with rxCache.mu.RUnlock.
func acquireRandomX(key []byte) (*randomx.Cache, error) {
	for {
		rxCache.mu.RLock()
		if rxCache.ok && bytes.Equal(rxCache.key, key) {
			return rxCache.c, nil
		}
		rxCache.mu.RUnlock()

		rxCache.mu.Lock()
		if !(rxCache.ok && bytes.Equal(rxCache.key, key)) {
			// A new seed re-initializes the same cache in place: freeing and
			// mapping 256 MiB again inside a worker whose address space is capped
			// can fail, and the memory is needed anyway.
			c := rxCache.c
			rxCache.ok = false
			if c == nil {
				var err error
				if c, err = randomx.NewCache(randomxFlags()); err != nil {
					rxCache.mu.Unlock()
					return nil, err
				}
			}
			c.Init(key)
			rxCache.c, rxCache.key, rxCache.ok = c, bytes.Clone(key), true
		}
		rxCache.mu.Unlock()
	}
}

// rxBackend is a way of running hashes: light mode on the worker's own
// cache, or fast mode on the dataset the node shares between its workers.
type rxBackend struct {
	gas int64 // per hash
	// open returns a VM for the seed. done releases it and reports whether
	// its results can be trusted (the fast dataset did not change under it).
	// A negative code is returned to the task instead.
	open func(s *execState, seed []byte) (vm *randomx.VM, done func() bool, code int32)
}

var rxLight = &rxBackend{gas: randomxHashGas, open: func(s *execState, seed []byte) (*randomx.VM, func() bool, int32) {
	cache, err := acquireRandomX(seed)
	if err != nil {
		s.trap(fmt.Errorf("%w: randomx cache: %v", ErrTrap, err))
	}
	vm, err := randomx.NewVM(randomxFlags(), cache, nil)
	if err != nil {
		rxCache.mu.RUnlock()
		s.trap(fmt.Errorf("%w: randomx vm: %v", ErrTrap, err))
	}
	return vm, func() bool { vm.Close(); rxCache.mu.RUnlock(); return true }, 0
}}

func (s *execState) instantiateRandomX(ctx context.Context, rt wazero.Runtime) error {
	for _, m := range []struct {
		name string
		be   *rxBackend
	}{{RandomXModule, rxLight}, {RandomXFastModule, rxFast}} {
		_, err := rt.NewHostModuleBuilder(m.name).
			NewFunctionBuilder().WithFunc(func(ctx context.Context, mod api.Module, a, b, c, d, e uint32) int32 {
			return s.rxHash(m.be, mod, a, b, c, d, e)
		}).Export("hash").
			NewFunctionBuilder().WithFunc(func(ctx context.Context, mod api.Module, a, b, c, d, e, f, g uint32, target uint64, o, om uint32) int32 {
			return s.rxSearch(m.be, mod, a, b, c, d, e, f, g, target, o, om)
		}).Export("search").
			Instantiate(ctx)
		if err != nil {
			return err
		}
	}
	return nil
}

// rxSeed reads and validates the key; a negative result is an error code.
func (s *execState) rxSeed(m api.Module, ptr, n uint32) ([]byte, int32) {
	if n == 0 || n > 60 {
		return nil, rxErrSeed
	}
	b, ok := m.Memory().Read(ptr, n)
	if !ok {
		s.trap(fmt.Errorf("%w: randomx seed ptr=%d len=%d", ErrMemoryAccess, ptr, n))
	}
	return bytes.Clone(b), 0
}

func (s *execState) rxHash(be *rxBackend, m api.Module, seedPtr, seedLen, inPtr, inLen, outPtr uint32) int32 {
	s.charge(inLen)
	seed, code := s.rxSeed(m, seedPtr, seedLen)
	if code != 0 {
		return code
	}
	if inLen > 1<<16 {
		return rxErrArgs
	}
	in, ok := m.Memory().Read(inPtr, inLen)
	if !ok {
		s.trap(fmt.Errorf("%w: randomx in ptr=%d len=%d", ErrMemoryAccess, inPtr, inLen))
	}
	in = bytes.Clone(in)
	s.chargeUnits(be.gas)

	vm, done, code := be.open(s, seed)
	if code != 0 {
		return code
	}
	release := once(done)
	defer release()
	var out [randomx.RANDOMX_HASH_SIZE]byte
	vm.CalculateHash(in, &out)
	if !release() {
		return rxErrNotReady
	}
	if !m.Memory().Write(outPtr, out[:]) {
		s.trap(fmt.Errorf("%w: randomx out ptr=%d", ErrMemoryAccess, outPtr))
	}
	return 0
}

func (s *execState) rxSearch(be *rxBackend, m api.Module, seedPtr, seedLen, blobPtr, blobLen, nonceOff, start, count uint32, target uint64, outPtr, outMax uint32) int32 {
	s.charge(blobLen)
	seed, code := s.rxSeed(m, seedPtr, seedLen)
	if code != 0 {
		return code
	}
	blobRO, ok := m.Memory().Read(blobPtr, blobLen)
	if !ok {
		s.trap(fmt.Errorf("%w: randomx blob ptr=%d len=%d", ErrMemoryAccess, blobPtr, blobLen))
	}
	if blobLen < 4 || blobLen > 1<<16 || uint64(nonceOff)+4 > uint64(blobLen) || outMax == 0 || uint64(outMax)*randomxRecord > 1<<24 {
		return rxErrArgs
	}
	blob := bytes.Clone(blobRO)

	vm, done, code := be.open(s, seed)
	if code != 0 {
		return code
	}
	release := once(done) // a trap (out of gas) must not leave the cache locked
	defer release()
	var found []byte
	var h [randomx.RANDOMX_HASH_SIZE]byte
	for i := uint32(0); i < count && uint32(len(found)/randomxRecord) < outMax; i++ {
		s.chargeUnits(be.gas) // billed before the work, so a bad budget stops early
		nonce := start + i
		binary.LittleEndian.PutUint32(blob[nonceOff:], nonce)
		vm.CalculateHash(blob, &h)
		if binary.LittleEndian.Uint64(h[24:]) < target {
			found = binary.LittleEndian.AppendUint32(found, nonce)
			found = append(found, h[:]...)
		}
	}
	if !release() {
		return rxErrNotReady
	}
	if len(found) > 0 && !m.Memory().Write(outPtr, found) {
		s.trap(fmt.Errorf("%w: randomx out ptr=%d len=%d", ErrMemoryAccess, outPtr, len(found)))
	}
	return int32(len(found) / randomxRecord)
}

// once makes done run only the first time and return that first verdict.
func once(done func() bool) func() bool {
	var ran, ok bool
	return func() bool {
		if !ran {
			ran, ok = true, done()
		}
		return ok
	}
}
