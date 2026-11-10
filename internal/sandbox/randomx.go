package sandbox

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
)

// RandomX kernel: the proof-of-work hash of Monero, run as host code, in
// fast mode only (see randomx_fast.go for the dataset it hashes against and
// for the module and capability names).
//
// Hashes are deterministic whatever the CPU and whether or not the JIT is
// available, so replicated verification works. Building the dataset is not
// billed in gas (gas must not depend on what an earlier task left behind);
// the wall-clock timeout covers the wait.

// Search results are records of a nonce (u32 LE) and its hash.
const randomxRecord = 4 + rxHashSize

const (
	rxErrArgs = -1 // bad lengths or arguments
	rxErrSeed = -2 // seed empty or too long
	// rxErrNotReady: the dataset is not built for this seed (it is being
	// built now) or changed during the call. Not a task failure: try again
	// shortly.
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

func (s *execState) instantiateRandomX(ctx context.Context, rt wazero.Runtime) error {
	_, err := rt.NewHostModuleBuilder(RandomXFastModule).
		NewFunctionBuilder().WithFunc(func(ctx context.Context, mod api.Module, a, b, c, d, e uint32) int32 {
		return s.rxHash(mod, a, b, c, d, e)
	}).Export("hash").
		NewFunctionBuilder().WithFunc(func(ctx context.Context, mod api.Module, a, b, c, d, e, f, g uint32, target uint64, o, om uint32) int32 {
		return s.rxSearch(mod, a, b, c, d, e, f, g, target, o, om)
	}).Export("search").
		Instantiate(ctx)
	return err
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

func (s *execState) rxHash(m api.Module, seedPtr, seedLen, inPtr, inLen, outPtr uint32) int32 {
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
	s.chargeUnits(randomxFastHashGas)

	vm, done, code := rxOpen(s, seed)
	if code != 0 {
		return code
	}
	release := once(done)
	defer release()
	var out [rxHashSize]byte
	vm.hash(in, &out)
	if !release() {
		return rxErrNotReady
	}
	if !m.Memory().Write(outPtr, out[:]) {
		s.trap(fmt.Errorf("%w: randomx out ptr=%d", ErrMemoryAccess, outPtr))
	}
	return 0
}

func (s *execState) rxSearch(m api.Module, seedPtr, seedLen, blobPtr, blobLen, nonceOff, start, count uint32, target uint64, outPtr, outMax uint32) int32 {
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

	vm, done, code := rxOpen(s, seed)
	if code != 0 {
		return code
	}
	release := once(done) // a trap (out of gas) must not leave the cache locked
	defer release()
	var found []byte
	var h [rxHashSize]byte
	for i := uint32(0); i < count && uint32(len(found)/randomxRecord) < outMax; i++ {
		s.chargeUnits(randomxFastHashGas) // billed before the work, so a bad budget stops early
		nonce := start + i
		binary.LittleEndian.PutUint32(blob[nonceOff:], nonce)
		vm.hash(blob, &h)
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
