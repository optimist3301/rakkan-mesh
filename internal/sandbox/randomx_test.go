package sandbox

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"runtime"
	"slices"
	"testing"
	"time"

	randomx "git.gammaspectra.live/P2Pool/go-randomx/v4"
)

// rxTask is the C task from rakkan-rx, built with clang --target=wasm32. The
// test is skipped where that checkout is absent.
const rxTaskPath = "../../../rakkan-rx/task/randomx_task.wasm"

func rxJob(seed, blob []byte, nonceOff, start, count, outMax uint32, target uint64) []byte {
	var b []byte
	for _, v := range []uint32{nonceOff, start, count, outMax} {
		b = binary.LittleEndian.AppendUint32(b, v)
	}
	b = binary.LittleEndian.AppendUint64(b, target)
	b = binary.LittleEndian.AppendUint32(b, uint32(len(seed)))
	b = binary.LittleEndian.AppendUint32(b, uint32(len(blob)))
	return append(append(b, seed...), blob...)
}

func TestRandomXSearchMatchesReference(t *testing.T) {
	mod, err := os.ReadFile(rxTaskPath)
	if err != nil {
		t.Skip("randomx task not built:", err)
	}
	seed := []byte("test key 000")
	blob := bytes.Repeat([]byte{0xAB}, 76)
	const off, start, count = 39, 1000, 4

	res, err := run(t, mod, rxJob(seed, blob, off, start, count, 8, math.MaxUint64), DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	if res.ExitCode != 0 || len(res.Output) != count*randomxRecord {
		t.Fatalf("exit %d, %d output bytes", res.ExitCode, len(res.Output))
	}

	c, err := randomx.NewCache(randomx.RANDOMX_FLAG_DEFAULT) // the reference runs the interpreter, not the JIT
	if err != nil {
		t.Fatal(err)
	}
	c.Init(seed)
	vm, _ := randomx.NewVM(randomx.RANDOMX_FLAG_DEFAULT, c, nil)
	for i := range count {
		b := bytes.Clone(blob)
		binary.LittleEndian.PutUint32(b[off:], start+uint32(i))
		var want [32]byte
		vm.CalculateHash(b, &want)
		rec := res.Output[i*randomxRecord:][:randomxRecord]
		if n := binary.LittleEndian.Uint32(rec); n != start+uint32(i) || !bytes.Equal(rec[4:], want[:]) {
			t.Fatalf("record %d: nonce %d hash %x, want %d %x", i, n, rec[4:], start+uint32(i), want)
		}
	}
	if res.GasUsed < count*randomxHashGas {
		t.Fatalf("gas %d for %d hashes", res.GasUsed, count)
	}

	// A tight target keeps fewer records, and only ones that pass.
	var mid uint64 = math.MaxUint64 / 2
	res2, err := run(t, mod, rxJob(seed, blob, off, start, count, 8, mid), DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i+randomxRecord <= len(res2.Output); i += randomxRecord {
		if binary.LittleEndian.Uint64(res2.Output[i+4+24:]) >= mid {
			t.Fatal("record above target")
		}
	}
}

func TestRandomXBilledAndBounded(t *testing.T) {
	mod, err := os.ReadFile(rxTaskPath)
	if err != nil {
		t.Skip("randomx task not built:", err)
	}
	seed, blob := []byte("test key 000"), bytes.Repeat([]byte{1}, 76)
	// 40 hashes do not fit the default 5e9 gas.
	if _, err := run(t, mod, rxJob(seed, blob, 39, 0, 40, 1, 0), DefaultLimits()); !errors.Is(err, ErrOutOfGas) {
		t.Fatalf("40 hashes: %v, want out of gas", err)
	}
	// An empty seed is refused by the kernel.
	res, err := run(t, mod, rxJob(nil, blob, 39, 0, 1, 1, 0), DefaultLimits())
	if err != nil || res.ExitCode != 1 {
		t.Fatalf("empty seed: %v %+v", err, res)
	}
}

func TestFastMatchesLight(t *testing.T) {
	light, err := os.ReadFile(rxTaskPath)
	if err != nil {
		t.Skip("randomx task not built:", err)
	}
	fast, err := os.ReadFile("../../../rakkan-rx/task/randomx_fast_task.wasm")
	if err != nil {
		t.Skip("randomx fast task not built:", err)
	}
	sh, err := NewRandomXShared()
	if err != nil {
		t.Skip("no shared dataset here:", err)
	}
	defer sh.Close()
	ctx, stop := context.WithCancel(context.Background())
	defer stop()
	go sh.Serve(ctx, runtime.NumCPU(), t.Logf)
	sh.Enable()
	defer func() {
		rxShared.mu.Lock()
		rxShared.v, rxShared.enabled = nil, false
		rxShared.mu.Unlock()
	}()
	if !slices.Contains(Capabilities(), RandomXFastCapability) {
		t.Fatal("capability not offered by a node with a dataset")
	}

	blob := bytes.Repeat([]byte{0xAB}, 76)
	for _, seed := range [][]byte{[]byte("test key 000"), []byte("another seed")} {
		job := rxJob(seed, blob, 39, 5000, 24, 8, math.MaxUint64)
		want, err := run(t, light, job, DefaultLimits())
		if err != nil {
			t.Fatal(err)
		}
		t0 := time.Now()
		var got *Result
		for {
			got, err = run(t, fast, job, DefaultLimits())
			if err != nil {
				t.Fatal(err)
			}
			if got.ExitCode != 3 {
				break
			}
			if time.Since(t0) > 90*time.Second {
				t.Fatal("dataset never became ready")
			}
			time.Sleep(200 * time.Millisecond)
		}
		t.Logf("seed %q ready after %s; 24 hashes took %s (%d gas)", seed, time.Since(t0).Round(time.Millisecond), got.Elapsed, got.GasUsed)
		if got.ExitCode != 0 || !bytes.Equal(got.Output, want.Output) {
			t.Fatalf("fast exit %d, output differs from light mode", got.ExitCode)
		}
	}
	// The first seed's dataset is gone: asking for it again is "not ready".
	res, _ := run(t, fast, rxJob([]byte("test key 000"), blob, 39, 0, 1, 1, 0), DefaultLimits())
	if res.ExitCode != 3 {
		t.Fatalf("stale seed: exit %d, want 3", res.ExitCode)
	}
}
