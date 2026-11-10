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
)

// rxTask is the C task from rakkan-rx, built with clang --target=wasm32. The
// tests are skipped where that checkout is absent.
const rxTaskPath = "../../../rakkan-rx/task/randomx_fast_task.wasm"

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

// withShared runs a dataset like the daemon's and returns the task.
func withShared(t *testing.T) (task []byte, sh *RandomXShared) {
	t.Helper()
	needRxlib(t, 1)
	task, err := os.ReadFile(rxTaskPath)
	if err != nil {
		t.Skip("randomx task not built:", err)
	}
	sh, err = NewRandomXShared()
	if err != nil {
		t.Skip("no shared dataset here:", err)
	}
	t.Cleanup(sh.Close)
	ctx, stop := context.WithCancel(context.Background())
	t.Cleanup(stop)
	go sh.Serve(ctx, runtime.NumCPU(), t.Logf)
	sh.Enable()
	t.Cleanup(func() {
		rxShared.mu.Lock()
		rxShared.v, rxShared.enabled = nil, false
		rxShared.mu.Unlock()
	})
	if !slices.Contains(Capabilities(), RandomXFastCapability) {
		t.Fatal("capability not offered by a node with a dataset")
	}
	return task, sh
}

// runReady runs the job until the dataset is built for its seed (exit code 3
// is "not ready, retry").
func runReady(t *testing.T, task, job []byte) *Result {
	t.Helper()
	t0 := time.Now()
	for {
		res, err := run(t, task, job, DefaultLimits())
		if err != nil {
			t.Fatal(err)
		}
		if res.ExitCode != 3 {
			t.Logf("ready after %s; ran %s, %d gas", time.Since(t0).Round(time.Millisecond), res.Elapsed, res.GasUsed)
			return res
		}
		if time.Since(t0) > 90*time.Second {
			t.Fatal("dataset never became ready")
		}
		time.Sleep(200 * time.Millisecond)
	}
}

func TestRandomXSearchMatchesReference(t *testing.T) {
	task, sh := withShared(t)
	blob := bytes.Repeat([]byte{0xAB}, 76)
	const off, start, count = 39, 1000, 24
	for _, seed := range [][]byte{[]byte("test key 000"), []byte("another seed")} {
		res := runReady(t, task, rxJob(seed, blob, off, start, count, 32, math.MaxUint64))
		if res.ExitCode != 0 || len(res.Output) != count*randomxRecord {
			t.Fatalf("exit %d, %d output bytes", res.ExitCode, len(res.Output))
		}
		// The dataset is built for seed by now; hash straight through the
		// library (checked against the RandomX vectors by TestRxlibVectors).
		vm, err := newRxlibVM(sh.v.ds)
		if err != nil {
			t.Fatal(err)
		}
		for i := range count {
			b := bytes.Clone(blob)
			binary.LittleEndian.PutUint32(b[off:], start+uint32(i))
			var want [rxHashSize]byte
			vm.hash(b, &want)
			rec := res.Output[i*randomxRecord:][:randomxRecord]
			if n := binary.LittleEndian.Uint32(rec); n != start+uint32(i) || !bytes.Equal(rec[4:], want[:]) {
				t.Fatalf("seed %q record %d: nonce %d hash %x, want %d %x", seed, i, n, rec[4:], start+uint32(i), want)
			}
		}
		vm.close()
		if res.GasUsed < count*randomxFastHashGas {
			t.Fatalf("gas %d for %d hashes", res.GasUsed, count)
		}

		// A tight target keeps fewer records, and only ones that pass.
		const mid uint64 = math.MaxUint64 / 2
		res2 := runReady(t, task, rxJob(seed, blob, off, start, count, 32, mid))
		if len(res2.Output) >= len(res.Output) {
			t.Fatalf("target did not filter: %d of %d records", len(res2.Output)/randomxRecord, count)
		}
		for i := 0; i+randomxRecord <= len(res2.Output); i += randomxRecord {
			if binary.LittleEndian.Uint64(res2.Output[i+4+24:]) >= mid {
				t.Fatal("record above target")
			}
		}
	}
	// The first seed's dataset is gone: asking for it again is "not ready".
	res, _ := run(t, task, rxJob([]byte("test key 000"), blob, off, 0, 1, 1, 0), DefaultLimits())
	if res.ExitCode != 3 {
		t.Fatalf("stale seed: exit %d, want 3", res.ExitCode)
	}
}

func TestRandomXBilledAndBounded(t *testing.T) {
	task, _ := withShared(t)
	seed, blob := []byte("test key 000"), bytes.Repeat([]byte{1}, 76)
	runReady(t, task, rxJob(seed, blob, 39, 0, 1, 1, 0)) // build the dataset
	// 400 hashes do not fit the default 5e9 gas.
	if _, err := run(t, task, rxJob(seed, blob, 39, 0, 400, 1, 0), DefaultLimits()); !errors.Is(err, ErrOutOfGas) {
		t.Fatalf("400 hashes: %v, want out of gas", err)
	}
	// An empty seed is refused by the kernel.
	res, err := run(t, task, rxJob(nil, blob, 39, 0, 1, 1, 0), DefaultLimits())
	if err != nil || res.ExitCode != 1 {
		t.Fatalf("empty seed: %v %+v", err, res)
	}
}
