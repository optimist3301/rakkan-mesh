//go:build linux || windows

package isolate

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/sandbox"
)

// A worker hashes with librandomx against the dataset it inherits from the
// node (on Linux as descriptor 3, strictly isolated: namespaces, seccomp, a
// capped address space; on Windows as an inherited handle), and gets the same
// hashes as the node does in process.
func TestPoolRandomXSharedDataset(t *testing.T) {
	if !sandbox.RandomXAvailable() {
		t.Skip("this build has no RandomX")
	}
	mod, err := os.ReadFile("../../../rakkan-rx/task/randomx_fast_task.wasm")
	if err != nil {
		t.Skip("randomx task not built:", err)
	}
	sh, err := sandbox.NewRandomXShared()
	if err != nil {
		t.Skip("no shared dataset here:", err)
	}
	defer sh.Close()
	ctx, stop := context.WithCancel(t.Context())
	defer stop()
	go sh.Serve(ctx, runtime.NumCPU(), t.Logf)
	sh.Enable()

	p := newRandomXPool(t, Config{
		HostExtraBytes: sandbox.RandomXFastHostBytes,
		ExtraFiles:     []*os.File{sh.File()},
	})
	seed, blob := []byte("test key 000"), bytes.Repeat([]byte{0xAB}, 76)
	var job []byte
	for _, v := range []uint32{39, 7, 6, 8} { // nonce offset, start, count, max records
		job = binary.LittleEndian.AppendUint32(job, v)
	}
	job = binary.LittleEndian.AppendUint64(job, math.MaxUint64)
	job = binary.LittleEndian.AppendUint32(job, uint32(len(seed)))
	job = binary.LittleEndian.AppendUint32(job, uint32(len(blob)))
	job = append(append(job, seed...), blob...)

	lim := sandbox.DefaultLimits()
	t0 := time.Now()
	for {
		res, err := p.Run(t.Context(), "", sandbox.Task{Module: mod, Input: job, Limits: lim})
		if err != nil {
			t.Fatal(err)
		}
		if res.ExitCode == 3 { // dataset not built yet
			if time.Since(t0) > 90*time.Second {
				t.Fatal("dataset never became ready")
			}
			time.Sleep(200 * time.Millisecond)
			continue
		}
		if res.ExitCode != 0 || len(res.Output) != 6*(4+32) {
			t.Fatalf("exit %d, %d output bytes", res.ExitCode, len(res.Output))
		}
		// The in-process node hashes the same nonces the same way.
		local := sandbox.NewRunner(lim)
		defer local.Close(t.Context())
		want, err := local.Run(t.Context(), sandbox.Task{Module: mod, Input: job, Limits: lim})
		if err != nil || want.ExitCode != 0 {
			t.Fatalf("in process: %v %+v", err, want)
		}
		if !bytes.Equal(res.Output, want.Output) || res.GasUsed != want.GasUsed {
			t.Fatalf("worker and node disagree")
		}
		return
	}
}
