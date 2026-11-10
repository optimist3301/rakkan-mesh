package sandbox

import (
	"context"
	"os"
	"runtime"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
)

func fileBytes(t *testing.T, f *os.File) int64 {
	t.Helper()
	var st syscall.Stat_t
	if err := syscall.Fstat(int(f.Fd()), &st); err != nil {
		t.Fatal(err)
	}
	return st.Blocks * 512
}

func waitFor(t *testing.T, what string, ok func() bool) {
	t.Helper()
	for t0 := time.Now(); !ok(); time.Sleep(50 * time.Millisecond) {
		if time.Since(t0) > 60*time.Second {
			t.Fatal("timed out waiting for", what)
		}
	}
}

// An idle dataset gives its memory back, and the next request builds it again.
func TestRandomXReleasedWhenIdle(t *testing.T) {
	needRxlib(t, 1)
	sh, err := NewRandomXShared()
	if err != nil {
		t.Skip("no shared dataset here:", err)
	}
	defer sh.Close()
	sh.IdleRelease = 2 * time.Second
	ctx, stop := context.WithCancel(t.Context())
	defer stop()
	go sh.Serve(ctx, runtime.NumCPU(), t.Logf)

	seed := []byte("test key 000")
	hash := func() string {
		vm, err := newRxlibVM(sh.v.ds)
		if err != nil {
			t.Fatal(err)
		}
		defer vm.close()
		return rxlibHash(t, vm, []byte("This is a test"))
	}
	if b := fileBytes(t, sh.f); b > 64<<20 {
		t.Fatalf("%d MiB used before any request", b>>20)
	}

	sh.v.setField(offWantLen, offWant, seed)
	waitFor(t, "the dataset", func() bool { return string(sh.v.seed()) == string(seed) })
	if b := fileBytes(t, sh.f); b < 2<<30 {
		t.Fatalf("only %d MiB used by a built dataset", b>>20)
	}
	if got := hash(); got != rxVectors[1].hash {
		t.Fatalf("hash %s", got)
	}

	// A request now and then keeps it.
	for range 4 {
		time.Sleep(time.Second)
		sh.v.touch()
	}
	if sh.v.seed() == nil {
		t.Fatal("released while in use")
	}

	// Idle: the memory goes back and the generation stays even (usable).
	waitFor(t, "the release", func() bool { return sh.v.seed() == nil && atomic.LoadUint64(sh.v.gen())&1 == 0 })
	if b := fileBytes(t, sh.f); b > 64<<20 {
		t.Fatalf("%d MiB still used after the release", b>>20)
	}
	time.Sleep(time.Second) // it must not rebuild by itself
	if sh.v.seed() != nil {
		t.Fatal("rebuilt without a request")
	}

	// The next request builds it again, with the same hashes.
	sh.v.setField(offWantLen, offWant, seed)
	waitFor(t, "the rebuild", func() bool { return string(sh.v.seed()) == string(seed) })
	if got := hash(); got != rxVectors[1].hash {
		t.Fatalf("hash after the rebuild %s", got)
	}
}
