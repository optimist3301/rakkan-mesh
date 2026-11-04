package framing

import (
	"bytes"
	"crypto/rand"
	"errors"
	"io"
	"runtime"
	"testing"
	"testing/iotest"
)

func TestReadFull(t *testing.T) {
	for _, n := range []int{0, 1, 1000, step, step + 1, 5*step + 123} {
		data := make([]byte, n+10)
		rand.Read(data)
		// One byte at a time exercises every growth step.
		var r io.Reader = bytes.NewReader(data)
		if n < 5000 {
			r = iotest.OneByteReader(r)
		}
		got, err := ReadFull(r, n)
		if err != nil || !bytes.Equal(got, data[:n]) {
			t.Fatalf("n=%d: %d bytes, %v", n, len(got), err)
		}
	}
	if _, err := ReadFull(bytes.NewReader(make([]byte, 99)), 100); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("short stream: %v", err)
	}
	// Data and EOF together at exactly n bytes is a complete frame.
	if got, err := ReadFull(iotest.DataErrReader(bytes.NewReader([]byte("abc"))), 3); err != nil || string(got) != "abc" {
		t.Fatalf("got %q, %v", got, err)
	}
	if _, err := ReadFull(bytes.NewReader(nil), -1); err == nil {
		t.Fatal("negative length accepted")
	}
}

func TestReadFullAllocatesWhatArrives(t *testing.T) {
	// A peer announcing 1 GiB and sending 10 bytes costs about a step.
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	if _, err := ReadFull(bytes.NewReader(make([]byte, 10)), 1<<30); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatal(err)
	}
	runtime.ReadMemStats(&after)
	if alloc := after.TotalAlloc - before.TotalAlloc; alloc > 2*step {
		t.Fatalf("allocated %d bytes for 10 received", alloc)
	}
}
