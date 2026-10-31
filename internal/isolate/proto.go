package isolate

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/sandbox"
)

// Wire protocol between the node and a worker over the worker's stdin and
// stdout: length-prefixed frames, one request followed by one response.
// The node treats responses as untrusted: a worker may be compromised.

const protoVersion = 1

// maxErrorMessage bounds error text accepted from a worker.
const maxErrorMessage = 4 << 10

func writeFrame(w io.Writer, payload []byte) error {
	var hdr [4]byte
	binary.BigEndian.PutUint32(hdr[:], uint32(len(payload)))
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	_, err := w.Write(payload)
	return err
}

func readFrame(r io.Reader, limit int) ([]byte, error) {
	var hdr [4]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return nil, err
	}
	n := binary.BigEndian.Uint32(hdr[:])
	if uint64(n) > uint64(limit) {
		return nil, fmt.Errorf("frame of %d bytes exceeds limit %d", n, limit)
	}
	buf := make([]byte, n)
	_, err := io.ReadFull(r, buf)
	return buf, err
}

// errorKinds maps sandbox errors to wire codes. Index 0 means success; the
// order is part of the protocol.
var errorKinds = []error{
	nil,
	sandbox.ErrLimitPolicy,
	sandbox.ErrModuleTooLarge,
	sandbox.ErrInputTooLarge,
	sandbox.ErrInvalidModule,
	sandbox.ErrForbiddenImport,
	sandbox.ErrMissingEntry,
	sandbox.ErrOutputLimit,
	sandbox.ErrMemoryAccess,
	sandbox.ErrOutOfGas,
	sandbox.ErrTimeout,
	sandbox.ErrTrap,
	context.Canceled,
}

const codeInternal = 255

func errorCode(err error) uint8 {
	for i, kind := range errorKinds[1:] {
		if errors.Is(err, kind) {
			return uint8(i + 1)
		}
	}
	return codeInternal
}

// remoteError carries a worker's error message while matching the sandbox
// sentinel error with errors.Is.
type remoteError struct {
	kind error
	msg  string
}

func (e *remoteError) Error() string { return e.msg }
func (e *remoteError) Unwrap() error { return e.kind }

type encoder struct{ b []byte }

func (e *encoder) u8(v uint8)   { e.b = append(e.b, v) }
func (e *encoder) u32(v uint32) { e.b = binary.BigEndian.AppendUint32(e.b, v) }
func (e *encoder) u64(v uint64) { e.b = binary.BigEndian.AppendUint64(e.b, v) }
func (e *encoder) bytes(v []byte) {
	e.u32(uint32(len(v)))
	e.b = append(e.b, v...)
}

type decoder struct {
	b   []byte
	err error
}

func (d *decoder) take(n uint64) []byte {
	if d.err != nil {
		return nil
	}
	if n > uint64(len(d.b)) {
		d.err = errors.New("truncated message")
		return nil
	}
	v := d.b[:n]
	d.b = d.b[n:]
	return v
}

func (d *decoder) u8() uint8 {
	if v := d.take(1); v != nil {
		return v[0]
	}
	return 0
}

func (d *decoder) u32() uint32 {
	if v := d.take(4); v != nil {
		return binary.BigEndian.Uint32(v)
	}
	return 0
}

func (d *decoder) u64() uint64 {
	if v := d.take(8); v != nil {
		return binary.BigEndian.Uint64(v)
	}
	return 0
}

func (d *decoder) bytes(limit int) []byte {
	n := d.u32()
	if d.err == nil && uint64(n) > uint64(limit) {
		d.err = fmt.Errorf("field of %d bytes exceeds limit %d", n, limit)
	}
	return d.take(uint64(n))
}

func (d *decoder) finish() error {
	if d.err == nil && len(d.b) > 0 {
		d.err = errors.New("trailing bytes in message")
	}
	return d.err
}

func encodeRequest(t sandbox.Task) []byte {
	e := &encoder{}
	e.u8(protoVersion)
	l := t.Limits
	for _, v := range []int64{int64(l.MaxModuleBytes), int64(l.MaxMemoryPages), int64(l.MaxInputBytes),
		int64(l.MaxOutputBytes), int64(l.MaxLogBytes), l.MaxGas, int64(l.Timeout)} {
		e.u64(uint64(v))
	}
	e.bytes([]byte(t.Entry))
	e.bytes(t.Module)
	e.bytes(t.Input)
	return e.b
}

func decodeRequest(b []byte) (sandbox.Task, error) {
	d := &decoder{b: b}
	if v := d.u8(); d.err == nil && v != protoVersion {
		return sandbox.Task{}, fmt.Errorf("protocol version %d, want %d", v, protoVersion)
	}
	var t sandbox.Task
	t.Limits = sandbox.Limits{
		MaxModuleBytes: int(int64(d.u64())),
		MaxMemoryPages: uint32(d.u64()),
		MaxInputBytes:  int(int64(d.u64())),
		MaxOutputBytes: int(int64(d.u64())),
		MaxLogBytes:    int(int64(d.u64())),
		MaxGas:         int64(d.u64()),
		Timeout:        time.Duration(d.u64()),
	}
	t.Entry = string(d.bytes(1 << 10))
	t.Module = d.bytes(len(b))
	t.Input = d.bytes(len(b))
	return t, d.finish()
}

func encodeResponse(res *sandbox.Result, err error) []byte {
	e := &encoder{}
	e.u8(protoVersion)
	if err != nil {
		msg := err.Error()
		if len(msg) > maxErrorMessage {
			msg = msg[:maxErrorMessage]
		}
		e.u8(errorCode(err))
		e.bytes([]byte(msg))
		return e.b
	}
	e.u8(0)
	e.u32(res.ExitCode)
	e.u64(uint64(res.GasUsed))
	e.u64(uint64(res.Elapsed))
	if res.LogTruncated {
		e.u8(1)
	} else {
		e.u8(0)
	}
	e.bytes(res.Output)
	e.bytes(res.Logs)
	return e.b
}

// decodeResponse parses and validates a worker's response against the
// limits of the task it was given. Violations indicate a misbehaving worker.
func decodeResponse(b []byte, lim sandbox.Limits) (*sandbox.Result, error, error) {
	d := &decoder{b: b}
	if v := d.u8(); d.err == nil && v != protoVersion {
		return nil, nil, fmt.Errorf("protocol version %d, want %d", v, protoVersion)
	}
	code := d.u8()
	if code != 0 {
		msg := string(d.bytes(maxErrorMessage))
		if err := d.finish(); err != nil {
			return nil, nil, err
		}
		kind := ErrWorkerInternal
		if int(code) < len(errorKinds) {
			kind = errorKinds[code]
		} else if code != codeInternal {
			return nil, nil, fmt.Errorf("unknown error code %d", code)
		}
		return nil, &remoteError{kind: kind, msg: msg}, nil
	}

	res := &sandbox.Result{
		ExitCode: d.u32(),
		GasUsed:  int64(d.u64()),
		Elapsed:  time.Duration(d.u64()),
	}
	res.LogTruncated = d.u8() != 0
	res.Output = d.bytes(lim.MaxOutputBytes)
	res.Logs = d.bytes(lim.MaxLogBytes)
	if err := d.finish(); err != nil {
		return nil, nil, err
	}
	if res.GasUsed < 0 || res.GasUsed > lim.MaxGas {
		return nil, nil, fmt.Errorf("reported gas %d outside budget %d", res.GasUsed, lim.MaxGas)
	}
	return res, nil, nil
}
