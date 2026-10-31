package meter

import (
	"errors"
	"fmt"
)

var errEOF = errors.New("unexpected end of input")

// reader decodes the primitive encodings of the WebAssembly binary format.
type reader struct {
	buf []byte
	pos int
}

func (r *reader) done() bool { return r.pos >= len(r.buf) }

func (r *reader) byte() (byte, error) {
	if r.pos >= len(r.buf) {
		return 0, errEOF
	}
	b := r.buf[r.pos]
	r.pos++
	return b, nil
}

func (r *reader) bytes(n uint64) ([]byte, error) {
	if n > uint64(len(r.buf)-r.pos) {
		return nil, errEOF
	}
	b := r.buf[r.pos : r.pos+int(n)]
	r.pos += int(n)
	return b, nil
}

func (r *reader) skip(n uint64) error {
	_, err := r.bytes(n)
	return err
}

// uleb reads an unsigned LEB128 value of at most bits bits.
func (r *reader) uleb(bits uint) (uint64, error) {
	var v uint64
	for shift := uint(0); ; shift += 7 {
		if shift >= bits+7 {
			return 0, fmt.Errorf("uleb%d too long", bits)
		}
		b, err := r.byte()
		if err != nil {
			return 0, err
		}
		v |= uint64(b&0x7f) << shift
		if b&0x80 == 0 {
			if bits < 64 && v>>bits != 0 {
				return 0, fmt.Errorf("uleb%d overflow", bits)
			}
			return v, nil
		}
	}
}

func (r *reader) u32() (uint32, error) {
	v, err := r.uleb(32)
	return uint32(v), err
}

// skipSleb skips a signed LEB128 value of at most bits bits.
func (r *reader) skipSleb(bits uint) error {
	for n := uint(0); n < (bits+6)/7; n++ {
		b, err := r.byte()
		if err != nil {
			return err
		}
		if b&0x80 == 0 {
			return nil
		}
	}
	return fmt.Errorf("sleb%d too long", bits)
}

func (r *reader) name() (string, error) {
	n, err := r.u32()
	if err != nil {
		return "", err
	}
	b, err := r.bytes(uint64(n))
	return string(b), err
}

func appendUleb(out []byte, v uint64) []byte {
	for {
		b := byte(v & 0x7f)
		v >>= 7
		if v != 0 {
			b |= 0x80
		}
		out = append(out, b)
		if v == 0 {
			return out
		}
	}
}

func appendSleb(out []byte, v int64) []byte {
	for {
		b := byte(v & 0x7f)
		v >>= 7
		done := (v == 0 && b&0x40 == 0) || (v == -1 && b&0x40 != 0)
		if !done {
			b |= 0x80
		}
		out = append(out, b)
		if done {
			return out
		}
	}
}

func appendName(out []byte, s string) []byte {
	return append(appendUleb(out, uint64(len(s))), s...)
}
