// Package framing reads length-prefixed data announced by untrusted peers.
package framing

import (
	"errors"
	"io"
)

// step is the most memory reserved ahead of data actually received.
const step = 1 << 20

// ReadFull reads exactly n bytes from r. Unlike allocating n bytes and
// calling io.ReadFull, it grows its buffer as data arrives, so a peer that
// announces a large frame and sends little makes the reader hold little.
func ReadFull(r io.Reader, n int) ([]byte, error) {
	if n < 0 {
		return nil, errors.New("negative frame length")
	}
	b := make([]byte, 0, min(n, step))
	for len(b) < n {
		if len(b) == cap(b) {
			// Double, up to n: amortized copies, and at most twice what
			// has arrived.
			grown := make([]byte, len(b), min(n, 2*cap(b)))
			copy(grown, b)
			b = grown
		}
		k, err := r.Read(b[len(b):cap(b)])
		b = b[:len(b)+k]
		if len(b) == n {
			return b, nil
		}
		if err == io.EOF {
			return nil, io.ErrUnexpectedEOF
		}
		if err != nil {
			return nil, err
		}
	}
	return b, nil
}
