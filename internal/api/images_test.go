package api

import (
	"bytes"
	"testing"
)

func TestDecodeImage(t *testing.T) {
	for _, c := range []struct {
		in   string
		want []byte
	}{
		{"iVBORw==", []byte("\x89PNG")},
		{"data:image/png;base64,iVBORw==", []byte("\x89PNG")},
	} {
		got, err := decodeImage(c.in)
		if err != nil || !bytes.Equal(got, c.want) {
			t.Errorf("%q: %q %v", c.in, got, err)
		}
	}
	for _, in := range []string{"https://example.com/a.png", "file:///etc/passwd", "data:image/png,raw", "not base64!"} {
		if _, err := decodeImage(in); err == nil {
			t.Errorf("%q decoded", in)
		}
	}
}
