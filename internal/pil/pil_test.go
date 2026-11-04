package pil

import (
	"bytes"
	"compress/gzip"
	"encoding/base64"
	"encoding/json"
	"os"
	"testing"
)

// sized is a resize or pad case: a w×h image in, a W×H image out.
type sized struct {
	W   int    `json:"w"`
	H   int    `json:"h"`
	OW  int    `json:"W"`
	OH  int    `json:"H"`
	In  string `json:"in"`
	Out string `json:"out"`
}

// load reads Pillow's vectors (tools/llmref/dsv41/image_vectors.py).
func load(t *testing.T) (v struct {
	Pillow string
	Decode []struct {
		Name, Mode, PNG, RGB string
		W, H                 int
	}
	Resize, Pad []sized
}) {
	t.Helper()
	f, err := os.Open("testdata/pillow.json.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	z, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.NewDecoder(z).Decode(&v); err != nil {
		t.Fatal(err)
	}
	return v
}

func unb64(t *testing.T, s string) []byte {
	t.Helper()
	b, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// firstDiff describes where two pixel arrays differ.
func firstDiff(w int, got, want []byte) (int, int, int) {
	for i := range min(len(got), len(want)) {
		if got[i] != want[i] {
			p := i / 3
			return p % w, p / w, i % 3
		}
	}
	return -1, -1, -1
}

func TestDecodeMatchesPillow(t *testing.T) {
	v := load(t)
	for _, c := range v.Decode {
		m, err := Decode(unb64(t, c.PNG))
		if err != nil {
			t.Errorf("%s: %v", c.Name, err)
			continue
		}
		want := unb64(t, c.RGB)
		if m.W != c.W || m.H != c.H || !bytes.Equal(m.Pix, want) {
			x, y, ch := firstDiff(c.W, m.Pix, want)
			t.Errorf("%s (Pillow mode %s): %d×%d, first difference at (%d, %d) channel %d", c.Name, c.Mode, m.W, m.H, x, y, ch)
		}
	}
	t.Logf("%d images, Pillow %s", len(v.Decode), v.Pillow)
}

func TestResizeMatchesPillow(t *testing.T) {
	v := load(t)
	for _, c := range v.Resize {
		in := &RGB{W: c.W, H: c.H, Pix: unb64(t, c.In)}
		got := in.Resize(c.OW, c.OH)
		want := unb64(t, c.Out)
		if !bytes.Equal(got.Pix, want) {
			x, y, ch := firstDiff(c.OW, got.Pix, want)
			t.Errorf("%d×%d to %d×%d: first difference at (%d, %d) channel %d", c.W, c.H, c.OW, c.OH, x, y, ch)
		}
	}
	t.Logf("%d resizes", len(v.Resize))
}

func TestPadMatchesPillow(t *testing.T) {
	v := load(t)
	for _, c := range v.Pad {
		in := &RGB{W: c.W, H: c.H, Pix: unb64(t, c.In)}
		got := in.Pad(c.OW, c.OH, [3]byte{127, 127, 127})
		want := unb64(t, c.Out)
		if got.W != c.OW || got.H != c.OH || !bytes.Equal(got.Pix, want) {
			x, y, ch := firstDiff(c.OW, got.Pix, want)
			t.Errorf("%d×%d padded to %d×%d: first difference at (%d, %d) channel %d", c.W, c.H, c.OW, c.OH, x, y, ch)
		}
	}
	t.Logf("%d pads", len(v.Pad))
}

func TestDecodeRefuses(t *testing.T) {
	for _, b := range [][]byte{nil, []byte("GIF89a"), []byte("\x89PNG\r\n\x1a\n")} {
		if _, err := Decode(b); err == nil {
			t.Errorf("%q decoded", b)
		}
	}
}
