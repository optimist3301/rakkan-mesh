package pil

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"math/rand/v2"
	"os"
	"slices"
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
	JPEG        []struct {
		Name, Mode, File, RGB, SHA256 string
		W, H                          int
	}
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

func TestJPEGMatchesPillow(t *testing.T) {
	v := load(t)
	bad := 0
	for _, c := range v.JPEG {
		m, err := Decode(unb64(t, c.File))
		if err != nil {
			t.Errorf("%s: %v", c.Name, err)
			continue
		}
		if c.SHA256 != "" {
			if sum := sha256.Sum256(m.Pix); m.W != c.W || m.H != c.H || hex.EncodeToString(sum[:]) != c.SHA256 {
				if bad++; bad <= 25 {
					t.Errorf("%s (Pillow mode %s): %d×%d, the pixels differ", c.Name, c.Mode, m.W, m.H)
				}
			}
			continue
		}
		if c.SHA256 != "" {
			if sum := sha256.Sum256(m.Pix); m.W != c.W || m.H != c.H || hex.EncodeToString(sum[:]) != c.SHA256 {
				if bad++; bad <= 25 {
					t.Errorf("%s (Pillow mode %s): %d×%d, the pixels differ", c.Name, c.Mode, m.W, m.H)
				}
			}
			continue
		}
		want := unb64(t, c.RGB)
		if m.W != c.W || m.H != c.H || !bytes.Equal(m.Pix, want) {
			x, y, ch := firstDiff(c.W, m.Pix, want)
			n := 0
			for i := range min(len(want), len(m.Pix)) {
				if m.Pix[i] != want[i] {
					n++
				}
			}
			if bad++; bad <= 25 {
				t.Errorf("%s (Pillow mode %s): %d×%d, %d of %d values differ, the first at (%d, %d) channel %d", c.Name, c.Mode, m.W, m.H, n, len(want), x, y, ch)
			}
		}
	}
	t.Logf("%d JPEG files, %d differ", len(v.JPEG), bad)
}

// Damaged files fail or decode; they do not crash the node.
func TestJPEGDamaged(t *testing.T) {
	v := load(t)
	r := rand.New(rand.NewPCG(5, 6))
	for i, c := range v.JPEG {
		if i%7 != 0 {
			continue
		}
		data := unb64(t, c.File)
		for range 40 {
			d := slices.Clone(data)
			switch r.IntN(3) {
			case 0:
				d = d[:r.IntN(len(d))]
			case 1:
				for range 1 + r.IntN(4) {
					d[r.IntN(len(d))] = byte(r.Uint32())
				}
			default:
				at := r.IntN(len(d))
				d = append(d[:at], d[min(at+1+r.IntN(16), len(d)):]...)
			}
			Decode(d)
		}
	}
}
