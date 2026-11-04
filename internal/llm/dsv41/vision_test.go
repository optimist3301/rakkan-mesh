package dsv41

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"path/filepath"
	"testing"
)

func loadVision(t testing.TB) (*Vision, golden) {
	t.Helper()
	cfg, err := LoadConfig(filepath.Join(fixture, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	tensors, err := ReadSafetensors(filepath.Join(fixture, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	vis, err := NewVision(cfg, tensors)
	if err != nil {
		t.Fatal(err)
	}
	g, err := ReadSafetensors(filepath.Join(fixture, "golden.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	return vis, golden(g)
}

// The fixture's image: prepared bit for bit as the reference's, encoded to
// rounding. Per block (Vision.afterBlock) the stream is 76% and 42%
// identical, the error 0.3% and 0.5% of the norm.
func TestVisionMatchesReference(t *testing.T) {
	vis, g := loadVision(t)
	png := g["vl.png"].Data
	img, err := vis.cfg.PrepareImage(png)
	if err != nil {
		t.Fatal(err)
	}
	want := g.f32(t, "vl.patches")
	if len(img.Patches) != len(want) {
		t.Fatalf("%d patch values, want %d", len(img.Patches), len(want))
	}
	for i := range want {
		if math.Float32bits(img.Patches[i]) != math.Float32bits(want[i]) {
			t.Fatalf("patch value %d: %v, want %v", i, img.Patches[i], want[i])
		}
	}
	t.Logf("image %d×%d px: %d×%d patches, %d×%d positions", img.Width, img.Height, img.VitH, img.VitW, img.LLMH, img.LLMW)

	vit, err := vis.ViT(img)
	if err != nil {
		t.Fatal(err)
	}
	var dv diff
	dv.add(vit, g.f32(t, "vl.vit"))
	t.Logf("ViT:     %v", dv)
	emb, err := vis.Encode(img)
	if err != nil {
		t.Fatal(err)
	}
	var de diff
	de.add(emb, g.f32(t, "vl.embeds"))
	t.Logf("aligner: %v", de)
	// Every step but the attention is the reference's to the bit on its
	// inputs; the attention (PyTorch's flash kernel, its own exp) differs
	// in the last bf16 bit of a few percent of values, which the blocks
	// then spread.
	if dv.maxL2 > 0.01 || de.maxL2 > 0.01 {
		t.Errorf("the encoder disagrees with the reference beyond rounding")
	}
}

// The encoder's bits are the same on every node (checked on amd64 and,
// under qemu, arm64, where Go fuses multiply-adds unless told not to).
func TestVisionDigest(t *testing.T) {
	vis, g := loadVision(t)
	img, err := vis.cfg.PrepareImage(g["vl.png"].Data)
	if err != nil {
		t.Fatal(err)
	}
	emb, err := vis.Encode(img)
	if err != nil {
		t.Fatal(err)
	}
	h := sha256.New()
	for _, v := range emb {
		binary.Write(h, binary.LittleEndian, math.Float32bits(v))
	}
	const want = "45e7e6f7ff139fcded31b0be98ba2299ab6450c76bd9b6899e20d04f5d1e70cc"
	if got := hex.EncodeToString(h.Sum(nil)); got != want {
		t.Errorf("digest %s, want %s", got, want)
	}
}

// runVL runs the fixture's prompt with an image, its span's inputs from
// rows (nil: the reference's encoder output), teacher-forced if force as
// run is. It returns per-layer differences and the logits of each step.
func runVL(t *testing.T, m *Model, vis *Vision, g golden, emb []float32, force bool) ([]diff, [][]float32) {
	c := m.Cfg
	tokens, types := g.ints(t, "vl.tokens"), g.ints(t, "vl.types")
	const steps = 6
	prompt := len(tokens) - steps
	img, err := vis.cfg.PrepareImage(g["vl.png"].Data)
	if err != nil {
		t.Fatal(err)
	}
	if emb == nil {
		emb = g.f32(t, "vl.embeds")
	}
	span, err := vis.Span(img.ImageGrid, emb)
	if err != nil {
		t.Fatal(err)
	}
	rows := make([][]float32, prompt)
	k := 0
	for i, ty := range types[:prompt] {
		if ty != TokenText {
			rows[i] = span[k]
			k++
		}
	}
	if k != len(span) {
		t.Fatalf("%d image positions, a span of %d", k, len(span))
	}
	layers := make([]diff, c.Layers)
	width := c.HCMult * c.Dim
	st := m.NewState()
	st.AfterBlock = func(layer, pos int, h, preMix []float32) {
		name, r := "vl.prefill", pos
		if pos >= prompt {
			name, r = fmt.Sprintf("vl.decode%d", pos-prompt), 0
		}
		name = fmt.Sprintf("%s.layer%d", name, layer)
		want := g.f32(t, name+".h")[r*width : (r+1)*width]
		layers[layer].add(h, want)
		if force {
			copy(h, want)
			copy(preMix, g.f32(t, name+".pre_mix")[r*c.HCMult:(r+1)*c.HCMult])
		}
	}
	logits := [][]float32{st.PrefillImages(tokens[:prompt], rows)}
	for _, tok := range tokens[prompt:] {
		logits = append(logits, st.Decode(tok))
	}
	return layers, logits
}

// A prompt with an image through the whole model: teacher-forced on the
// reference's encoder output, every block agrees with the reference almost
// bit for bit, as for text: the image bias routes the span, Engram skips
// it and its n-grams stop at it.
func TestImagePromptMatchesReference(t *testing.T) {
	m, _ := loadFixture(t)
	vis, g := loadVision(t)
	layers, logits := runVL(t, m, vis, g, nil, true)
	for i, d := range layers {
		t.Logf("layer %d: %s", i, d)
		if float64(d.same) < 0.999*float64(d.n) || d.maxRel > 0.01 {
			t.Errorf("layer %d differs from the reference: %s", i, d)
		}
	}
	for i, l := range logits {
		name := "vl.prefill.logits"
		if i > 0 {
			name = fmt.Sprintf("vl.decode%d.logits", i-1)
		}
		var d diff
		d.add(l, g.f32(t, name))
		if argmax(l) != argmax(g.f32(t, name)) || d.maxRel > 1e-5 {
			t.Errorf("%s: %s", name, d)
		}
	}
	// With the engine's own encoder, free-running: the first token agrees.
	img, _ := vis.cfg.PrepareImage(g["vl.png"].Data)
	emb, err := vis.Encode(img)
	if err != nil {
		t.Fatal(err)
	}
	_, own := runVL(t, m, vis, g, emb, false)
	if argmax(own[0]) != argmax(g.f32(t, "vl.prefill.logits")) {
		t.Error("with the engine's encoder, the first token differs from the reference's")
	}
}
