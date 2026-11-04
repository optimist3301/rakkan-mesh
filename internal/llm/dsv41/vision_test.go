package dsv41

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
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
