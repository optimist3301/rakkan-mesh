package dsv41

import (
	"compress/gzip"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"os"
	"slices"
	"testing"

	"github.com/optimist3301/rakkan-mesh/internal/num"
)

// imageArgs are the reference's ModelArgs of image preparation.
type imageArgs struct {
	Patch      int  `json:"vision_patch_size"`
	Downsample int  `json:"vision_downsample_ratio"`
	MaxTokens  int  `json:"vision_max_n_token"`
	MinPixels  int  `json:"vision_min_pixels"`
	MaxWHRatio *int `json:"vision_max_wh_ratio"`
}

func (a imageArgs) config() *Config {
	c := &Config{Vision: VisionConfig{Layers: 1, Patch: a.Patch, Downsample: a.Downsample, MaxTokens: a.MaxTokens, MinPixels: a.MinPixels}}
	if a.MaxWHRatio != nil {
		c.Vision.MaxWHRatio = *a.MaxWHRatio
	}
	return c
}

// imageVectors are the reference's (tools/llmref/dsv41/image_vectors.py).
func imageVectors(t *testing.T) (v struct {
	Pillow string
	Grid   []struct {
		Args imageArgs
		W, H int
		Grid []int // n_llm_h, n_llm_w, best_height, best_width; nil: none
	}
	Load []struct {
		Args          imageArgs
		File          string // PNG or JPEG
		Grid          []int  // n_vit_h, n_vit_w, n_llm_h, n_llm_w
		SHA256, First string
	}
}) {
	t.Helper()
	f, err := os.Open("testdata/images.json.gz")
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

func TestPlanImageMatchesReference(t *testing.T) {
	v := imageVectors(t)
	for _, c := range v.Grid {
		g, err := c.Args.config().PlanImage(c.W, c.H)
		if c.Grid == nil {
			if err == nil {
				t.Errorf("%+v %d×%d: %+v, want no grid", c.Args, c.W, c.H, g)
			}
			continue
		}
		got := []int{g.LLMH, g.LLMW, g.Height, g.Width}
		if err != nil || !slices.Equal(got, c.Grid) {
			t.Errorf("%+v %d×%d: %v %v, want %v", c.Args, c.W, c.H, got, err, c.Grid)
		}
	}
	t.Logf("%d plans", len(v.Grid))
}

func TestPrepareImageMatchesReference(t *testing.T) {
	v := imageVectors(t)
	for i, c := range v.Load {
		data, _ := base64.StdEncoding.DecodeString(c.File)
		img, err := c.Args.config().PrepareImage(data)
		if err != nil {
			t.Errorf("case %d: %v", i, err)
			continue
		}
		grid := []int{img.VitH, img.VitW, img.LLMH, img.LLMW}
		if !slices.Equal(grid, c.Grid) {
			t.Errorf("case %d: grid %v, want %v", i, grid, c.Grid)
			continue
		}
		bits := make([]byte, 2*len(img.Patches))
		for j, x := range img.Patches {
			binary.LittleEndian.PutUint16(bits[2*j:], num.ToBF16(x))
		}
		sum := sha256.Sum256(bits)
		if hex.EncodeToString(sum[:]) != c.SHA256 {
			first, _ := base64.StdEncoding.DecodeString(c.First)
			at := -1
			for j := range first {
				if first[j] != bits[j] {
					at = j / 2
					break
				}
			}
			t.Errorf("case %d (%+v, grid %v): patches differ; in the first patch from value %d", i, c.Args, grid, at)
		}
	}
	t.Logf("%d images", len(v.Load))
}

func TestImageSpanTypes(t *testing.T) {
	g := ImageGrid{LLMH: 2, LLMW: 3}
	want := []int{TokenImageStart, TokenImage, TokenImage, TokenImage, TokenImageNewLine,
		TokenImage, TokenImage, TokenImage, TokenImageNewLine, TokenImageEnd}
	if got := g.Types(); !slices.Equal(got, want) || g.Tokens() != len(want) {
		t.Errorf("%v (%d tokens)", got, g.Tokens())
	}
}

func TestRealConfigVision(t *testing.T) {
	cfg, err := LoadConfig("../../../tools/llmref/dsv41/upstream/config.json")
	if err != nil {
		t.Fatal(err)
	}
	want := VisionConfig{Layers: 32, Dim: 1024, Heads: 16, Inter: 2816, Patch: 14, RopeTheta: 10000,
		Downsample: 3, MaxTokens: 1024, MinPixels: 295936}
	if cfg.Vision != want || cfg.ImageToken != 129264 {
		t.Errorf("%+v, image token %d", cfg.Vision, cfg.ImageToken)
	}
}
