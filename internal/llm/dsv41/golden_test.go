package dsv41

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"path/filepath"
	"slices"
	"testing"

	"github.com/optimist3301/rakkan-mesh/internal/num"
)

// The fixture: a tiny model with every mechanism of the real one, and the
// reference's outputs for it (tools/llmref/dsv41/tiny.py).
const fixture = "../testdata/dsv41-tiny"

const promptLen, decodeSteps = 23, 17

type golden map[string]*Tensor

func loadFixture(t testing.TB) (*Model, golden) {
	t.Helper()
	cfg, err := LoadConfig(filepath.Join(fixture, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	tables, err := BuildEngramTables(cfg, tokenizer(t))
	if err != nil {
		t.Fatal(err)
	}
	m, err := Load(fixture, tables)
	if err != nil {
		t.Fatal(err)
	}
	g, err := ReadSafetensors(filepath.Join(fixture, "golden.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	return m, golden(g)
}

func (g golden) f32(t testing.TB, name string) []float32 {
	t.Helper()
	x, ok := g[name]
	if !ok {
		t.Fatalf("golden %s missing", name)
	}
	v, err := toF32(name, x)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func (g golden) ints(t testing.TB, name string) []int {
	t.Helper()
	x := g[name]
	out := make([]int, x.Elems())
	for i := range out {
		switch x.DType {
		case "I64":
			out[i] = int(int64(binary.LittleEndian.Uint64(x.Data[8*i:])))
		case "I32":
			out[i] = int(int32(binary.LittleEndian.Uint32(x.Data[4*i:])))
		case "BOOL":
			out[i] = int(x.Data[i])
		default:
			t.Fatalf("%s: %s", name, x.DType)
		}
	}
	return out
}

// diff summarizes how a computed vector differs from the reference's.
type diff struct {
	n, same  int
	maxRel   float64 // largest |difference| / rms of the reference
	maxUlpBF int     // largest difference in bf16 units in the last place
	maxL2    float64 // largest ||difference|| / ||reference|| of one added vector
}

func (d *diff) add(got, want []float32) {
	var ss float64
	for _, w := range want {
		ss += float64(w) * float64(w)
	}
	rms := math.Sqrt(ss/float64(len(want))) + 1e-30
	var dd float64
	for i := range want {
		e := float64(got[i]) - float64(want[i])
		dd += e * e
	}
	d.maxL2 = max(d.maxL2, math.Sqrt(dd/(ss+1e-300)))
	for i := range want {
		d.n++
		if math.Float32bits(got[i]) == math.Float32bits(want[i]) {
			d.same++
			continue
		}
		d.maxRel = max(d.maxRel, math.Abs(float64(got[i]-want[i]))/rms)
		gb, wb := int(int16(num.ToBF16(got[i]))), int(int16(num.ToBF16(want[i])))
		d.maxUlpBF = max(d.maxUlpBF, abs(gb-wb))
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func (d diff) String() string {
	return fmt.Sprintf("%6.2f%% identical, max %.2g rms, %d bf16 ulp, L2 error up to %.2g", 100*float64(d.same)/float64(max(d.n, 1)), d.maxRel, d.maxUlpBF, d.maxL2)
}

// stepName is the golden's name of the step that computed position pos.
func stepName(pos int) string {
	if pos < promptLen {
		return "prefill"
	}
	return fmt.Sprintf("decode%d", pos-promptLen)
}

// row is position pos's row of a golden tensor [1, positions, ...].
func row(v []float32, pos, width int) []float32 {
	if pos >= promptLen {
		pos = 0
	}
	return v[pos*width : (pos+1)*width]
}

// run runs the fixture's tokens: the prompt, then each further token,
// teacher-forcing the stream to the reference's after every block if
// force. It returns per-layer stream differences, the index agreement and
// the logits after the prompt and after each decode step.
func run(t *testing.T, m *Model, g golden, force bool) (layers []diff, idxSame, idxN int, logits [][]float32) {
	c := m.Cfg
	tokens := g.ints(t, "tokens")
	layers = make([]diff, c.Layers)
	st := m.NewState()
	width := c.HCMult * c.Dim
	st.AfterBlock = func(layer, pos int, h, preMix []float32) {
		name := fmt.Sprintf("%s.layer%d", stepName(pos), layer)
		want := row(g.f32(t, name+".h"), pos, width)
		layers[layer].add(h, want)
		if force {
			copy(h, want)
			copy(preMix, row(g.f32(t, name+".pre_mix"), pos, c.HCMult))
		}
	}
	st.Indices = func(layer, pos int, idx []int) {
		name := fmt.Sprintf("%s.layer%d.topk_idxs", stepName(pos), layer)
		all := g.ints(t, name)
		want, offset := slices.Clone(all), c.Window
		if pos < promptLen {
			// The prefill's chunk of window rows precedes the compressed
			// ones: offset by the prompt length.
			k := len(all) / promptLen
			want, offset = slices.Clone(all[pos*k:(pos+1)*k]), promptLen
		}
		for i := range want {
			if want[i] >= 0 {
				want[i] -= offset
			}
		}
		idxN++
		if slices.Equal(idx, want) {
			idxSame++
		} else if idxN-idxSame <= 3 {
			t.Logf("layer %d pos %d: indices %v, reference %v", layer, pos, idx, want)
		}
	}
	logits = append(logits, st.Prefill(tokens[:promptLen]))
	for i := range decodeSteps {
		logits = append(logits, st.Decode(tokens[promptLen+i]))
	}
	return layers, idxSame, idxN, logits
}

func compareLogits(t *testing.T, g golden, logits [][]float32) (argmaxSame int, worst diff) {
	for i, l := range logits {
		name := "prefill.logits"
		if i > 0 {
			name = fmt.Sprintf("decode%d.logits", i-1)
		}
		want := g.f32(t, name)
		worst.add(l, want)
		if argmax(l) == argmax(want) {
			argmaxSame++
		}
	}
	return argmaxSame, worst
}

func argmax(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}

// Teacher-forced, every block starts from the reference's stream, so
// differences cannot compound: what is left is rounding where the
// reference's order of operations is unspecified (norms, softmax, float32
// products, exp). Each block must agree with the reference almost bit for
// bit, the indexers must select the same positions, and the logits must
// pick the same tokens.
func TestFixtureTeacherForced(t *testing.T) {
	m, g := loadFixture(t)
	layers, idxSame, idxN, logits := run(t, m, g, true)
	for i, d := range layers {
		t.Logf("layer %d: %s", i, d)
		if float64(d.same) < 0.999*float64(d.n) || d.maxRel > 0.01 {
			t.Errorf("layer %d differs from the reference: %s", i, d)
		}
	}
	if idxSame != idxN {
		t.Errorf("indices: %d of %d selections agree", idxSame, idxN)
	}
	same, worst := compareLogits(t, g, logits)
	t.Logf("logits: %s", worst)
	if same != len(logits) || worst.maxRel > 1e-5 {
		t.Errorf("logits: argmax agrees on %d of %d steps, %s", same, len(logits), worst)
	}
}

// Free-running, small differences compound. Activations are quantized to
// FP8 (3 mantissa bits) before every product, so a difference of one bf16
// unit can move a value across an FP8 rounding boundary or a block's
// maximum across a power of two, rescaling 32 values; selections of
// near-equal index scores then differ too. Any two implementations of the
// reference part ways like this (its GPU and CPU runs as well), so only
// the first token is required to agree.
func TestFixtureFreeRunning(t *testing.T) {
	m, g := loadFixture(t)
	_, idxSame, idxN, logits := run(t, m, g, false)
	same, _ := compareLogits(t, g, logits)
	t.Logf("indices: %d of %d selections agree; argmax agrees on %d of %d steps", idxSame, idxN, same, len(logits))
	if argmax(logits[0]) != argmax(g.f32(t, "prefill.logits")) {
		t.Error("the first token differs from the reference's")
	}
}

// Engram follows the reference exactly: hashing, FP8 rows, the FP8
// projection and the gate.
func TestEngramMatchesReference(t *testing.T) {
	m, g := loadFixture(t)
	c := m.Cfg
	tokens := g.ints(t, "tokens")
	width := c.HCMult * c.Dim
	var act num.ActFP8
	for _, layer := range c.EngramLayers {
		b := m.layers[layer]
		var hist []int64
		var d diff
		for pos := range promptLen {
			hist = append(hist, m.head.engram.TokenMap[tokens[pos]])
			if len(hist) > m.head.engram.MaxNgram {
				hist = hist[1:]
			}
			h := slices.Clone(row(g.f32(t, fmt.Sprintf("prefill.layer%d.h", layer-1)), pos, width))
			if err := b.engram.apply(h, m.head.engram.hash(hist)[b.engram.index], c, &act); err != nil {
				t.Fatal(err)
			}
			d.add(h, row(g.f32(t, fmt.Sprintf("prefill.layer%d.engram", layer)), pos, width))
		}
		if d.same != d.n {
			t.Errorf("layer %d: %s", layer, d)
		}
	}
}

// The engine's own results are the same on every processor: the digest of
// a free run's logits is fixed (run the tests on arm64 too).
func TestFixtureDigest(t *testing.T) {
	m, g := loadFixture(t)
	_, _, _, logits := run(t, m, g, false)
	h := sha256.New()
	for _, l := range logits {
		for _, v := range l {
			binary.Write(h, binary.LittleEndian, math.Float32bits(v))
		}
	}
	const want = "9e3ff53c4fac637fb656a79b9cf7225e1a80096473a156c22dc4193eddae323e"
	if got := hex.EncodeToString(h.Sum(nil)); got != want {
		t.Errorf("digest %s, want %s", got, want)
	}
}
