package dsv41

import (
	"bytes"
	"fmt"
	"math"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// fixtureParts loads the fixture's tensors for building parts.
func fixtureParts(t *testing.T) (*Config, map[string]*Tensor, *EngramTables) {
	t.Helper()
	cfg, err := LoadConfig(filepath.Join(fixture, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	tables, err := BuildEngramTables(cfg, tokenizer(t))
	if err != nil {
		t.Fatal(err)
	}
	tensors, err := ReadSafetensors(filepath.Join(fixture, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	return cfg, tensors, tables
}

// pipeline runs tokens through stages split at cuts, passing each
// position's Payload between them as bytes, and returns the logits of
// every position. replace, if set, swaps a stage for a new one replaying
// its inputs before the given position.
func pipeline(t *testing.T, cfg *Config, head *Head, parts []*Model, tokens []int, prompt int, replace map[int]int) [][]float32 {
	t.Helper()
	stages := make([]*Stage, len(parts))
	inputs := make([][][]byte, len(parts))
	for i, p := range parts {
		stages[i] = p.NewStage()
	}
	var logits [][]float32
	for pos := range tokens {
		prefill := 0
		if pos < prompt {
			prefill = prompt
		}
		p, err := head.Embed(tokens[:pos+1], prefill)
		if err != nil {
			t.Fatal(err)
		}
		for i, part := range parts {
			if at, ok := replace[i]; ok && at == pos {
				stages[i] = part.NewStage()
				for q, in := range inputs[i] {
					p, err := DecodePayloads(in, cfg, 1, len(tokens))
					if err != nil {
						t.Fatal(err)
					}
					if err := stages[i].Forward(p[0], q); err != nil {
						t.Fatal(err)
					}
				}
			}
			b, err := AppendPayload(nil, cfg, p)
			if err != nil {
				t.Fatal(err)
			}
			if len(b) > MaxPayloadBytes(cfg) {
				t.Fatalf("payload of %d bytes, bound %d", len(b), MaxPayloadBytes(cfg))
			}
			inputs[i] = append(inputs[i], b)
			ps, err := DecodePayloads(b, cfg, 1, len(tokens))
			if err != nil {
				t.Fatal(err)
			}
			p = ps[0]
			if err := stages[i].Forward(p, pos); err != nil {
				t.Fatalf("stage %d (layers %d..%d), position %d: %v", i, parts[i].from, parts[i].to, pos, err)
			}
		}
		logits = append(logits, head.Logits(p))
	}
	return logits
}

// Any split of the layers into stages computes what one process does, bit
// for bit: with cuts inside groups of layers sharing a KV source, between
// an index source and the layers using its selection, before and after
// the candidate source and the Engram layers.
func TestStagesMatchOneProcess(t *testing.T) {
	cfg, tensors, tables := fixtureParts(t)
	m, err := New(cfg, tensors, tables)
	if err != nil {
		t.Fatal(err)
	}
	_, g := loadFixture(t)
	tokens := g.ints(t, "tokens")
	prompt := promptLen

	st := m.NewState()
	var want [][]float32
	for pos, tok := range tokens {
		prefill := 0
		if pos < prompt {
			prefill = prompt
		}
		want = append(want, st.step(tok, prefill, true))
	}

	head, err := NewHead(cfg, map[string]*Tensor{"embed.weight": tensors["embed.weight"], "norm.weight": tensors["norm.weight"], "head.weight": tensors["head.weight"]}, tables)
	if err != nil {
		t.Fatal(err)
	}
	L := cfg.Layers
	// Every way to cut the layers into consecutive stages.
	for mask := range 1 << (L - 1) {
		cuts := []int{0}
		for i := 1; i < L; i++ {
			if mask&(1<<(i-1)) != 0 {
				cuts = append(cuts, i)
			}
		}
		cuts = append(cuts, L)
		var parts []*Model
		for i := range len(cuts) - 1 {
			part, err := NewPart(cfg, layerTensors(tensors, cuts[i], cuts[i+1]), cuts[i], cuts[i+1], nil)
			if err != nil {
				t.Fatal(err)
			}
			parts = append(parts, part)
		}
		// A stage replaced mid-prompt and one mid-generation replay their inputs.
		replace := map[int]int{0: 7, len(parts) - 1: prompt + 3}
		got := pipeline(t, cfg, head, parts, tokens, prompt, replace)
		for pos := range want {
			if !sameBits(got[pos], want[pos]) {
				t.Fatalf("stages %v: logits differ at position %d", cuts, pos)
			}
		}
	}
}

// Engram tables read through a ReaderAt, as from disk, give the same.
func TestEngramTableApart(t *testing.T) {
	cfg, tensors, tables := fixtureParts(t)
	m, err := New(cfg, tensors, tables)
	if err != nil {
		t.Fatal(err)
	}
	apart := layerTensors(tensors, 0, cfg.Layers)
	engram := map[int]EngramTable{}
	for _, l := range cfg.EngramLayers {
		p := fmt.Sprintf("layers.%d.engram.embed.", l)
		engram[l] = EngramTable{Rows: bytes.NewReader(apart[p+"weight"].Data), Scales: bytes.NewReader(apart[p+"scale"].Data)}
		delete(apart, p+"weight")
		delete(apart, p+"scale")
	}
	part, err := NewPart(cfg, apart, 0, cfg.Layers, engram)
	if err != nil {
		t.Fatal(err)
	}
	head := m.Head()
	tokens := []int{5, 17, 99, 5, 17, 300, 2}
	got := pipeline(t, cfg, head, []*Model{part}, tokens, 4, nil)
	whole := m.NewState()
	for pos, tok := range tokens {
		prefill := 0
		if pos < 4 {
			prefill = 4
		}
		if want := whole.step(tok, prefill, true); !sameBits(got[pos], want) {
			t.Fatalf("position %d differs", pos)
		}
	}
}

// A stage refuses payloads that do not fit it rather than misbehave.
func TestStageRejectsBadPayloads(t *testing.T) {
	cfg, tensors, tables := fixtureParts(t)
	head, err := NewHead(cfg, layerTensors(tensors, -1, -1), tables)
	if err != nil {
		t.Fatal(err)
	}
	first, err := NewPart(cfg, layerTensors(tensors, 0, 3), 0, 3, nil)
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewPart(cfg, layerTensors(tensors, 3, cfg.Layers), 3, cfg.Layers, nil)
	if err != nil {
		t.Fatal(err)
	}
	a, b := first.NewStage(), second.NewStage()
	tokens := []int{1, 2, 3, 4}
	for pos := range tokens {
		p, _ := head.Embed(tokens[:pos+1], 0)
		if err := a.Forward(p, pos); err != nil {
			t.Fatal(err)
		}
		if pos == len(tokens)-1 {
			bad := *p
			bad.TopK = append(slices.Clone(p.TopK), 50)
			if err := b.Forward(&bad, pos); err == nil {
				t.Error("selection of a missing entry accepted")
			}
			bad = *p
			bad.Source = -1
			if err := b.Forward(&bad, pos); err == nil {
				t.Error("payload from the wrong KV source accepted")
			}
			if err := b.Forward(p, pos+1); err == nil {
				t.Error("position out of order accepted")
			}
		}
		if err := b.Forward(p, pos); err != nil {
			t.Fatal(err)
		}
	}
	enc, err := AppendPayload(nil, cfg, &Payload{H: make([]float32, cfg.HCMult*cfg.Dim), PreMix: make([]float32, cfg.HCMult),
		Hashes: [][]int64{make([]int64, 6), make([]int64, 6)}, Source: -1})
	if err != nil {
		t.Fatal(err)
	}
	for cut := range len(enc) {
		if _, err := DecodePayloads(enc[:cut], cfg, 1, 64); err == nil {
			t.Fatalf("payload cut at %d decoded", cut)
		}
	}
	if _, err := DecodePayloads(append(enc, 0), cfg, 1, 64); err == nil {
		t.Fatal("payload with trailing bytes decoded")
	}
}

// layerTensors picks the tensors of layers [from, to), or the head's when
// from < 0.
func layerTensors(all map[string]*Tensor, from, to int) map[string]*Tensor {
	out := map[string]*Tensor{}
	for name, t := range all {
		if l, ok := TensorLayer(name); ok && l >= from && l < to || !ok && from < 0 && IsHeadTensor(name) {
			out[name] = t
		}
	}
	return out
}

func sameBits(a, b []float32) bool {
	return slices.EqualFunc(a, b, func(x, y float32) bool { return math.Float32bits(x) == math.Float32bits(y) })
}

// The sizes estimated from the config bound the fixture's real ones.
func TestLayerSizes(t *testing.T) {
	cfg, tensors, _ := fixtureParts(t)
	for i := range cfg.Layers {
		var stored, params int64
		for name, x := range layerTensors(tensors, i, i+1) {
			if strings.Contains(name, ".engram.embed.") {
				continue
			}
			stored += int64(len(x.Data))
			n := int64(x.Elems())
			if x.DType == "I8" {
				n *= 2 // packed FP4
			}
			if strings.HasSuffix(name, ".scale") && x.DType == "F8_E8M0" {
				continue
			}
			if strings.Contains(name, ".ffn.experts.") {
				n = n * int64(cfg.ActiveExperts) / int64(cfg.RoutedExperts)
			}
			params += n
		}
		held := LayerBytes(cfg, i)
		if stored > held || held > 3*stored {
			t.Errorf("layer %d: %d bytes stored, estimated %d held", i, stored, held)
		}
		if p := ActiveParams(cfg, i); math.Abs(float64(p-params)) > 0.05*float64(params) {
			t.Errorf("layer %d: %d active parameters, estimated %d", i, params, p)
		}
		if a := ActiveBytes(cfg, i); a >= held || a <= 0 {
			t.Errorf("layer %d: %d active of %d bytes", i, a, held)
		}
	}
	for i, l := range cfg.EngramLayers {
		p := fmt.Sprintf("layers.%d.engram.embed.", l)
		if n := int64(len(tensors[p+"weight"].Data) + len(tensors[p+"scale"].Data)); n != EngramTableBytes(cfg, l) {
			t.Errorf("Engram layer %d (%d): %d bytes, estimated %d", l, i, n, EngramTableBytes(cfg, l))
		}
	}
}
