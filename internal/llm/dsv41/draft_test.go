package dsv41

import (
	"fmt"
	"testing"
)

// The draft against the reference's forward_spec on the fixture: fed the
// reference's main streams (prefill.main_hidden, decode*.main_hidden) and
// its tokens, it drafts the same tokens, with logits and confidence to
// rounding and the first draft block's stream mostly bit for bit.
func TestDraftMatchesReference(t *testing.T) {
	cfg, tensors, tables := fixtureParts(t)
	_, g := loadFixture(t)
	head, err := NewHead(cfg, layerTensors(tensors, -1, -1), tables)
	if err != nil {
		t.Fatal(err)
	}
	dt := map[string]*Tensor{}
	for name, x := range tensors {
		if IsDraftTensor(name) {
			dt[name] = x
		}
	}
	d, err := NewDraft(cfg, dt)
	if err != nil {
		t.Fatal(err)
	}
	s := d.NewState()
	width := cfg.Dim * len(cfg.DraftTargets)
	prefill := g.f32(t, "prefill.main_hidden")
	for pos := range promptLen {
		if err := s.AddMain(prefill[pos*width : (pos+1)*width]); err != nil {
			t.Fatal(err)
		}
	}
	var stream diff
	var logitsD, confD diff
	idsSame := 0
	for step := range decodeSteps {
		name := fmt.Sprintf("decode%d", step)
		if err := s.AddMain(g.f32(t, name+".main_hidden")); err != nil {
			t.Fatal(err)
		}
		want := g.ints(t, name+".spec.ids")
		var hs []float32
		s.st.AfterBlock = func(layer, pos int, h, _ []float32) {
			if layer == cfg.Layers {
				hs = append(hs, h...)
			}
		}
		p, err := s.Propose(want[0], head)
		if err != nil {
			t.Fatal(err)
		}
		stream.add(hs, g.f32(t, name+".mtp0.h"))
		wl := g.f32(t, name+".spec.logits")
		for j, l := range p.Logits {
			logitsD.add(l, wl[j*cfg.Vocab:(j+1)*cfg.Vocab])
		}
		confD.add(p.Confidence, g.f32(t, name+".spec.confidence"))
		same := true
		for j := range want {
			if p.Tokens[j] != want[j] {
				same = false
			}
		}
		if same {
			idsSame++
		} else {
			t.Logf("step %d: drafted %v, reference %v", step, p.Tokens, want)
		}
	}
	t.Logf("draft block 0's stream: %s", stream)
	t.Logf("logits: %s; confidence: %s; drafts the same at %d of %d steps", logitsD, confD, idsSame, decodeSteps)
	if stream.same*1000 < stream.n*999 {
		t.Errorf("draft stream: %s", stream)
	}
	if idsSame < decodeSteps-1 {
		t.Errorf("drafts differ at %d of %d steps", decodeSteps-idsSame, decodeSteps)
	}
}

// Payload.Main, the target layers' input means, is the reference's
// main_hidden bit for bit when the stream is the reference's.
func TestMainMatchesReference(t *testing.T) {
	m, g := loadFixture(t)
	c := m.Cfg
	tokens := g.ints(t, "tokens")
	width := c.HCMult * c.Dim
	st := m.NewState()
	st.AfterBlock = func(layer, pos int, h, preMix []float32) {
		name := fmt.Sprintf("%s.layer%d", stepName(pos), layer)
		copy(h, row(g.f32(t, name+".h"), pos, width))
		copy(preMix, row(g.f32(t, name+".pre_mix"), pos, c.HCMult))
	}
	main := len(c.DraftTargets) * c.Dim
	for pos := range tokens {
		prefill := 0
		if pos < promptLen {
			prefill = promptLen
		}
		p, err := m.Head().Embed(tokens[:pos+1], prefill)
		if err != nil {
			t.Fatal(err)
		}
		p.WantMain = true
		if err := st.run(p, pos); err != nil {
			t.Fatal(err)
		}
		want := row(g.f32(t, stepName(pos)+".main_hidden"), pos, main)
		if !sameBits(p.Main, want) {
			var d diff
			d.add(p.Main, want)
			t.Fatalf("position %d: %s", pos, d)
		}
		// And it survives the wire.
		b, err := AppendPayload(nil, c, p)
		if err != nil {
			t.Fatal(err)
		}
		q, _, err := DecodePayload(b, c, len(tokens))
		if err != nil || !q.WantMain || !sameBits(q.Main, p.Main) {
			t.Fatalf("position %d: Main through the encoding: %v", pos, err)
		}
	}
}

// Positions rewound leave no trace: after wrong tokens run and rewound,
// the stages compute what they would have without them, whatever the
// depth (into a compressor's open group too) and the split.
func TestStagesRewind(t *testing.T) {
	cfg, tensors, tables := fixtureParts(t)
	head, err := NewHead(cfg, layerTensors(tensors, -1, -1), tables)
	if err != nil {
		t.Fatal(err)
	}
	_, g := loadFixture(t)
	tokens := g.ints(t, "tokens")
	run := func(parts []*Stage, toks []int, pos int) *Payload {
		prefill := 0
		if pos < promptLen {
			prefill = promptLen
		}
		p, err := head.Embed(toks[:pos+1], prefill)
		if err != nil {
			t.Fatal(err)
		}
		for _, st := range parts {
			if err := st.Forward(p, pos); err != nil {
				t.Fatal(err)
			}
		}
		return p
	}
	stages := func(cuts []int) []*Stage {
		var out []*Stage
		for i := range len(cuts) - 1 {
			part, err := NewPart(cfg, layerTensors(tensors, cuts[i], cuts[i+1]), cuts[i], cuts[i+1], nil)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, part.NewStage())
		}
		return out
	}
	for _, cuts := range [][]int{{0, 8}, {0, 3, 6, 8}} {
		want := stages(cuts)
		var ref [][]byte
		for pos := range tokens {
			b, _ := AppendPayload(nil, cfg, run(want, tokens, pos))
			ref = append(ref, b)
		}
		got := stages(cuts)
		pos := 0
		// Runs of wrong tokens of each depth, rewound, before the right ones.
		for depth := 1; pos < len(tokens); depth = depth%5 + 1 {
			wrong := append([]int(nil), tokens...)
			for q := pos; q < min(pos+depth, len(tokens)); q++ {
				wrong[q] = (tokens[q] + 7) % cfg.Vocab
				run(got, wrong, q)
			}
			for q := pos; q < min(pos+depth, len(tokens)); q++ {
				if b, _ := AppendPayload(nil, cfg, run(got, tokens, q)); string(b) != string(ref[q]) {
					t.Fatalf("stages %v: position %d after rewinding %d positions differs", cuts, q, depth)
				}
			}
			pos += depth
		}
	}
	// No further back than kept.
	st := stages([]int{0, 8})
	for pos := range Rewindable + 2 {
		run(st, tokens, pos)
	}
	if err := st[0].Forward(&Payload{}, 0); err == nil {
		t.Fatal("rewound further than kept")
	}
}
