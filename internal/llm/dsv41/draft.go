package dsv41

import (
	"errors"
	"fmt"
	"strings"

	"github.com/optimist3301/rakkan-mesh/internal/num"
)

// DSpark, the draft model (docs/dev/deepseek-v41.md, section 10; the
// reference's DSparkBlock, model.py:1103). It drafts DraftBlock tokens at
// once from the main model's stream: the mean of its copies at the input
// of the target layers (Payload.Main), projected and normalized (main_x),
// gives each draft block's window keys and values, one per main position;
// the draft's own positions, the last accepted token then noise tokens,
// attend to that window and to one another, all to all. The main model's
// classifier, a Markov head adding logits for each drafted token's
// successor, and a confidence head finish it.

// Draft is the draft model's weights.
type Draft struct {
	m           *Model // its blocks, as layers Layers.. of a model
	mainProj    *linear
	mainNorm    []float32
	norm        []float32
	markovEmbed []uint16 // [vocab, rank] bf16
	markovHead  *linear  // [vocab, rank]
	confidence  *linear  // [1, dim+rank]
}

// IsDraftTensor reports whether a checkpoint tensor is the draft model's.
func IsDraftTensor(name string) bool { return strings.HasPrefix(name, "mtp.") }

// NewDraft builds the draft model from its tensors (mtp.*).
func NewDraft(cfg *Config, tensors map[string]*Tensor) (*Draft, error) {
	c := cfg
	if c.MTPLayers == 0 || c.DraftBlock == 0 || len(c.DraftTargets) == 0 {
		return nil, errors.New("dsv41: the model has no DSpark draft")
	}
	w := &weights{t: tensors, used: map[string]bool{}}
	m := &Model{Cfg: c, from: c.Layers, to: c.Layers + c.MTPLayers, ropes: [2]*rope{newRope(c, false), newRope(c, true)}}
	m.layers = make([]*block, c.Layers+c.MTPLayers)
	for j := range c.MTPLayers {
		b, err := m.loadBlock(w, c.Layers+j)
		if err != nil {
			return nil, err
		}
		if b.attn.ratio != 0 {
			return nil, fmt.Errorf("dsv41: draft block %d compresses its KV", j)
		}
		m.layers[c.Layers+j] = b
	}
	d := &Draft{m: m}
	first, last := "mtp.0.", fmt.Sprintf("mtp.%d.", c.MTPLayers-1)
	var err error
	if d.mainProj, err = w.lin(first+"main_proj", c.Dim, c.Dim*len(c.DraftTargets), kindFP8); err != nil {
		return nil, err
	}
	if d.mainNorm, err = w.vec(first+"main_norm.weight", c.Dim); err != nil {
		return nil, err
	}
	if d.norm, err = w.vec(last+"norm.weight", c.Dim); err != nil {
		return nil, err
	}
	t, err := w.get(last + "markov_head.embed.weight")
	if err != nil {
		return nil, err
	}
	if t.DType != "BF16" || !shapeIs(t, c.Vocab, c.DraftRank) {
		return nil, fmt.Errorf("dsv41: markov_head.embed %s %v", t.DType, t.Shape)
	}
	d.markovEmbed = toBF16(t)
	if d.markovHead, err = w.lin(last+"markov_head.head", c.Vocab, c.DraftRank, kindBF16F32); err != nil {
		return nil, err
	}
	if d.confidence, err = w.lin(last+"confidence_head.proj", 1, c.Dim+c.DraftRank, kindBF16F32); err != nil {
		return nil, err
	}
	var unused []string
	for name := range tensors {
		if IsDraftTensor(name) && !w.used[name] && !strings.HasSuffix(name, ".bias_vl") {
			unused = append(unused, name)
		}
	}
	if len(unused) > 0 {
		return nil, fmt.Errorf("dsv41: unexpected draft tensors: %v", unused)
	}
	return d, nil
}

// DraftState is a generation's draft: each draft block's window of keys
// and values from the main model's positions.
type DraftState struct {
	d      *Draft
	st     *State // scratch, and the block arithmetic
	window [][][]float32
	main   int // main positions seen
}

// NewState starts a generation's draft.
func (d *Draft) NewState() *DraftState {
	c := d.m.Cfg
	s := &DraftState{d: d, st: d.m.NewState(), window: make([][][]float32, c.MTPLayers)}
	for j := range s.window {
		s.window[j] = make([][]float32, c.Window)
	}
	return s
}

// Main returns how many main positions the draft has seen.
func (s *DraftState) Main() int { return s.main }

// AddMain gives the draft the main model's next position: its Payload.Main
// (the stream's means at the target layers). The reference's prefill
// (forward_spec at position 0) and decode steps do this for each position.
func (s *DraftState) AddMain(main []float32) error {
	d, c := s.d, s.d.m.Cfg
	if len(main) != c.Dim*len(c.DraftTargets) {
		return fmt.Errorf("dsv41: %d values of the main stream, want %d", len(main), c.Dim*len(c.DraftTargets))
	}
	pos := s.main
	x := make([]float32, c.Dim)
	d.mainProj.apply(x, main, &s.st.act)
	rmsNorm(x, x, d.mainNorm, c.NormEps)
	for j := range c.MTPLayers {
		a := d.m.layers[c.Layers+j].attn
		kv := make([]float32, c.HeadDim)
		a.wkv.apply(kv, x, &s.st.act)
		rmsNorm(kv, kv, a.kvNorm, c.NormEps)
		a.rope.apply(kv[c.HeadDim-c.RopeDim:], pos, false)
		num.ActQuantInPlace(kv)
		s.window[j][pos%c.Window] = kv
	}
	s.main++
	return nil
}

// Proposal is a draft: Tokens[0] is the token it was given, Tokens[1:]
// the DraftBlock tokens drafted after it (greedily), with the logits each
// was chosen from and the confidence head's score for each position.
type Proposal struct {
	Tokens     []int
	Logits     [][]float32
	Confidence []float32
}

// Propose drafts the tokens after token, which is at the position after
// the last main position seen. head is the main model's.
func (s *DraftState) Propose(token int, head *Head) (*Proposal, error) {
	d, c := s.d, s.d.m.Cfg
	B, dim, hc, hd, rd := c.DraftBlock, c.Dim, c.HCMult, c.HeadDim, c.RopeDim
	if s.main == 0 {
		return nil, errors.New("dsv41: a draft before any main position")
	}
	if token < 0 || token >= c.Vocab {
		return nil, fmt.Errorf("dsv41: token %d", token)
	}
	start := s.main // the block's first position
	st := s.st
	hs := make([][]float32, B)
	preMix := make([][]float32, B)
	for j := range B {
		t := c.DraftNoise
		if j == 0 {
			t = token
		}
		hs[j] = make([]float32, hc*dim)
		for k := range hc {
			for e := range dim {
				hs[j][k*dim+e] = num.BF16(head.embed[t*dim+e])
			}
		}
		preMix[j] = make([]float32, hc)
		preMix[j][0] = 1
	}
	for l := range c.MTPLayers {
		b := d.m.layers[c.Layers+l]
		a := b.attn
		// The attention's inputs, queries and keys of all positions first:
		// each position attends to all of the block's.
		res := make([][]float32, B)
		post := make([][]float32, B)
		comb := make([][]float32, B)
		aPre := make([][]float32, B)
		qs := make([][]float32, B)
		kvs := make([][]float32, B)
		for j := range B {
			res[j] = append([]float32(nil), hs[j]...)
			var pre []float32
			pre, post[j], comb[j] = st.mixes(&b.hcAttn, hs[j])
			aPre[j] = pre
			x := make([]float32, dim)
			hcPre(x, hs[j], preMix[j])
			rmsNorm(x, x, b.attnNorm, c.NormEps)
			qr := make([]float32, c.QLoRA)
			a.wqA.apply(qr, x, &st.act)
			rmsNorm(qr, qr, a.qNorm, c.NormEps)
			q := make([]float32, c.Heads*hd)
			a.wqB.apply(q, qr, &st.act)
			for h := range c.Heads {
				a.rope.apply(q[(h+1)*hd-rd:(h+1)*hd], start+j, false)
			}
			kv := make([]float32, hd)
			a.wkv.apply(kv, x, &st.act)
			rmsNorm(kv, kv, a.kvNorm, c.NormEps)
			a.rope.apply(kv[hd-rd:], start+j, false)
			num.ActQuantInPlace(kv)
			qs[j], kvs[j] = q, kv
		}
		// The window's slots in their order (get_dspark_topk_idxs), then
		// the block's keys.
		var rows [][]float32
		for slot := range min(c.Window, start) {
			rows = append(rows, s.window[l][slot])
		}
		rows = append(rows, kvs...)
		for j := range B {
			o := st.sparseAttention(qs[j], rows, a.sink)
			for h := range c.Heads {
				a.rope.apply(o[(h+1)*hd-rd:(h+1)*hd], start+j, true)
			}
			gw := c.Heads * hd / c.OGroups
			low := make([]float32, c.OGroups*c.OLoRA)
			for g := range c.OGroups {
				og := o[g*gw : (g+1)*gw]
				num.ActQuantInPlace(og)
				num.QuantizeActFP8(og, &st.act)
				a.woA.applyRowsFP8(low[g*c.OLoRA:(g+1)*c.OLoRA], g*c.OLoRA, &st.act)
			}
			y := make([]float32, dim)
			a.woB.apply(y, low, &st.act)
			hcPost(hs[j], y, res[j], post[j], comb[j])

			copy(res[j], hs[j])
			fPre, fPost, fComb := st.mixes(&b.hcFFN, hs[j])
			x := make([]float32, dim)
			hcPre(x, hs[j], aPre[j])
			rmsNorm(x, x, b.ffnNorm, c.NormEps)
			y = st.moe(b.ffn, x)
			hcPost(hs[j], y, res[j], fPost, fComb)
			preMix[j] = fPre
		}
		if st.AfterBlock != nil {
			for j := range B {
				st.AfterBlock(b.id, start+j, hs[j], preMix[j])
			}
		}
	}
	// The head: the main model's classifier on the draft's norm, the
	// Markov head's logits added, tokens chosen greedily one after another.
	p := &Proposal{Tokens: []int{token}}
	rank := c.DraftRank
	feat := make([]float32, dim+rank)
	for j := range B {
		x := make([]float32, dim)
		hcPre(x, hs[j], preMix[j])
		copy(feat, x)
		n := make([]float32, dim)
		rmsNorm(n, x, d.norm, c.NormEps)
		logits := make([]float32, c.Vocab)
		head.head.apply(logits, n, nil)
		prev := p.Tokens[j]
		emb := make([]float32, rank)
		for e := range rank {
			emb[e] = num.BF16(d.markovEmbed[prev*rank+e])
		}
		bias := make([]float32, c.Vocab)
		d.markovHead.apply(bias, emb, nil)
		for v := range logits {
			logits[v] += bias[v]
		}
		p.Tokens = append(p.Tokens, argmax(logits))
		p.Logits = append(p.Logits, logits)
		copy(feat[dim:], emb)
		conf := make([]float32, 1)
		d.confidence.apply(conf, feat, nil)
		p.Confidence = append(p.Confidence, conf[0])
	}
	return p, nil
}

// argmax is the first index of the largest value, as torch's.
func argmax(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}
