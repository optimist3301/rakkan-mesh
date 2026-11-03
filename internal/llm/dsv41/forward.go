package dsv41

import (
	"fmt"
	"math"
	"slices"

	"github.com/optimist3301/rakkan-mesh/internal/num"
)

// State is one generation's caches. Positions run from 0; the prompt goes
// through Prefill once, then each new token through Decode.
//
// Prefill computes each prompt position as Decode would, except where the
// reference's prefill lays out its sliding-window indices differently (the
// window of an early position is padded at the end rather than the start,
// and has min(prompt length, window) slots): that changes how attention's
// online softmax groups its entries, and so the rounding.
type State struct {
	m   *Model
	pos int

	window   [][][]float32 // [layer][slot][head dim]: each layer's ring of window KV
	compKV   [][][]float32 // [layer][group]: compressed KV of KV sources
	indexK   [][][]float32 // [layer][group]: indexer keys of KV sources
	compBuf  []compressorState
	tokens   []int // the last tokens, oldest first, for Engram's n-grams
	act      num.ActFP8
	prefillN int // prompt length while prefilling, else 0

	// Hooks, for tests: called after each block with the stream ([hc][dim])
	// and the next block's input mix, which they may overwrite; and with
	// each index source's selected compressed positions (-1: none).
	AfterBlock func(layer, pos int, h, preMix []float32)
	Indices    func(layer, pos int, idx []int)

	trace func(name string, v []float32) // for tests: sublayer outputs
}

type compressorState struct {
	kv, score [][]float32 // [slot][head dim], the open group's tokens
}

// NewState starts a generation.
func (m *Model) NewState() *State {
	c := m.Cfg
	s := &State{m: m}
	s.window = make([][][]float32, c.Layers)
	s.compKV = make([][][]float32, c.Layers)
	s.indexK = make([][][]float32, c.Layers)
	s.compBuf = make([]compressorState, c.Layers)
	for i := range c.Layers {
		s.window[i] = make([][]float32, c.Window)
		if r := c.CompressRatios[i]; r > 1 && c.isKVSource(i) {
			s.compBuf[i] = compressorState{kv: make([][]float32, r), score: make([][]float32, r)}
		}
	}
	return s
}

// Pos is the next position.
func (s *State) Pos() int { return s.pos }

// Prefill runs the prompt and returns the logits after its last token.
func (s *State) Prefill(tokens []int) []float32 {
	var logits []float32
	for i, t := range tokens {
		logits = s.step(t, len(tokens), i == len(tokens)-1)
	}
	return logits
}

// Decode runs one token and returns the logits after it.
func (s *State) Decode(token int) []float32 { return s.step(token, 0, true) }

// shared is what attention layers hand down within a step: the latest
// source's compressed KV and indexer keys, the latest selection and the
// candidate pool (the reference's SharedAttentionRuntime). source is the
// latest KV source run, -1 for none, and newKV and newIdxK what it cached
// at this position, if anything: a later pipeline stage's copy of its
// caches grows by them.
type shared struct {
	compKV, indexK *[][]float32
	topk           []int
	candidates     []bool
	source         int
	newKV, newIdxK []float32
}

// step runs a whole model on one token, in one process.
func (s *State) step(token, prefill int, wantLogits bool) []float32 {
	h := s.m.head
	s.tokens = append(s.tokens, token)
	if len(s.tokens) > max(s.m.Cfg.EngramMaxNgram, 1) {
		s.tokens = s.tokens[1:]
	}
	p, err := h.Embed(s.tokens, prefill)
	if err == nil {
		err = s.run(p, s.pos)
	}
	if err != nil {
		panic(err) // a whole model in memory: only a bad token gets here
	}
	if !wantLogits {
		return nil
	}
	return h.Logits(p)
}

// run runs the blocks the model has on the payload of position pos, which
// must be the state's next: the payload leaves as the next stage's input.
func (s *State) run(p *Payload, pos int) error {
	m, c := s.m, s.m.Cfg
	if pos != s.pos {
		return fmt.Errorf("dsv41: position %d, the stage expects %d", pos, s.pos)
	}
	if want := sourceBefore(c, m.from); p.Source != want {
		return fmt.Errorf("dsv41: payload from the KV source %d, layer %d follows %d", p.Source, m.from, want)
	}
	sh := shared{source: p.Source, newKV: p.CompKV, newIdxK: p.IndexK, topk: p.TopK, candidates: p.Candidates}
	if src := p.Source; src >= 0 {
		// A copy of an earlier stage's caches.
		if p.CompKV != nil {
			s.compKV[src] = append(s.compKV[src], p.CompKV)
		}
		if p.IndexK != nil {
			s.indexK[src] = append(s.indexK[src], p.IndexK)
		}
		if len(s.compKV[src]) > pos+1 || len(s.indexK[src]) > pos+1 {
			return fmt.Errorf("dsv41: more compressed entries than positions")
		}
		sh.compKV, sh.indexK = &s.compKV[src], &s.indexK[src]
		for _, g := range p.TopK {
			if g >= len(s.compKV[src]) {
				return fmt.Errorf("dsv41: selected entry %d of %d", g, len(s.compKV[src]))
			}
		}
	} else if len(p.TopK) > 0 || p.CompKV != nil || p.IndexK != nil {
		return fmt.Errorf("dsv41: compressed entries before any KV source")
	}
	s.pos++
	s.prefillN = p.Prefill
	for i := m.from; i < m.to; i++ {
		b := m.layers[i]
		if b.engram != nil {
			if err := b.engram.apply(p.H, p.Hashes[b.engram.index], c, &s.act); err != nil {
				return err
			}
		}
		p.PreMix = s.block(b, p.H, p.PreMix, pos, &sh)
		if s.AfterBlock != nil {
			s.AfterBlock(i, pos, p.H, p.PreMix)
		}
	}
	p.Source, p.CompKV, p.IndexK, p.TopK, p.Candidates = sh.source, sh.newKV, sh.newIdxK, sh.topk, sh.candidates
	return nil
}

// sourceBefore is the latest KV source before layer l, -1 for none: whose
// caches the layers from l on read until they reach a source of their own.
func sourceBefore(c *Config, l int) int {
	src := -1
	for _, k := range c.KVSources {
		if k < l && c.isKVSource(k) {
			src = max(src, k)
		}
	}
	return src
}

// mixes computes a sublayer's hyper-connection coefficients from the
// stream: mix = (fn · h) · rsqrt(mean(h²) + eps), then split.
func (s *State) mixes(p *hcParams, h []float32) (pre, post, comb []float32) {
	c := s.m.Cfg
	hc := c.HCMult
	mix := make([]float32, (2+hc)*hc)
	p.fn.apply(mix, h, nil)
	r := rsqrt(meanSquare(h) + c.NormEps)
	for i := range mix {
		mix[i] = float32(mix[i] * r)
	}
	pre, post, comb = make([]float32, hc), make([]float32, hc), make([]float32, hc*hc)
	splitSinkhorn(mix, p.scale, p.base, hc, c.HCSinkhornIters, c.HCEps, pre, post, comb)
	return pre, post, comb
}

// block runs one transformer block on the stream h in place and returns
// the input mix of the next block (docs/dev/deepseek-v41.md, section 5).
func (s *State) block(b *block, h, preMix []float32, pos int, sh *shared) []float32 {
	c := s.m.Cfg
	dim := c.Dim
	x := make([]float32, dim)

	res := slices.Clone(h)
	aPre, aPost, aComb := s.mixes(&b.hcAttn, h)
	hcPre(x, h, preMix)
	rmsNorm(x, x, b.attnNorm, c.NormEps)
	y := s.attention(b, x, pos, sh)
	if s.trace != nil {
		s.trace("attn", y)
	}
	hcPost(h, y, res, aPost, aComb)

	copy(res, h)
	fPre, fPost, fComb := s.mixes(&b.hcFFN, h)
	hcPre(x, h, aPre)
	rmsNorm(x, x, b.ffnNorm, c.NormEps)
	y = s.moe(b.ffn, x)
	if s.trace != nil {
		s.trace("ffn", y)
	}
	hcPost(h, y, res, fPost, fComb)
	return fPre
}

func (s *State) moe(f *moe, x []float32) []float32 {
	c := s.m.Cfg
	z := make([]float32, len(f.experts))
	f.gate.apply(z, x, nil)
	score := make([]float32, len(z))
	for e, v := range z {
		switch f.scoreFn {
		case "sqrtsoftplus":
			score[e] = sqrt32(softplus(v))
		case "sigmoid":
			score[e] = sigmoid(v)
		default:
			panic("dsv41: softmax gating not implemented")
		}
	}
	biased := make([]float32, len(z))
	for e := range z {
		biased[e] = score[e] + f.bias[e]
	}
	sel := topK(biased, f.active)
	if s.trace != nil {
		ids := make([]float32, len(sel))
		for i, e := range sel {
			ids[i] = float32(e)
		}
		s.trace("experts", ids)
	}
	w := make([]float32, len(sel))
	var sum float32
	for i, e := range sel {
		w[i] = score[e]
		sum += w[i]
	}
	for i := range w {
		if f.normTopK && len(sel) > 1 {
			w[i] /= sum + 1e-20
		}
		w[i] *= f.scale
	}
	// Experts in ascending order, then the shared one, summed in float32.
	order := slices.Clone(sel)
	slices.Sort(order)
	y := make([]float32, c.Dim)
	for _, e := range order {
		out := s.expert(&f.experts[e], x, w[slices.Index(sel, e)], true, f.limit)
		for d := range y {
			y[d] += out[d]
		}
	}
	out := s.expert(&f.shared, x, 0, false, f.limit)
	for d := range y {
		y[d] = bf16(y[d] + out[d])
	}
	return y
}

func (s *State) expert(e *expert, x []float32, weight float32, weighted bool, limit float32) []float32 {
	return runExpert(e, x, weight, weighted, limit, &s.act)
}

// runExpert is an expert's SwiGLU on x, its output weighted if weighted.
func runExpert(e *expert, x []float32, weight float32, weighted bool, limit float32, act *num.ActFP8) []float32 {
	g := make([]float32, e.w1.rows)
	u := make([]float32, e.w3.rows)
	e.w1.apply(g, x, act)
	e.w3.apply(u, x, act)
	for i := range g {
		gi, ui := g[i], u[i]
		if limit > 0 {
			ui = min(max(ui, -limit), limit)
			gi = min(gi, limit)
		}
		v := float32(silu(gi) * ui)
		if weighted {
			v = float32(weight * v)
		}
		g[i] = bf16(v)
	}
	out := make([]float32, e.w2.rows)
	e.w2.apply(out, g, act)
	return out
}

// attention runs a layer's attention for the token at pos, x its input
// (after attn_norm). docs/dev/deepseek-v41.md, section 6.
func (s *State) attention(b *block, x []float32, pos int, sh *shared) []float32 {
	c := s.m.Cfg
	a := b.attn
	hd, rd := c.HeadDim, c.RopeDim

	qr := make([]float32, c.QLoRA)
	a.wqA.apply(qr, x, &s.act)
	rmsNorm(qr, qr, a.qNorm, c.NormEps)
	q := make([]float32, c.Heads*hd)
	a.wqB.apply(q, qr, &s.act)
	for h := range c.Heads {
		a.rope.apply(q[(h+1)*hd-rd:(h+1)*hd], pos, false)
	}
	if s.trace != nil {
		s.trace("qr", qr)
		s.trace("q", q)
	}

	kv := make([]float32, hd)
	a.wkv.apply(kv, x, &s.act)
	rmsNorm(kv, kv, a.kvNorm, c.NormEps)
	a.rope.apply(kv[hd-rd:], pos, false)
	num.ActQuantInPlace(kv)
	if s.trace != nil {
		s.trace("kv", kv)
	}
	ring := s.window[b.id]
	ring[pos%c.Window] = kv

	rows := s.windowRows(ring, pos)
	if a.ratio > 0 {
		var latent []float32
		if a.kvSource {
			latent = s.compress(b, x, pos)
			sh.compKV, sh.indexK = &s.compKV[b.id], &s.indexK[b.id]
			sh.source, sh.newKV, sh.newIdxK = b.id, nil, nil
		}
		if a.idxSrc {
			sh.topk = s.indexer(b, x, qr, latent, pos, sh)
			if s.Indices != nil {
				s.Indices(b.id, pos, sh.topk)
			}
		}
		if latent != nil {
			a.rope.apply(latent[hd-rd:], pos+1-a.ratio, false)
			num.FP4QuantInPlace(latent, num.FP4ScaleE4M3)
			s.compKV[b.id] = append(s.compKV[b.id], latent)
			sh.newKV = latent
		}
		for _, g := range sh.topk {
			if g < 0 {
				rows = append(rows, nil)
			} else {
				rows = append(rows, (*sh.compKV)[g])
			}
		}
	}

	o := s.sparseAttention(q, rows, a.sink)
	if s.trace != nil {
		s.trace("o", o)
	}
	for h := range c.Heads {
		a.rope.apply(o[(h+1)*hd-rd:(h+1)*hd], pos, true)
	}
	gw := c.Heads * hd / c.OGroups
	low := make([]float32, c.OGroups*c.OLoRA)
	for g := range c.OGroups {
		og := o[g*gw : (g+1)*gw]
		num.ActQuantInPlace(og)
		num.QuantizeActFP8(og, &s.act)
		a.woA.applyRowsFP8(low[g*c.OLoRA:(g+1)*c.OLoRA], g*c.OLoRA, &s.act)
	}
	if s.trace != nil {
		s.trace("low", low)
	}
	y := make([]float32, c.Dim)
	a.woB.apply(y, low, &s.act)
	return y
}

// windowRows lists the window KV the token at pos attends to, in the
// reference's order (get_window_topk_idxs); nil marks an empty slot.
func (s *State) windowRows(ring [][]float32, pos int) [][]float32 {
	win := len(ring)
	var rows [][]float32
	if s.prefillN > 0 {
		// Prefill: the positions of the window in order, padded at the end,
		// min(prompt length, window) slots.
		start := max(pos-win+1, 0)
		for i := range min(s.prefillN, win) {
			if p := start + i; p <= pos {
				rows = append(rows, ring[p%win])
			} else {
				rows = append(rows, nil)
			}
		}
		return rows
	}
	// Decode: the ring from its oldest slot; slots beyond pos are empty.
	oldest := pos%win + 1
	for i := range win {
		slot := (oldest + i) % win
		if pos < win-1 && slot > pos {
			rows = append(rows, nil)
		} else {
			rows = append(rows, ring[slot])
		}
	}
	return rows
}

// compress feeds the token to the layer's compressor and returns the new
// compressed latent (before RoPE) when it completes a group, else nil.
func (s *State) compress(b *block, x []float32, pos int) []float32 {
	c := s.m.Cfg
	a := b.attn
	hd := c.HeadDim
	if a.ratio == 1 {
		kv := make([]float32, hd)
		a.compWKV.apply(kv, x, nil)
		rmsNorm(kv, kv, a.compNorm, c.NormEps)
		return kv
	}
	st := &s.compBuf[b.id]
	kv, sc := make([]float32, hd), make([]float32, hd)
	a.compWKV.apply(kv, x, nil)
	a.compGate.apply(sc, x, nil)
	slot := pos % a.ratio
	st.kv[slot], st.score[slot] = kv, sc
	if (pos+1)%a.ratio != 0 {
		return nil
	}
	// Per channel: a softmax over the group's tokens weighs their KV.
	out := make([]float32, hd)
	for d := range hd {
		m := negInf
		for j := range a.ratio {
			m = max(m, st.score[j][d])
		}
		var z float32
		e := make([]float32, a.ratio)
		for j := range a.ratio {
			e[j] = num.Exp32(st.score[j][d] - m)
			z += e[j]
		}
		var v float32
		for j := range a.ratio {
			v += float32(st.kv[j][d] * float32(e[j]/z))
		}
		out[d] = bf16(v)
	}
	rmsNorm(out, out, a.compNorm, c.NormEps)
	return out
}

// indexer scores the compressed positions visible at pos and returns the
// selected ones in ascending order, -1 for slots left empty
// (docs/dev/deepseek-v41.md, section 6.4).
func (s *State) indexer(b *block, x, qr, latent []float32, pos int, sh *shared) []int {
	c := s.m.Cfg
	a := b.attn
	id, rd, ih := c.IndexDim, c.RopeDim, c.IndexHeads
	if latent != nil && a.kvSource {
		k := make([]float32, id)
		a.idxWK.apply(k, latent, nil)
		rmsNorm(k, k, a.idxKNorm, c.NormEps)
		a.rope.apply(k[id-rd:], pos+1-a.ratio, false)
		num.FP4QuantInPlace(k, num.FP4ScaleE8M0)
		s.indexK[b.id] = append(s.indexK[b.id], k)
		sh.newIdxK = k
	}
	q := make([]float32, ih*id)
	a.idxWqB.apply(q, qr, &s.act)
	for h := range ih {
		a.rope.apply(q[(h+1)*id-rd:(h+1)*id], pos, false)
	}
	num.FP4QuantInPlace(q, num.FP4ScaleE8M0)
	w := make([]float32, ih)
	a.idxWeights.apply(w, x, nil)
	scale := float32(1 / math.Sqrt(float64(id)) / math.Sqrt(float64(ih)))
	for h := range w {
		w[h] = bf16(w[h] * scale)
	}

	visible := (pos + 1) / a.ratio
	width := visible // the reference scores the whole prompt's entries when prefilling
	if s.prefillN > 0 {
		width = s.prefillN / a.ratio
	}
	keys := *sh.indexK
	score := make([]float32, width)
	terms := make([]float32, ih)
	for t := range width {
		if t >= visible {
			score[t] = negInf
			continue
		}
		for h := range ih {
			d := bf16(num.Dot32(q[h*id:(h+1)*id], keys[t]))
			terms[h] = bf16(float32(max(d, 0) * w[h]))
		}
		var sum float32
		for _, v := range terms {
			sum += v
		}
		score[t] = bf16(sum)
	}

	if b.id == c.CandidateSource {
		sh.candidates = candidatePool(score, visible, c.CandidateBlocks, c.CandidateBlockSize)
	} else if c.CandidateSource >= 0 && c.CandidateSource < b.id && sh.candidates != nil {
		for t := range score {
			if t >= len(sh.candidates) || !sh.candidates[t] {
				score[t] = negInf
			}
		}
	}
	k := min(c.IndexTopK, width)
	sel := topK(score, k)
	slices.Sort(sel)
	for i, t := range sel {
		if t >= visible {
			sel[i] = -1
		}
	}
	return sel
}

// candidatePool selects the blocks of positions later indexers search:
// the best `blocks` by their best score, the block of the newest visible
// position always (select_candidate_blocks).
func candidatePool(score []float32, visible, blocks, size int) []bool {
	n := (len(score) + size - 1) / size
	bs := make([]float32, n)
	for i := range n {
		bs[i] = negInf
		for t := i * size; t < min((i+1)*size, len(score)); t++ {
			bs[i] = max(bs[i], score[t])
		}
	}
	if visible > 0 {
		bs[(visible-1)/size] = float32(math.Inf(1))
	}
	keep := make([]bool, len(score))
	for _, i := range topK(bs, min(blocks, n)) {
		if bs[i] == negInf {
			continue
		}
		for t := i * size; t < min((i+1)*size, len(score)); t++ {
			keep[t] = true
		}
	}
	return keep
}

// sparseAttention attends each head of q to rows (nil: an empty slot),
// one shared key/value per row, with a per-head sink, as the reference's
// sparse_attn kernel: an online softmax over blocks of 64 rows, the
// probabilities rounded to bf16 before they weigh the values.
func (s *State) sparseAttention(q []float32, rows [][]float32, sink []float32) []float32 {
	c := s.m.Cfg
	hd := c.HeadDim
	scale := float32(1 / math.Sqrt(float64(hd)))
	o := make([]float32, len(q))
	sc := make([]float32, 64)
	acc := make([]float32, hd)
	for h := range c.Heads {
		qh := q[h*hd : (h+1)*hd]
		m := float32(-1e30)
		var sum float32
		clear(acc)
		for t := 0; t < len(rows); t += 64 {
			blk := rows[t:min(t+64, len(rows))]
			bm := negInf
			for j, r := range blk {
				if r == nil {
					sc[j] = negInf
					continue
				}
				sc[j] = float32(num.Dot32(qh, r) * scale)
				bm = max(bm, sc[j])
			}
			prev := m
			m = max(prev, bm)
			rescale := num.Exp32(prev - m)
			var bsum float32
			for j := range blk {
				sc[j] = num.Exp32(sc[j] - m)
				bsum += sc[j]
			}
			sum = float32(sum*rescale) + bsum
			for d := range hd {
				var v float32
				for j, r := range blk {
					if r != nil {
						v += float32(bf16(sc[j]) * r[d])
					}
				}
				acc[d] = float32(acc[d]*rescale) + v
			}
		}
		sum += num.Exp32(sink[h] - m)
		for d := range hd {
			o[h*hd+d] = bf16(acc[d] / sum)
		}
	}
	return o
}
