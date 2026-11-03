package dsv41

import (
	"errors"
	"fmt"
	"math"

	"github.com/optimist3301/rakkan-mesh/internal/gpu"
)

// A part's blocks on a GPU (package gpu, any Vulkan device): GPUPart holds
// their weights in device memory, GPUStage one generation's caches and
// scratch in an arena. GPUStage.Forward computes what Stage.Forward does,
// bit for bit: each kernel repeats the CPU's arithmetic, and the shared
// state of compressed attention (selections, candidate pool, cache
// entries) stays on the device within a step, leaving it in the Payload.
// Only Engram's table rows are read on the host, from wherever the table
// is, and copied in.

// GPUPart is a part's weights on a device.
type GPUPart struct {
	m      *Model
	d      *gpu.Device
	blocks []*gpuBlock // by layer; nil outside the part
	vecs   []float32   // the blocks' vectors, copied into each stage's arena
	mats   []*gpu.Matrix
}

type gpuBlock struct {
	b                          *block
	hcAttn, hcFFN              *gpu.Matrix
	wqA, wqB, wkv, woB         *gpu.Matrix
	woA                        []*gpu.Matrix // per output group
	compWKV, compGate          *gpu.Matrix
	idxWK, idxWqB, idxW        *gpu.Matrix
	gate, w1, w2, w3           *gpu.Matrix // routed experts stacked
	s1, s2, s3                 *gpu.Matrix // the shared expert
	ewkv                       *gpu.Matrix
	attnNorm, ffnNorm          int // offsets in vecs
	aBase, aScale, fBase, fScl int
	qNorm, kvNorm, sink        int
	compNorm, idxKNorm         int
	gateBias, qk               int
}

// ToGPU uploads the part's blocks to a device.
func (m *Model) ToGPU(d *gpu.Device) (*GPUPart, error) {
	c := m.Cfg
	if c.HCMult != 4 {
		return nil, errors.New("dsv41: the GPU kernels take 4 stream copies")
	}
	p := &GPUPart{m: m, d: d, blocks: make([]*gpuBlock, c.Layers)}
	fail := func(err error) (*GPUPart, error) {
		p.Free()
		return nil, err
	}
	for i := m.from; i < m.to; i++ {
		g, err := p.upload(m.layers[i])
		if err != nil {
			return fail(fmt.Errorf("dsv41: layer %d to the GPU: %w", i, err))
		}
		p.blocks[i] = g
	}
	return p, nil
}

// Free releases the part's device memory.
func (p *GPUPart) Free() {
	for _, m := range p.mats {
		m.Free()
	}
	p.mats = nil
}

func (p *GPUPart) vec(v []float32) int {
	off := len(p.vecs)
	p.vecs = append(p.vecs, v...)
	return off
}

func (p *GPUPart) mat(rows, cols int, f gpu.Format) (*gpu.Matrix, error) {
	m, err := p.d.Upload(rows, cols, f)
	if err == nil {
		p.mats = append(p.mats, m)
	}
	return m, err
}

// format is a linear's weights as package gpu takes them.
func format(l *linear) gpu.Format {
	switch l.kind {
	case kindFP8:
		return gpu.Format{FP8: l.q, E8M0: l.scales}
	case kindFP4:
		return gpu.Format{FP4: l.q, E8M0: l.scales}
	case kindBF16, kindBF16F32:
		return gpu.Format{BF16: l.bf}
	}
	return gpu.Format{F32: l.f}
}

func (p *GPUPart) lin(l *linear) (*gpu.Matrix, error) {
	if l == nil {
		return nil, nil
	}
	return p.mat(l.rows, l.cols, format(l))
}

func (p *GPUPart) upload(b *block) (*gpuBlock, error) {
	c := p.m.Cfg
	g := &gpuBlock{b: b}
	a := b.attn
	g.attnNorm, g.ffnNorm = p.vec(b.attnNorm), p.vec(b.ffnNorm)
	g.aBase, g.aScale = p.vec(b.hcAttn.base), p.vec(b.hcAttn.scale)
	g.fBase, g.fScl = p.vec(b.hcFFN.base), p.vec(b.hcFFN.scale)
	g.qNorm, g.kvNorm, g.sink = p.vec(a.qNorm), p.vec(a.kvNorm), p.vec(a.sink)
	if a.compNorm != nil {
		g.compNorm, g.idxKNorm = p.vec(a.compNorm), p.vec(a.idxKNorm)
	}
	g.gateBias = p.vec(b.ffn.bias)
	var err error
	set := func(dst **gpu.Matrix, l *linear) {
		if err == nil {
			*dst, err = p.lin(l)
		}
	}
	set(&g.hcAttn, b.hcAttn.fn)
	set(&g.hcFFN, b.hcFFN.fn)
	set(&g.wqA, a.wqA)
	set(&g.wqB, a.wqB)
	set(&g.wkv, a.wkv)
	set(&g.woB, a.woB)
	set(&g.compWKV, a.compWKV)
	set(&g.compGate, a.compGate)
	set(&g.idxWK, a.idxWK)
	set(&g.idxWqB, a.idxWqB)
	set(&g.idxW, a.idxWeights)
	set(&g.gate, b.ffn.gate)
	set(&g.s1, b.ffn.shared.w1)
	set(&g.s2, b.ffn.shared.w2)
	set(&g.s3, b.ffn.shared.w3)
	if err != nil {
		return nil, err
	}
	// wo_a by output group: each its own product, as on the CPU.
	cols := a.woA.cols
	for gi := range c.OGroups {
		lo, hi := gi*c.OLoRA, (gi+1)*c.OLoRA
		nb := cols / 32
		m, err := p.mat(c.OLoRA, cols, gpu.Format{FP8: a.woA.q[lo*cols : hi*cols], E8M0: a.woA.scales[lo/32*nb : hi/32*nb]})
		if err != nil {
			return nil, err
		}
		g.woA = append(g.woA, m)
	}
	// The routed experts, stacked: an expert's rows in one part.
	stack := func(get func(e *expert) *linear) (*gpu.Matrix, error) {
		first := get(&b.ffn.experts[0])
		var q, s []byte
		for i := range b.ffn.experts {
			l := get(&b.ffn.experts[i])
			q, s = append(q, l.q...), append(s, l.scales...)
		}
		return p.mat(first.rows*len(b.ffn.experts), first.cols, gpu.Format{FP4: q, E8M0: s, AlignRows: first.rows})
	}
	if g.w1, err = stack(func(e *expert) *linear { return e.w1 }); err != nil {
		return nil, err
	}
	if g.w2, err = stack(func(e *expert) *linear { return e.w2 }); err != nil {
		return nil, err
	}
	if g.w3, err = stack(func(e *expert) *linear { return e.w3 }); err != nil {
		return nil, err
	}
	if e := b.engram; e != nil {
		g.qk = p.vec(e.qk)
		if g.ewkv, err = p.lin(e.wkv); err != nil {
			return nil, err
		}
	}
	return g, nil
}

// GPUStage is one generation on a GPUPart: Stage on the GPU.
type GPUStage struct {
	p         *GPUPart
	a         *gpu.Arena
	positions int
	pos       int
	l         gpuLayout
	compN     []int // entries in each source's caches
	// The state of the current step's shared attention, as State's.
	topkN               int
	topkSet, candSet    bool
	candLen             int
	source              int
	newKV, newIdx       bool
	maxEntries, maxRows int
}

// gpuLayout is where things are in a stage's arena, in words.
type gpuLayout struct {
	vecs                              int
	h, res, preMix, mix               int
	aPre, aPost, aComb, fPre, fPost   int
	fComb, x, y                       int
	qr, q, o, low, acc, rows, cs      int
	latent, iq, iw, scores, bs, sel   int
	cand, topk, flags                 int
	z, score, biased, esel, order, ew int
	eg, eu, eouts, sg, su, sout       int
	eqx, exs                          int
	emb, ekv                          int
	qx, xs                            int
	window, comp, idx                 []int // by layer
	kvSlots, scSlots                  []int // the compressors' open groups
	words                             int
}

// csSlots are the rotary tables a step uses: plain and compressed at the
// position, compressed at the one before (a group's first, for ratio 2).
const csSlots = 3

// NewStage starts a generation of up to positions on the part.
func (p *GPUPart) NewStage(positions int) (*GPUStage, error) {
	c := p.m.Cfg
	s := &GPUStage{p: p, positions: positions, compN: make([]int, c.Layers), source: -1}
	l := &s.l
	at := 0
	alloc := func(n int) int {
		off := at
		at += (n + 3) &^ 3
		return off
	}
	dim, hc, hd := c.Dim, c.HCMult, c.HeadDim
	inter := c.MoEInter
	k := c.ActiveExperts
	maxCols := max(dim, hd*c.Heads/c.OGroups, c.OGroups*c.OLoRA, c.QLoRA, inter, hc*dim)
	if c.EngramDim > 0 {
		maxCols = max(maxCols, (c.EngramMaxNgram-1)*c.EngramHeads*c.EngramDim)
	}
	s.maxEntries = positions + 1
	s.maxRows = c.Window
	l.vecs = alloc(len(p.vecs))
	l.h, l.res, l.preMix, l.mix = alloc(hc*dim), alloc(hc*dim), alloc(hc), alloc((2+hc)*hc)
	l.aPre, l.aPost, l.aComb = alloc(hc), alloc(hc), alloc(hc*hc)
	l.fPre, l.fPost, l.fComb = alloc(hc), alloc(hc), alloc(hc*hc)
	l.x, l.y = alloc(dim), alloc(dim)
	l.qr, l.q, l.o = alloc(c.QLoRA), alloc(c.Heads*hd), alloc(c.Heads*hd)
	l.low, l.acc = alloc(c.OGroups*c.OLoRA), alloc(c.Heads*hd)
	l.rows, l.cs = alloc(c.Window), alloc(csSlots*c.RopeDim)
	l.latent = alloc(hd)
	l.iq, l.iw = alloc(c.IndexHeads*c.IndexDim), alloc(c.IndexHeads)
	l.scores, l.bs, l.sel = alloc(positions), alloc(positions), alloc(positions)
	l.cand, l.topk, l.flags = alloc(positions), alloc(max(c.IndexTopK, 1)), alloc(positions)
	E := c.RoutedExperts
	l.z, l.score, l.biased = alloc(E), alloc(E), alloc(E)
	l.esel, l.order, l.ew = alloc(k), alloc(k), alloc(k)
	l.eg, l.eu, l.eouts = alloc(k*inter), alloc(k*inter), alloc(k*dim)
	l.sg, l.su, l.sout = alloc(inter), alloc(inter), alloc(dim)
	l.eqx, l.exs = alloc(k*max(dim, inter)), alloc(k*max(dim, inter)/32)
	if c.EngramDim > 0 {
		l.emb, l.ekv = alloc((c.EngramMaxNgram-1)*c.EngramHeads*c.EngramDim), alloc(dim*(hc+1))
	}
	l.qx, l.xs = alloc(maxCols), alloc(maxCols/32)
	l.window = make([]int, c.Layers)
	l.comp, l.idx = make([]int, c.Layers), make([]int, c.Layers)
	l.kvSlots, l.scSlots = make([]int, c.Layers), make([]int, c.Layers)
	src := sourceBefore(c, p.m.from)
	for i := range c.Layers {
		inPart := i >= p.m.from && i < p.m.to
		if inPart {
			l.window[i] = alloc(c.Window * hd)
		}
		if (inPart && c.isKVSource(i)) || i == src {
			r := c.CompressRatios[i]
			n := positions/r + 1
			l.comp[i], l.idx[i] = alloc(n*hd), alloc(n*c.IndexDim)
			if inPart && r > 1 {
				l.kvSlots[i], l.scSlots[i] = alloc(r*hd), alloc(r*hd)
			}
		}
	}
	l.words = at
	staging := 2*hc*dim + 2*(hd+c.IndexDim) + 2*positions + c.Layers*(c.Window+4) + csSlots*c.RopeDim + 2*max(c.IndexTopK, 1) + 4096
	if c.EngramDim > 0 {
		staging += (c.EngramMaxNgram - 1) * c.EngramHeads * c.EngramDim * len(c.EngramLayers)
	}
	a, err := p.d.NewArena(at, max(staging, len(p.vecs)))
	if err != nil {
		return nil, err
	}
	a.SetQuantScratch(l.qx, l.xs)
	s.a = a
	b := a.Begin()
	b.Write(l.vecs, p.vecs)
	if err := b.Submit(); err != nil {
		a.Free()
		return nil, err
	}
	return s, nil
}

// Close releases the stage's arena.
func (s *GPUStage) Close() {
	if s.a != nil {
		s.a.Free()
		s.a = nil
	}
}

// Pos is the next position the stage expects.
func (s *GPUStage) Pos() int { return s.pos }

func words(v []int) []uint32 {
	out := make([]uint32, len(v))
	for i, x := range v {
		out[i] = uint32(int32(x))
	}
	return out
}

// Forward runs the part's blocks on the Payload of position pos, as
// Stage.Forward does.
func (s *GPUStage) Forward(p *Payload, pos int) error {
	m, c := s.p.m, s.p.m.Cfg
	l := &s.l
	if pos != s.pos {
		return fmt.Errorf("dsv41: position %d, the stage expects %d", pos, s.pos)
	}
	if pos >= s.positions {
		return fmt.Errorf("dsv41: position %d beyond the stage's %d", pos, s.positions)
	}
	if want := sourceBefore(c, m.from); p.Source != want {
		return fmt.Errorf("dsv41: payload from the KV source %d, layer %d follows %d", p.Source, m.from, want)
	}
	if p.Prefill > s.positions || len(p.Candidates) > s.positions {
		return errors.New("dsv41: payload beyond the stage's positions")
	}
	b := s.a.Begin()
	b.Write(l.h, p.H)
	b.Write(l.preMix, p.PreMix)
	s.source, s.newKV, s.newIdx = p.Source, false, false
	if src := p.Source; src >= 0 {
		n := s.compN[src]
		if p.CompKV != nil {
			if n >= s.maxEntries {
				return errors.New("dsv41: more compressed entries than positions")
			}
			b.Write(l.comp[src]+n*c.HeadDim, p.CompKV)
			s.newKV = true
		}
		if p.IndexK != nil {
			b.Write(l.idx[src]+n*c.IndexDim, p.IndexK)
			s.newIdx = true
		}
		if p.CompKV != nil {
			s.compN[src]++
		}
		if (p.CompKV == nil) != (p.IndexK == nil) {
			// Every KV source is an index source: both come, or neither.
			return errors.New("dsv41: a compressed entry without its indexer key")
		}
		for _, g := range p.TopK {
			if g >= s.compN[src] {
				return fmt.Errorf("dsv41: selected entry %d of %d", g, s.compN[src])
			}
		}
	} else if len(p.TopK) > 0 || p.CompKV != nil || p.IndexK != nil {
		return errors.New("dsv41: compressed entries before any KV source")
	}
	s.topkN, s.topkSet = len(p.TopK), p.TopK != nil
	if s.topkSet {
		b.WriteWords(l.topk, words(p.TopK))
	}
	s.candSet, s.candLen = p.Candidates != nil, len(p.Candidates)
	if s.candSet {
		flags := make([]uint32, len(p.Candidates))
		for i, v := range p.Candidates {
			if v {
				flags[i] = 1
			}
		}
		b.WriteWords(l.cand, flags)
	}
	// The rotary tables of the step.
	rd := c.RopeDim
	b.Write(l.cs, m.ropes[0].at(pos))
	b.Write(l.cs+rd, m.ropes[1].at(pos))
	if pos > 0 {
		b.Write(l.cs+2*rd, m.ropes[1].at(pos-1))
	}
	for i := m.from; i < m.to; i++ {
		if err := s.block(b, s.p.blocks[i], p, pos); err != nil {
			b.Submit()
			return err
		}
	}
	// What leaves: the stream, the mix, the shared state.
	out := make([]float32, c.HCMult*c.Dim)
	mix := make([]float32, c.HCMult)
	b.Read(l.h, out)
	b.Read(l.preMix, mix)
	var kv, ik, topk, cand []float32
	if s.source >= 0 && s.source >= m.from {
		n := s.compN[s.source] - 1
		if s.newKV {
			kv = make([]float32, c.HeadDim)
			b.Read(l.comp[s.source]+n*c.HeadDim, kv)
		}
		if s.newIdx {
			ik = make([]float32, c.IndexDim)
			b.Read(l.idx[s.source]+n*c.IndexDim, ik)
		}
	}
	if s.topkSet && s.topkN > 0 {
		topk = make([]float32, s.topkN)
		b.Read(l.topk, topk)
	}
	if s.candSet && s.candLen > 0 {
		cand = make([]float32, s.candLen)
		b.Read(l.cand, cand)
	}
	if err := b.Submit(); err != nil {
		return err
	}
	s.pos++
	p.H, p.PreMix = out, mix
	if s.source >= m.from {
		p.Source, p.CompKV, p.IndexK = s.source, kv, ik
	}
	if s.topkSet {
		p.TopK = make([]int, s.topkN)
		for i, v := range topk {
			p.TopK[i] = int(int32(math.Float32bits(v)))
		}
	}
	if s.candSet {
		p.Candidates = make([]bool, s.candLen)
		for i, v := range cand {
			p.Candidates[i] = math.Float32bits(v) != 0
		}
	}
	return nil
}

// block records one block, as State.block.
func (s *GPUStage) block(b *gpu.Batch, g *gpuBlock, p *Payload, pos int) error {
	c := s.p.m.Cfg
	l := &s.l
	dim, hc := c.Dim, c.HCMult
	v := l.vecs
	if e := g.b.engram; e != nil {
		emb, err := e.embeddings(p.Hashes[e.index], c)
		if err != nil {
			return err
		}
		b.Write(l.emb, emb)
		b.MatVecs([]*gpu.Matrix{g.ewkv}, []int{l.ekv}, l.emb)
		b.EngramGate(dim, l.h, l.ekv, v+g.qk, c.NormEps, float32(1/math.Sqrt(float64(dim))))
	}
	b.Copy(l.h, l.res, hc*dim)
	b.MatVecs([]*gpu.Matrix{g.hcAttn}, []int{l.mix}, l.h)
	b.HCSplit(l.mix, l.h, hc*dim, v+g.aBase, v+g.aScale, c.HCSinkhornIters, c.HCEps, c.NormEps, l.aPre, l.aPost, l.aComb)
	b.HCPre(dim, l.h, l.preMix, l.x)
	b.DSNorm(l.x, l.x, v+g.attnNorm, dim, 1, 0, c.NormEps)
	s.attention(b, g, pos, p.Prefill)
	b.HCPost(dim, l.h, l.y, l.res, l.aPost, l.aComb)

	b.Copy(l.h, l.res, hc*dim)
	b.MatVecs([]*gpu.Matrix{g.hcFFN}, []int{l.mix}, l.h)
	b.HCSplit(l.mix, l.h, hc*dim, v+g.fBase, v+g.fScl, c.HCSinkhornIters, c.HCEps, c.NormEps, l.fPre, l.fPost, l.fComb)
	b.HCPre(dim, l.h, l.aPre, l.x)
	b.DSNorm(l.x, l.x, v+g.ffnNorm, dim, 1, 0, c.NormEps)
	s.moe(b, g)
	b.HCPost(dim, l.h, l.y, l.res, l.fPost, l.fComb)
	b.Copy(l.fPre, l.preMix, hc)
	return nil
}

// bf16Lin records a bfloat16 product: rounded to bfloat16 for kindBF16.
func bf16Lin(b *gpu.Batch, m *gpu.Matrix, lin *linear, out, in int) {
	b.MatVecs([]*gpu.Matrix{m}, []int{out}, in)
	if lin.kind == kindBF16 {
		b.RoundScale(out, lin.rows, 1)
	}
}

// attention records State.attention for the block's input at x into y.
func (s *GPUStage) attention(b *gpu.Batch, g *gpuBlock, pos, prefillN int) {
	c := s.p.m.Cfg
	l := &s.l
	a := g.b.attn
	id := g.b.id
	hd, rd, v := c.HeadDim, c.RopeDim, l.vecs
	cs := l.cs // plain rope at pos
	if a.ratio > 0 {
		cs = l.cs + rd // compressed
	}
	csGroup := cs // at pos+1-ratio
	if a.ratio == 2 {
		csGroup = l.cs + 2*rd
	} else if a.ratio > 2 {
		panic("dsv41: GPU compression ratios above 2 are not supported")
	}

	b.MatVecs([]*gpu.Matrix{g.wqA}, []int{l.qr}, l.x)
	b.DSNorm(l.qr, l.qr, v+g.qNorm, c.QLoRA, 1, 0, c.NormEps)
	b.MatVecs([]*gpu.Matrix{g.wqB}, []int{l.q}, l.qr)
	b.RopeDS(l.q+hd-rd, rd/2, c.Heads, hd, cs, false)

	slot := l.window[id] + (pos%c.Window)*hd
	b.MatVecs([]*gpu.Matrix{g.wkv}, []int{slot}, l.x)
	b.DSNorm(slot, slot, v+g.kvNorm, hd, 1, 0, c.NormEps)
	b.RopeDS(slot+hd-rd, rd/2, 1, 0, cs, false)
	b.ActQuantInPlace(slot, hd)

	rows := s.windowRows(id, pos, prefillN)
	nTop := 0
	if a.ratio > 0 {
		latent := false
		if a.kvSource {
			if a.ratio == 1 {
				bf16Lin(b, g.compWKV, a.compWKV, l.latent, l.x)
				latent = true
			} else {
				j := pos % a.ratio
				b.MatVecs([]*gpu.Matrix{g.compWKV, g.compGate}, []int{l.kvSlots[id] + j*hd, l.scSlots[id] + j*hd}, l.x)
				if (pos+1)%a.ratio == 0 {
					b.Compress(hd, a.ratio, l.kvSlots[id], l.scSlots[id], l.latent)
					latent = true
				}
			}
			if latent {
				b.DSNorm(l.latent, l.latent, v+g.compNorm, hd, 1, 0, c.NormEps)
			}
			s.source, s.newKV, s.newIdx = id, false, false
		}
		n := s.compN[id] // this source's entries before the step's
		if a.idxSrc {
			s.indexer(b, g, pos, prefillN, latent, n, csGroup, cs)
		}
		if latent {
			b.RopeDS(l.latent+hd-rd, rd/2, 1, 0, csGroup, false)
			b.FP4QuantInPlace(l.latent, hd, true)
			b.Copy(l.latent, l.comp[id]+n*hd, hd)
			s.compN[id]++
			s.newKV = true
		}
		if s.topkSet {
			nTop = s.topkN
		}
	}
	comp := 0
	if s.source >= 0 {
		comp = l.comp[s.source]
	}
	b.WriteWords(l.rows, words(rows))
	b.SparseAttention(c.Heads, hd, l.q, l.rows, len(rows), l.topk, nTop, comp, v+g.sink, float32(1/math.Sqrt(float64(hd))), l.acc, l.o)
	b.RopeDS(l.o+hd-rd, rd/2, c.Heads, hd, cs, true)
	gw := c.Heads * hd / c.OGroups
	for gi, m := range g.woA {
		og := l.o + gi*gw
		b.ActQuantInPlace(og, gw)
		b.MatVecs([]*gpu.Matrix{m}, []int{l.low + gi*c.OLoRA}, og)
	}
	b.MatVecs([]*gpu.Matrix{g.woB}, []int{l.y}, l.low)
}

// windowRows lists the window rows the position attends to, as
// State.windowRows, as word offsets (-1 for an empty slot).
func (s *GPUStage) windowRows(layer, pos, prefillN int) []int {
	c := s.p.m.Cfg
	win := c.Window
	ring := s.l.window[layer]
	row := func(slot int) int { return ring + slot*c.HeadDim }
	var rows []int
	if prefillN > 0 {
		start := max(pos-win+1, 0)
		for i := range min(prefillN, win) {
			if p := start + i; p <= pos {
				rows = append(rows, row(p%win))
			} else {
				rows = append(rows, -1)
			}
		}
		return rows
	}
	oldest := pos%win + 1
	for i := range win {
		slot := (oldest + i) % win
		if pos < win-1 && slot > pos {
			rows = append(rows, -1)
		} else {
			rows = append(rows, row(slot))
		}
	}
	return rows
}

// indexer records State.indexer; n is the source's entries before the step.
func (s *GPUStage) indexer(b *gpu.Batch, g *gpuBlock, pos, prefillN int, latent bool, n, csGroup, cs int) {
	c := s.p.m.Cfg
	l := &s.l
	a := g.b.attn
	id := g.b.id
	idim, rd, ih, v := c.IndexDim, c.RopeDim, c.IndexHeads, l.vecs
	if latent && a.kvSource {
		k := l.idx[id] + n*idim
		bf16Lin(b, g.idxWK, a.idxWK, k, l.latent)
		b.DSNorm(k, k, v+g.idxKNorm, idim, 1, 0, c.NormEps)
		b.RopeDS(k+idim-rd, rd/2, 1, 0, csGroup, false)
		b.FP4QuantInPlace(k, idim, false)
		s.newIdx = true
	}
	b.MatVecs([]*gpu.Matrix{g.idxWqB}, []int{l.iq}, l.qr)
	b.RopeDS(l.iq+idim-rd, rd/2, ih, idim, cs, false)
	b.FP4QuantInPlace(l.iq, ih*idim, false)
	bf16Lin(b, g.idxW, a.idxWeights, l.iw, l.x)
	b.RoundScale(l.iw, ih, float32(1/math.Sqrt(float64(idim))/math.Sqrt(float64(ih))))

	visible := (pos + 1) / a.ratio
	width := visible
	if prefillN > 0 {
		width = prefillN / a.ratio
	}
	keys := l.idx[s.source]
	switch {
	case id == c.CandidateSource:
		b.IndexScores(width, visible, l.iq, keys, l.iw, ih, idim, l.scores, 0, 0, false)
		if width > 0 {
			b.CandidatePool(width, visible, c.CandidateBlocks, c.CandidateBlockSize, l.scores, l.bs, l.sel, l.cand)
		}
		s.candSet, s.candLen = true, width
	case c.CandidateSource >= 0 && c.CandidateSource < id && s.candSet:
		b.IndexScores(width, visible, l.iq, keys, l.iw, ih, idim, l.scores, l.cand, s.candLen, true)
	default:
		b.IndexScores(width, visible, l.iq, keys, l.iw, ih, idim, l.scores, 0, 0, false)
	}
	k := min(c.IndexTopK, width)
	if k > 0 {
		b.TopKSorted(width, k, l.scores, l.topk, l.flags, visible)
	}
	s.topkSet, s.topkN = true, k
}

// moe records State.moe for the input at x into y.
func (s *GPUStage) moe(b *gpu.Batch, g *gpuBlock) {
	c := s.p.m.Cfg
	l := &s.l
	f := g.b.ffn
	k, E, inter, dim := f.active, len(f.experts), c.MoEInter, c.Dim
	b.MatVecs([]*gpu.Matrix{g.gate}, []int{l.z}, l.x)
	b.MoEScores(E, l.z, l.vecs+g.gateBias, f.scoreFn == "sigmoid", l.score, l.biased)
	b.TopK(E, k, l.biased, l.esel)
	b.MoEWeights(k, l.score, l.esel, f.normTopK, f.scale, l.order, l.ew)
	b.ExpertMatVecs(g.w1, inter, l.order, k, l.x, 0, l.eg, inter, l.eqx, l.exs)
	b.ExpertMatVecs(g.w3, inter, l.order, k, l.x, 0, l.eu, inter, l.eqx, l.exs)
	b.SiluMulDS(inter, l.eg, l.eu, f.limit, true, l.ew, k, inter)
	b.ExpertMatVecs(g.w2, dim, l.order, k, l.eg, inter, l.eouts, dim, l.eqx, l.exs)
	b.MatVecs([]*gpu.Matrix{g.s1, g.s3}, []int{l.sg, l.su}, l.x)
	b.SiluMulDS(inter, l.sg, l.su, f.limit, false, 0, 1, 0)
	b.MatVecs([]*gpu.Matrix{g.s2}, []int{l.sout}, l.sg)
	b.MoESum(dim, l.eouts, k, dim, l.sout, l.y)
}
