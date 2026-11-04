package dsv41

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/optimist3301/rakkan-mesh/internal/gpu"
	"github.com/optimist3301/rakkan-mesh/internal/num"
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
	// hostExperts: the routed experts stay on the host (GPUOptions).
	hostExperts bool
}

// HostExperts reports whether the part's routed experts run on the host.
func (p *GPUPart) HostExperts() bool { return p.hostExperts }

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

// GPUOptions chooses what of a part goes to the GPU.
type GPUOptions struct {
	// HostExperts keeps the routed experts in RAM, computed on the CPU:
	// the GPU holds the rest of each block (about 2.5% of its weights for
	// DeepSeek-V4.1-Flash), selects the experts and receives their outputs.
	HostExperts bool
}

// ToGPU uploads the part's blocks to a device.
func (m *Model) ToGPU(d *gpu.Device, o GPUOptions) (*GPUPart, error) {
	c := m.Cfg
	if c.HCMult != 4 {
		return nil, errors.New("dsv41: the GPU kernels take 4 stream copies")
	}
	p := &GPUPart{m: m, d: d, blocks: make([]*gpuBlock, c.Layers), hostExperts: o.HostExperts}
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
		var q, s [][]byte
		for i := range b.ffn.experts {
			l := get(&b.ffn.experts[i])
			q, s = append(q, l.q), append(s, l.scales)
		}
		return p.mat(first.rows*len(b.ffn.experts), first.cols, gpu.Format{FP4Chunks: q, E8M0Chunks: s, AlignRows: first.rows})
	}
	if p.hostExperts {
		goto engram
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
engram:
	if e := b.engram; e != nil {
		g.qk = p.vec(e.qk)
		if g.ewkv, err = p.lin(e.wkv); err != nil {
			return nil, err
		}
	}
	return g, nil
}

// GPUStage is one generation on a GPUPart: Stage on the GPU. It takes
// positions in batches of up to GPUBatch (ForwardBatch): each block runs
// for all of them before the next, its E4M3 products reading the weights
// once for the batch, and its attention position by position, in order.
type GPUStage struct {
	p         *GPUPart
	a         *gpu.Arena
	positions int
	pos       int
	l         gpuLayout
	compN     []int // entries in each source's caches
	source    int   // the latest KV source run, as State's shared.source
	// The batch's positions' shared attention state, as State's.
	pp         []gpuPos
	maxEntries int
	// For rewinding (State.Rewind): the caches' lengths before each of the
	// last Rewindable positions, newest last; the rows they replaced are
	// in the arena's undo slots, by position mod Rewindable.
	hist  [][]int
	start []int      // compN at the batch's start
	act   num.ActFP8 // host experts' scratch
}

// gpuPos is a batch position's shared attention state.
type gpuPos struct {
	topkN, candLen   int
	topkSet, candSet bool
	mainN            int   // Main's entries, received and made
	appended         []int // layers whose caches it grew
	newKV, newIdx    bool
	entry            int // the index of the entry it cached, if new
}

// GPUBatch is how many positions ForwardBatch takes at once.
const GPUBatch = gpu.MatBatchMax

// gpuLayout is where things are in a stage's arena, in words. Arrays of
// the batch's positions are stride apart: the field's size.
type gpuLayout struct {
	vecs                            int
	h, res, preMix, mix             int
	aPre, aPost, aComb, fPre, fPost int
	fComb, x, y                     int
	qr, q, o, low, acc, rows, cs    int
	kvb, kvExt, scExt               int // the batch's window rows, a compressor's open groups
	latent, iq, iw, scores, bs, sel int
	cand, topk, flags               int
	z, score, biased, esel          int
	order, ew                       int
	eg, eu, eouts, sg, su, sout     int
	eqx, exs                        int
	emb, ekv, embLen                int
	main                            int // the batch's Payload.Main, all targets' room
	qx, xs, maxCols                 int
	window, comp, idx               []int // by layer
	undo                            []int // by layer: Rewindable sets of a ring row, a compressor's two
	kvSlots, scSlots                []int // the compressors' open groups between batches
	words                           int
}

// csSlots are the rotary tables a position uses: plain and compressed at
// the position, compressed at the one before (a group's first, ratio 2).
const csSlots = 3

// Bytes is the device memory the part's weights take.
func (p *GPUPart) Bytes() int64 {
	var n int64
	for _, m := range p.mats {
		n += m.Bytes()
	}
	return n
}

// StageBytes is the device memory a stage of positions takes.
func (p *GPUPart) StageBytes(positions int) int64 {
	l, _ := p.layout(positions)
	return 4 * int64(l.words)
}

// DropHostWeights lets the host copy of the weights on the GPU go: the
// CPU can no longer run the part. Routed experts kept on the host stay.
func (p *GPUPart) DropHostWeights() { p.m.dropHostWeights(p.hostExperts) }

func (m *Model) dropHostWeights(keepExperts bool) {
	drop := func(l *linear) {
		if l != nil {
			l.q, l.scales, l.bf, l.f = nil, nil, nil, nil
		}
	}
	for _, b := range m.layers {
		if b == nil {
			continue
		}
		a := b.attn
		for _, l := range []*linear{b.hcAttn.fn, b.hcFFN.fn, a.wqA, a.wqB, a.wkv, a.woA, a.woB, a.compWKV, a.compGate,
			a.idxWK, a.idxWqB, a.idxWeights, b.ffn.gate, b.ffn.shared.w1, b.ffn.shared.w2, b.ffn.shared.w3} {
			drop(l)
		}
		for i := range b.ffn.experts {
			if keepExperts {
				break
			}
			e := &b.ffn.experts[i]
			drop(e.w1)
			drop(e.w2)
			drop(e.w3)
		}
		if b.engram != nil {
			drop(b.engram.wkv)
		}
	}
	m.hostless = true
}

// NewStage starts a generation of up to positions on the part.
func (p *GPUPart) NewStage(positions int) (*GPUStage, error) {
	c := p.m.Cfg
	s := &GPUStage{p: p, positions: positions, compN: make([]int, c.Layers), source: -1, maxEntries: positions + 1}
	l, staging := p.layout(positions)
	s.l = l
	a, err := p.d.NewArena(l.words, staging)
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

// layout places a stage's data in its arena, and sizes its staging.
func (p *GPUPart) layout(positions int) (gpuLayout, int) {
	c := p.m.Cfg
	var lay gpuLayout
	l := &lay
	at := 0
	alloc := func(n int) int {
		off := at
		at += (n + 3) &^ 3
		return off
	}
	B := GPUBatch
	batch := func(n int) int { return alloc(B * n) }
	dim, hc, hd := c.Dim, c.HCMult, c.HeadDim
	inter, k := c.MoEInter, c.ActiveExperts
	l.maxCols = max(dim, hd*c.Heads/c.OGroups, c.OGroups*c.OLoRA, c.QLoRA, inter, hc*dim)
	if c.EngramDim > 0 {
		l.embLen = (c.EngramMaxNgram - 1) * c.EngramHeads * c.EngramDim
		l.maxCols = max(l.maxCols, l.embLen)
	}
	l.vecs = alloc(len(p.vecs))
	l.h, l.res, l.preMix, l.mix = batch(hc*dim), batch(hc*dim), batch(hc), alloc((2+hc)*hc)
	l.aPre, l.aPost, l.aComb = batch(hc), batch(hc), batch(hc*hc)
	l.fPre, l.fPost, l.fComb = batch(hc), batch(hc), batch(hc*hc)
	l.x, l.y = batch(dim), batch(dim)
	l.qr, l.q, l.o = batch(c.QLoRA), batch(c.Heads*hd), batch(c.Heads*hd)
	l.low, l.acc = batch(c.OGroups*c.OLoRA), alloc(c.Heads*hd)
	l.rows, l.cs = alloc(c.Window), batch(csSlots*c.RopeDim)
	l.kvb, l.kvExt, l.scExt = batch(hd), alloc((B+2)*hd), alloc((B+2)*hd)
	l.latent = alloc(hd)
	l.iq, l.iw = batch(c.IndexHeads*c.IndexDim), alloc(c.IndexHeads)
	l.scores, l.bs, l.sel = alloc(positions), alloc(positions), alloc(positions)
	l.cand, l.topk, l.flags = batch(positions), batch(max(c.IndexTopK, 1)), alloc(positions)
	E := c.RoutedExperts
	l.z, l.score, l.biased, l.esel = alloc(E), alloc(E), alloc(E), alloc(k)
	l.order, l.ew = batch(k), batch(k)
	l.eg, l.eu, l.eouts = batch(k*inter), batch(k*inter), batch(k*dim)
	l.sg, l.su, l.sout = batch(inter), batch(inter), batch(dim)
	l.eqx, l.exs = batch(k*max(dim, inter)), batch(k*max(dim, inter)/32)
	if c.EngramDim > 0 {
		l.emb, l.ekv = batch(l.embLen), batch(dim*(hc+1))
	}
	l.qx, l.xs = batch(l.maxCols), batch(l.maxCols/32)
	l.main = batch(max(len(c.DraftTargets), 1) * dim)
	l.window = make([]int, c.Layers)
	l.undo = make([]int, c.Layers)
	for i := p.m.from; i < p.m.to; i++ {
		l.undo[i] = alloc(Rewindable * 3 * hd)
	}
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
	staging := B*(2*hc*dim+2*(hd+c.IndexDim)+2*positions+2*max(c.IndexTopK, 1)+csSlots*c.RopeDim+c.Layers*(c.Window+4)+2*k*dim) + 4096
	if c.EngramDim > 0 {
		staging += B * l.embLen * len(c.EngramLayers)
	}
	return lay, max(staging, len(p.vecs))
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
func (s *GPUStage) Forward(p *Payload, pos int) error { return s.ForwardBatch([]*Payload{p}, pos) }

// ForwardBatch runs the part's blocks on the Payloads of positions pos,
// pos+1, ... (GPUBatch at most), as Stage.Forward on each in turn does.
func (s *GPUStage) ForwardBatch(ps []*Payload, pos int) error {
	m, c := s.p.m, s.p.m.Cfg
	l := &s.l
	B := len(ps)
	switch {
	case B == 0:
		return nil
	case B > GPUBatch:
		return fmt.Errorf("dsv41: %d positions in a batch, at most %d", B, GPUBatch)
	case pos > s.pos || s.pos-pos > len(s.hist):
		return fmt.Errorf("dsv41: position %d, the stage expects %d", pos, s.pos)
	case pos+B > s.positions:
		return fmt.Errorf("dsv41: positions to %d beyond the stage's %d", pos+B, s.positions)
	}
	hc, dim, rd := c.HCMult, c.Dim, c.RopeDim
	s.pp = make([]gpuPos, B)
	b := s.a.Begin()
	if pos < s.pos {
		s.rewind(b, pos)
	}
	s.start = slices.Clone(s.compN)
	fail := func(err error) error {
		b.Submit()
		return err
	}
	src := sourceBefore(c, m.from)
	n := 0
	if src >= 0 {
		n = s.compN[src]
	}
	for i, p := range ps {
		q := pos + i
		switch {
		case p.Source != src:
			return fail(fmt.Errorf("dsv41: payload from the KV source %d, layer %d follows %d", p.Source, m.from, src))
		case p.Prefill > s.positions || len(p.Candidates) > s.positions:
			return fail(errors.New("dsv41: payload beyond the stage's positions"))
		case (p.CompKV == nil) != (p.IndexK == nil):
			// Every KV source is an index source: both come, or neither.
			return fail(errors.New("dsv41: a compressed entry without its indexer key"))
		case src < 0 && (len(p.TopK) > 0 || p.CompKV != nil):
			return fail(errors.New("dsv41: compressed entries before any KV source"))
		}
		b.Write(l.h+i*hc*dim, p.H)
		b.Write(l.preMix+i*hc, p.PreMix)
		if p.WantMain {
			if len(p.Main)%dim != 0 || len(p.Main) > dim*len(c.DraftTargets) {
				return fail(errors.New("dsv41: payload does not fit the model"))
			}
			b.Write(l.main+i*len(c.DraftTargets)*dim, p.Main)
			s.pp[i].mainN = len(p.Main) / dim
		}
		if p.CompKV != nil {
			if n >= s.maxEntries {
				return fail(errors.New("dsv41: more compressed entries than positions"))
			}
			b.Write(l.comp[src]+n*c.HeadDim, p.CompKV)
			b.Write(l.idx[src]+n*c.IndexDim, p.IndexK)
			s.pp[i].newKV, s.pp[i].newIdx, s.pp[i].entry = true, true, n
			s.pp[i].appended = append(s.pp[i].appended, src)
			n++
		}
		for _, g := range p.TopK {
			if g >= n {
				return fail(fmt.Errorf("dsv41: selected entry %d of %d", g, n))
			}
		}
		pp := &s.pp[i]
		pp.topkN, pp.topkSet = len(p.TopK), p.TopK != nil
		if pp.topkSet {
			b.WriteWords(l.topk+i*max(c.IndexTopK, 1), words(p.TopK))
		}
		pp.candSet, pp.candLen = p.Candidates != nil, len(p.Candidates)
		if pp.candSet {
			flags := make([]uint32, len(p.Candidates))
			for j, v := range p.Candidates {
				if v {
					flags[j] = 1
				}
			}
			b.WriteWords(l.cand+i*s.positions, flags)
		}
		cs := l.cs + i*csSlots*rd
		b.Write(cs, m.ropes[0].at(q))
		b.Write(cs+rd, m.ropes[1].at(q))
		if q > 0 {
			b.Write(cs+2*rd, m.ropes[1].at(q-1))
		}
	}
	if src >= 0 {
		s.compN[src] = n
	}
	s.source = src
	for i := m.from; i < m.to; i++ {
		var err error
		if b, err = s.block(b, s.p.blocks[i], ps, pos); err != nil {
			if b != nil {
				b.Submit()
			}
			return err
		}
	}
	// What leaves: the stream, the mix, the shared state.
	type outs struct{ h, mix, kv, ik, topk, cand, main []float32 }
	res := make([]outs, B)
	for i := range B {
		o, pp := &res[i], &s.pp[i]
		o.h, o.mix = make([]float32, hc*dim), make([]float32, hc)
		b.Read(l.h+i*hc*dim, o.h)
		b.Read(l.preMix+i*hc, o.mix)
		if s.source >= m.from && pp.newKV {
			o.kv, o.ik = make([]float32, c.HeadDim), make([]float32, c.IndexDim)
			b.Read(l.comp[s.source]+pp.entry*c.HeadDim, o.kv)
			b.Read(l.idx[s.source]+pp.entry*c.IndexDim, o.ik)
		}
		if pp.topkSet && pp.topkN > 0 {
			o.topk = make([]float32, pp.topkN)
			b.Read(l.topk+i*max(c.IndexTopK, 1), o.topk)
		}
		if pp.candSet && pp.candLen > 0 {
			o.cand = make([]float32, pp.candLen)
			b.Read(l.cand+i*s.positions, o.cand)
		}
		if ps[i].WantMain && pp.mainN > 0 {
			o.main = make([]float32, pp.mainN*dim)
			b.Read(l.main+i*len(c.DraftTargets)*dim, o.main)
		}
	}
	if err := b.Submit(); err != nil {
		return err
	}
	// The caches' lengths before each position, for rewinding.
	counts := slices.Clone(s.start)
	for i := range B {
		s.hist = append(s.hist, slices.Clone(counts))
		for _, l := range s.pp[i].appended {
			counts[l]++
		}
	}
	if len(s.hist) > Rewindable {
		s.hist = s.hist[len(s.hist)-Rewindable:]
	}
	s.pos += B
	for i, p := range ps {
		o, pp := &res[i], &s.pp[i]
		p.H, p.PreMix = o.h, o.mix
		if p.WantMain {
			p.Main = o.main
		}
		if s.source >= m.from {
			p.Source, p.CompKV, p.IndexK = s.source, o.kv, o.ik
		}
		if pp.topkSet {
			p.TopK = make([]int, pp.topkN)
			for j, v := range o.topk {
				p.TopK[j] = int(int32(math.Float32bits(v)))
			}
		}
		if pp.candSet {
			p.Candidates = make([]bool, pp.candLen)
			for j, v := range o.cand {
				p.Candidates[j] = math.Float32bits(v) != 0
			}
		}
	}
	return nil
}

// block records one block for the batch, as State.block for each
// position, and returns the batch to go on with: a new one if host
// experts had it submitted.
func (s *GPUStage) block(b *gpu.Batch, g *gpuBlock, ps []*Payload, pos int) (*gpu.Batch, error) {
	c := s.p.m.Cfg
	l := &s.l
	B := len(ps)
	dim, hc := c.Dim, c.HCMult
	v := l.vecs
	h := func(i int) int { return l.h + i*hc*dim }
	x := func(i int) int { return l.x + i*dim }
	if e := g.b.engram; e != nil {
		for i, p := range ps {
			emb, err := e.embeddings(p.Hashes[e.index], c)
			if err != nil {
				return b, err
			}
			b.Write(l.emb+i*l.embLen, emb)
		}
		b.MatVecsBatch(g.ewkv, l.ekv, dim*(hc+1), l.emb, l.embLen, B, l.qx, l.xs)
		for i := range B {
			b.EngramGate(dim, h(i), l.ekv+i*dim*(hc+1), v+g.qk, c.NormEps, float32(1/math.Sqrt(float64(dim))))
		}
	}
	for i, p := range ps {
		if p.WantMain && slices.Contains(c.DraftTargets, g.b.id) {
			// The draft reads the attention input of its target layers.
			pp := &s.pp[i]
			b.HCMean(dim, h(i), l.main+(i*len(c.DraftTargets)+pp.mainN)*dim)
			pp.mainN++
		}
	}
	for i := range B {
		b.Copy(h(i), l.res+i*hc*dim, hc*dim)
		b.MatVecs([]*gpu.Matrix{g.hcAttn}, []int{l.mix}, h(i))
		b.HCSplit(l.mix, h(i), hc*dim, v+g.aBase, v+g.aScale, c.HCSinkhornIters, c.HCEps, c.NormEps, l.aPre+i*hc, l.aPost+i*hc, l.aComb+i*hc*hc)
		b.HCPre(dim, h(i), l.preMix+i*hc, x(i))
		b.DSNorm(x(i), x(i), v+g.attnNorm, dim, 1, 0, c.NormEps)
	}
	s.attention(b, g, ps, pos)
	for i := range B {
		b.HCPost(dim, h(i), l.y+i*dim, l.res+i*hc*dim, l.aPost+i*hc, l.aComb+i*hc*hc)
		b.Copy(h(i), l.res+i*hc*dim, hc*dim)
		b.MatVecs([]*gpu.Matrix{g.hcFFN}, []int{l.mix}, h(i))
		b.HCSplit(l.mix, h(i), hc*dim, v+g.fBase, v+g.fScl, c.HCSinkhornIters, c.HCEps, c.NormEps, l.fPre+i*hc, l.fPost+i*hc, l.fComb+i*hc*hc)
		b.HCPre(dim, h(i), l.aPre+i*hc, x(i))
		b.DSNorm(x(i), x(i), v+g.ffnNorm, dim, 1, 0, c.NormEps)
	}
	b, err := s.moe(b, g, B)
	if err != nil {
		return nil, err
	}
	for i := range B {
		b.HCPost(dim, h(i), l.y+i*dim, l.res+i*hc*dim, l.fPost+i*hc, l.fComb+i*hc*hc)
		b.Copy(l.fPre+i*hc, l.preMix+i*hc, hc)
	}
	return b, nil
}

// bf16Lin records a bfloat16 product: rounded to bfloat16 for kindBF16.
func bf16Lin(b *gpu.Batch, m *gpu.Matrix, lin *linear, out, in int) {
	b.MatVecs([]*gpu.Matrix{m}, []int{out}, in)
	if lin.kind == kindBF16 {
		b.RoundScale(out, lin.rows, 1)
	}
}

// attention records State.attention for the batch's inputs at x into y:
// the products for all positions at once, the rest position by position.
func (s *GPUStage) attention(b *gpu.Batch, g *gpuBlock, ps []*Payload, pos int) {
	c := s.p.m.Cfg
	l := &s.l
	a := g.b.attn
	id := g.b.id
	B := len(ps)
	hd, rd, v := c.HeadDim, c.RopeDim, l.vecs
	qDim, ql, lowDim := c.Heads*hd, c.QLoRA, c.OGroups*c.OLoRA
	if a.ratio > 2 {
		panic("dsv41: GPU compression ratios above 2 are not supported")
	}
	// The rotary tables of position i: plain or compressed, and at the
	// group's first position.
	cs := func(i int) int {
		if a.ratio > 0 {
			return l.cs + i*csSlots*rd + rd
		}
		return l.cs + i*csSlots*rd
	}
	csGroup := func(i int) int {
		if a.ratio == 2 {
			return l.cs + i*csSlots*rd + 2*rd
		}
		return cs(i)
	}

	b.MatVecsBatch(g.wqA, l.qr, ql, l.x, c.Dim, B, l.qx, l.xs)
	for i := range B {
		b.DSNorm(l.qr+i*ql, l.qr+i*ql, v+g.qNorm, ql, 1, 0, c.NormEps)
	}
	b.MatVecsBatch(g.wqB, l.q, qDim, l.qr, ql, B, l.qx, l.xs)
	b.MatVecsBatch(g.wkv, l.kvb, hd, l.x, c.Dim, B, l.qx, l.xs)
	if a.idxSrc {
		b.MatVecsBatch(g.idxWqB, l.iq, c.IndexHeads*c.IndexDim, l.qr, ql, B, l.qx, l.xs)
	}
	for i := range B {
		b.RopeDS(l.q+i*qDim+hd-rd, rd/2, c.Heads, hd, cs(i), false)
		kv := l.kvb + i*hd
		b.DSNorm(kv, kv, v+g.kvNorm, hd, 1, 0, c.NormEps)
		b.RopeDS(kv+hd-rd, rd/2, 1, 0, cs(i), false)
		b.ActQuantInPlace(kv, hd)
	}
	// A compressor's open group from before the batch, at the head of its
	// rows: ext row j is position base+j.
	base := pos
	if a.kvSource && a.ratio > 1 {
		base = pos - pos%a.ratio
		for q := base; q < pos; q++ {
			b.Copy(l.kvSlots[id]+(q%a.ratio)*hd, l.kvExt+(q-base)*hd, hd)
			b.Copy(l.scSlots[id]+(q%a.ratio)*hd, l.scExt+(q-base)*hd, hd)
		}
	}
	for i, p := range ps {
		q := pos + i
		pp := &s.pp[i]
		x := l.x + i*c.Dim
		rows := s.windowRows(id, q, p.Prefill, pos)
		nTop := 0
		if a.ratio > 0 {
			latent := false
			if a.kvSource {
				if a.ratio == 1 {
					bf16Lin(b, g.compWKV, a.compWKV, l.latent, x)
					latent = true
				} else {
					j := q - base
					b.MatVecs([]*gpu.Matrix{g.compWKV, g.compGate}, []int{l.kvExt + j*hd, l.scExt + j*hd}, x)
					if (q+1)%a.ratio == 0 {
						first := q + 1 - a.ratio - base
						b.Compress(hd, a.ratio, l.kvExt+first*hd, l.scExt+first*hd, l.latent)
						latent = true
					}
				}
				if latent {
					b.DSNorm(l.latent, l.latent, v+g.compNorm, hd, 1, 0, c.NormEps)
				}
				s.source = id
				pp.newKV, pp.newIdx = false, false
			}
			n := s.compN[id] // this source's entries before this position's
			if a.idxSrc {
				s.indexer(b, g, i, q, p.Prefill, latent, n, csGroup(i), cs(i))
			}
			if latent {
				b.RopeDS(l.latent+hd-rd, rd/2, 1, 0, csGroup(i), false)
				b.FP4QuantInPlace(l.latent, hd, true)
				b.Copy(l.latent, l.comp[id]+n*hd, hd)
				s.compN[id]++
				pp.appended = append(pp.appended, id)
				pp.newKV, pp.entry = true, n
			}
			if pp.topkSet {
				nTop = pp.topkN
			}
		}
		comp := 0
		if s.source >= 0 {
			comp = l.comp[s.source]
		}
		o := l.o + i*qDim
		b.WriteWords(l.rows, words(rows))
		b.SparseAttention(c.Heads, hd, l.q+i*qDim, l.rows, len(rows), l.topk+i*max(c.IndexTopK, 1), nTop, comp, v+g.sink,
			float32(1/math.Sqrt(float64(hd))), l.acc, o)
		b.RopeDS(o+hd-rd, rd/2, c.Heads, hd, cs(i), true)
		gw := qDim / c.OGroups
		for gi := range c.OGroups {
			b.ActQuantInPlace(o+gi*gw, gw)
		}
	}
	// The batch's window rows into the ring, and a compressor's groups.
	// What each position replaces goes to its undo slots first.
	for i := range B {
		q := pos + i
		u := l.undo[id] + (q%Rewindable)*3*hd
		ring := l.window[id] + (q%c.Window)*hd
		b.Copy(ring, u, hd)
		b.Copy(l.kvb+i*hd, ring, hd)
		if a.kvSource && a.ratio > 1 {
			kv, sc := l.kvSlots[id]+(q%a.ratio)*hd, l.scSlots[id]+(q%a.ratio)*hd
			b.Copy(kv, u+hd, hd)
			b.Copy(sc, u+2*hd, hd)
			b.Copy(l.kvExt+(q-base)*hd, kv, hd)
			b.Copy(l.scExt+(q-base)*hd, sc, hd)
		}
	}
	gw := qDim / c.OGroups
	for gi, m := range g.woA {
		b.MatVecsBatch(m, l.low+gi*c.OLoRA, lowDim, l.o+gi*gw, qDim, B, l.qx, l.xs)
	}
	b.MatVecsBatch(g.woB, l.y, c.Dim, l.low, lowDim, B, l.qx, l.xs)
}

// windowRows lists the window rows position pos attends to, as
// State.windowRows, as word offsets (-1 for an empty slot): the ring's for
// positions before the batch's first, first, the batch's own after.
func (s *GPUStage) windowRows(layer, pos, prefillN, first int) []int {
	c := s.p.m.Cfg
	win := c.Window
	row := func(q int) int {
		if q >= first {
			return s.l.kvb + (q-first)*c.HeadDim
		}
		return s.l.window[layer] + (q%win)*c.HeadDim
	}
	var rows []int
	if prefillN > 0 {
		start := max(pos-win+1, 0)
		for i := range min(prefillN, win) {
			if q := start + i; q <= pos {
				rows = append(rows, row(q))
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
			// The latest position at most pos in that slot.
			rows = append(rows, row(pos-((pos%win-slot+win)%win)))
		}
	}
	return rows
}

// indexer records State.indexer for batch position i (position pos); n is
// the source's entries before it.
func (s *GPUStage) indexer(b *gpu.Batch, g *gpuBlock, i, pos, prefillN int, latent bool, n, csGroup, cs int) {
	c := s.p.m.Cfg
	l := &s.l
	a := g.b.attn
	id := g.b.id
	pp := &s.pp[i]
	idim, rd, ih, v := c.IndexDim, c.RopeDim, c.IndexHeads, l.vecs
	if latent && a.kvSource {
		k := l.idx[id] + n*idim
		bf16Lin(b, g.idxWK, a.idxWK, k, l.latent)
		b.DSNorm(k, k, v+g.idxKNorm, idim, 1, 0, c.NormEps)
		b.RopeDS(k+idim-rd, rd/2, 1, 0, csGroup, false)
		b.FP4QuantInPlace(k, idim, false)
		pp.newIdx = true
	}
	iq := l.iq + i*ih*idim
	b.RopeDS(iq+idim-rd, rd/2, ih, idim, cs, false)
	b.FP4QuantInPlace(iq, ih*idim, false)
	bf16Lin(b, g.idxW, a.idxWeights, l.iw, l.x+i*c.Dim)
	b.RoundScale(l.iw, ih, float32(1/math.Sqrt(float64(idim))/math.Sqrt(float64(ih))))

	visible := (pos + 1) / a.ratio
	width := visible
	if prefillN > 0 {
		width = prefillN / a.ratio
	}
	keys := l.idx[s.source]
	cand := l.cand + i*s.positions
	switch {
	case id == c.CandidateSource:
		b.IndexScores(width, visible, iq, keys, l.iw, ih, idim, l.scores, 0, 0, false)
		if width > 0 {
			b.CandidatePool(width, visible, c.CandidateBlocks, c.CandidateBlockSize, l.scores, l.bs, l.sel, cand)
		}
		pp.candSet, pp.candLen = true, width
	case c.CandidateSource >= 0 && c.CandidateSource < id && pp.candSet:
		b.IndexScores(width, visible, iq, keys, l.iw, ih, idim, l.scores, cand, pp.candLen, true)
	default:
		b.IndexScores(width, visible, iq, keys, l.iw, ih, idim, l.scores, 0, 0, false)
	}
	k := min(c.IndexTopK, width)
	if k > 0 {
		b.SelectSorted(width, k, l.scores, l.topk+i*max(c.IndexTopK, 1), l.flags, visible)
	}
	pp.topkSet, pp.topkN = true, k
}

// moe records State.moe for the batch's inputs at x into y. With host
// experts the batch is submitted once the experts are chosen: the CPU
// computes them as State.expert does, and a new batch takes their outputs.
func (s *GPUStage) moe(b *gpu.Batch, g *gpuBlock, B int) (*gpu.Batch, error) {
	c := s.p.m.Cfg
	l := &s.l
	f := g.b.ffn
	k, E, inter, dim := f.active, len(f.experts), c.MoEInter, c.Dim
	for i := range B {
		b.MatVecs([]*gpu.Matrix{g.gate}, []int{l.z}, l.x+i*dim)
		b.MoEScores(E, l.z, l.vecs+g.gateBias, f.scoreFn == "sigmoid", l.score, l.biased)
		b.TopK(E, k, l.biased, l.esel)
		b.MoEWeights(k, l.score, l.esel, f.normTopK, f.scale, l.order+i*k, l.ew+i*k)
	}
	// The shared expert.
	b.MatVecsBatch(g.s1, l.sg, inter, l.x, dim, B, l.qx, l.xs)
	b.MatVecsBatch(g.s3, l.su, inter, l.x, dim, B, l.qx, l.xs)
	b.SiluMulDS(inter, l.sg, l.su, f.limit, false, 0, B, inter)
	b.MatVecsBatch(g.s2, l.sout, dim, l.sg, inter, B, l.qx, l.xs)
	if s.p.hostExperts {
		x, order, w := make([]float32, B*dim), make([]float32, B*k), make([]float32, B*k)
		b.Read(l.x, x)
		b.Read(l.order, order)
		b.Read(l.ew, w)
		if err := b.Submit(); err != nil {
			return nil, err
		}
		outs := make([]float32, 0, B*k*dim)
		for i := range B * k {
			e := int(math.Float32bits(order[i]))
			if e >= E {
				return nil, fmt.Errorf("dsv41: expert %d of %d", e, E)
			}
			outs = append(outs, runExpert(&f.experts[e], x[i/k*dim:(i/k+1)*dim], w[i], true, f.limit, &s.act)...)
		}
		b = s.a.Begin()
		b.Write(l.eouts, outs)
	} else {
		b.ExpertMatVecs(g.w1, inter, l.order, B*k, l.x, dim, k, l.eg, inter, l.eqx, l.exs)
		b.ExpertMatVecs(g.w3, inter, l.order, B*k, l.x, dim, k, l.eu, inter, l.eqx, l.exs)
		b.SiluMulDS(inter, l.eg, l.eu, f.limit, true, l.ew, B*k, inter)
		b.ExpertMatVecs(g.w2, dim, l.order, B*k, l.eg, inter, 1, l.eouts, dim, l.eqx, l.exs)
	}
	for i := range B {
		b.MoESum(dim, l.eouts+i*k*dim, k, dim, l.sout+i*dim, l.y+i*dim)
	}
	return b, nil
}

// rewind records taking the stage back to before position pos (State.
// Rewind): each later position's replaced rows put back, newest first,
// and the caches' lengths as they were.
func (s *GPUStage) rewind(b *gpu.Batch, pos int) {
	c, l, m := s.p.m.Cfg, &s.l, s.p.m
	hd := c.HeadDim
	for q := s.pos - 1; q >= pos; q-- {
		for id := m.from; id < m.to; id++ {
			a := m.layers[id].attn
			u := l.undo[id] + (q%Rewindable)*3*hd
			b.Copy(u, l.window[id]+(q%c.Window)*hd, hd)
			if a.kvSource && a.ratio > 1 {
				b.Copy(u+hd, l.kvSlots[id]+(q%a.ratio)*hd, hd)
				b.Copy(u+2*hd, l.scSlots[id]+(q%a.ratio)*hd, hd)
			}
		}
	}
	n := s.pos - pos
	s.compN = slices.Clone(s.hist[len(s.hist)-n])
	s.hist = s.hist[:len(s.hist)-n]
	s.pos = pos
}
