package dsv41

import (
	"bytes"
	"fmt"
	"io"
	"path/filepath"
	"slices"
	"sort"
	"strings"
)

// Model is a loaded DeepSeek-V4.1 (the backbone; DSpark and vision are
// not loaded yet), or a part of one: the blocks of a pipeline stage
// (NewPart), without the head.
type Model struct {
	Cfg      *Config
	head     *Head
	layers   []*block // by layer; nil outside [from, to)
	from, to int
	ropes    [2]*rope // without and with compression
}

// Head is what a pipeline's coordinator runs itself: the embedding, the
// Engram hashes over the token history, the final norm and the classifier.
type Head struct {
	Cfg    *Config
	embed  []uint16 // [vocab, dim] bf16
	norm   []float32
	head   *linear // bf16 widened to float32, float32 logits
	engram *EngramTables
}

// EngramTable is the n-gram memory of an Engram layer, which may stay on
// disk: FP8 rows [rows, engram dim] and their E8M0 scales [rows, dim/32].
// A step reads 24 rows of it. Both must allow concurrent reads.
type EngramTable struct {
	Rows, Scales io.ReaderAt
}

type block struct {
	id                int
	attnNorm, ffnNorm []float32
	hcAttn, hcFFN     hcParams
	attn              *attention
	ffn               *moe
	engram            *engram
}

type hcParams struct {
	fn          *linear   // [(2+hc)·hc, hc·dim] float32
	base, scale []float32 // [(2+hc)·hc], [3]
}

type attention struct {
	ratio              int
	kvSource, idxSrc   bool
	wqA, wqB, wkv      *linear
	qNorm, kvNorm      []float32
	sink               []float32
	woA, woB           *linear
	compWKV, compGate  *linear // compressor (KV sources)
	compNorm           []float32
	idxWqB, idxWeights *linear // indexer (index sources)
	idxWK              *linear // indexer keys (KV sources)
	idxKNorm           []float32
	rope               *rope
}

type moe struct {
	gate     *linear // bf16 widened to float32
	bias     []float32
	experts  []expert
	shared   expert
	limit    float32
	scoreFn  string
	normTopK bool
	scale    float32
	active   int
}

type expert struct{ w1, w2, w3 *linear }

type engram struct {
	index int // which Engram layer, for its hash tables
	table EngramTable
	dim   int
	wkv   *linear
	qk    []float32 // q_weight·k_weight, [hc, model dim]
}

// Load reads a model from a directory with config.json, the safetensors
// shards (all read into memory) and the Engram hash tables.
func Load(dir string, tables *EngramTables) (*Model, error) {
	cfg, err := LoadConfig(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	files, err := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	if err != nil {
		return nil, err
	}
	all := map[string]*Tensor{}
	for _, f := range files {
		if filepath.Base(f) == "golden.safetensors" {
			continue
		}
		ts, err := ReadSafetensors(f)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", f, err)
		}
		for k, v := range ts {
			all[k] = v
		}
	}
	return New(cfg, all, tables)
}

// New builds a model from its tensors, named as in the published
// checkpoint.
func New(cfg *Config, tensors map[string]*Tensor, tables *EngramTables) (*Model, error) {
	w := &weights{t: tensors, used: map[string]bool{}}
	h, err := newHead(w, cfg, tables)
	if err != nil {
		return nil, err
	}
	m, err := newPart(w, cfg, 0, cfg.Layers, nil)
	if err != nil {
		return nil, err
	}
	m.head = h
	return m, w.unused()
}

// NewPart builds blocks [from, to) of a model, for a pipeline stage, from
// their tensors. The Engram tables of its Engram layers are read through
// engram (by layer) when given, else they must be among the tensors.
func NewPart(cfg *Config, tensors map[string]*Tensor, from, to int, engram map[int]EngramTable) (*Model, error) {
	if from < 0 || to > cfg.Layers || from >= to {
		return nil, fmt.Errorf("dsv41: layers %d..%d outside the model's %d", from, to, cfg.Layers)
	}
	w := &weights{t: tensors, used: map[string]bool{}, engram: engram}
	m, err := newPart(w, cfg, from, to, engram)
	if err != nil {
		return nil, err
	}
	return m, w.unused()
}

// NewHead builds a coordinator's part of a model from its tensors
// (embed.weight, norm.weight, head.weight). tables are the Engram hash
// tables (BuildEngramTables), needed when the model has Engram layers.
func NewHead(cfg *Config, tensors map[string]*Tensor, tables *EngramTables) (*Head, error) {
	w := &weights{t: tensors, used: map[string]bool{}}
	h, err := newHead(w, cfg, tables)
	if err != nil {
		return nil, err
	}
	return h, w.unused()
}

func newHead(w *weights, c *Config, tables *EngramTables) (*Head, error) {
	h := &Head{Cfg: c, engram: tables}
	if len(c.EngramLayers) > 0 {
		if err := tables.check(c); err != nil {
			return nil, err
		}
	}
	t, err := w.get("embed.weight")
	if err != nil {
		return nil, err
	}
	if t.DType != "BF16" || !shapeIs(t, c.Vocab, c.Dim) {
		return nil, fmt.Errorf("dsv41: embed.weight %s %v", t.DType, t.Shape)
	}
	h.embed = toBF16(t)
	if h.norm, err = w.vec("norm.weight", c.Dim); err != nil {
		return nil, err
	}
	if h.head, err = w.lin("head", c.Vocab, c.Dim, kindBF16F32); err != nil {
		return nil, err
	}
	return h, nil
}

func newPart(w *weights, c *Config, from, to int, engram map[int]EngramTable) (*Model, error) {
	m := &Model{Cfg: c, from: from, to: to, ropes: [2]*rope{newRope(c, false), newRope(c, true)}}
	m.layers = make([]*block, c.Layers)
	for i := from; i < to; i++ {
		b, err := m.loadBlock(w, i)
		if err != nil {
			return nil, err
		}
		m.layers[i] = b
	}
	return m, nil
}

// unused reports tensors given and not used, but for DSpark's, vision's
// and the image routing biases.
func (w *weights) unused() error {
	var unused []string
	for name := range w.t {
		if !w.used[name] && !strings.HasPrefix(name, "mtp.") && !strings.HasPrefix(name, "vision.") &&
			!strings.HasPrefix(name, "aligner.") && !strings.HasPrefix(name, "image_") && !strings.HasSuffix(name, ".bias_vl") {
			unused = append(unused, name)
		}
	}
	if len(unused) > 0 {
		sort.Strings(unused)
		return fmt.Errorf("dsv41: unexpected tensors: %v", unused)
	}
	return nil
}

// Head is the coordinator's part of a whole model.
func (m *Model) Head() *Head { return m.head }

func (m *Model) loadBlock(w *weights, i int) (*block, error) {
	c := m.Cfg
	p := fmt.Sprintf("layers.%d.", i)
	b := &block{id: i}
	var err error
	vec := func(dst *[]float32, name string, n int) {
		if err == nil {
			*dst, err = w.vec(p+name, n)
		}
	}
	lin := func(dst **linear, name string, rows, cols int, kind linearKind) {
		if err == nil {
			*dst, err = w.lin(p+name, rows, cols, kind)
		}
	}
	vec(&b.attnNorm, "attn_norm.weight", c.Dim)
	vec(&b.ffnNorm, "ffn_norm.weight", c.Dim)
	mix := (2 + c.HCMult) * c.HCMult
	for _, h := range []struct {
		dst  *hcParams
		name string
	}{{&b.hcAttn, "hc_attn_"}, {&b.hcFFN, "hc_ffn_"}} {
		vec(&h.dst.base, h.name+"base", mix)
		vec(&h.dst.scale, h.name+"scale", 3)
		if err == nil {
			var fn *Tensor
			if fn, err = w.get(p + h.name + "fn"); err == nil {
				if fn.DType != "F32" || !shapeIs(fn, mix, c.HCMult*c.Dim) {
					err = fmt.Errorf("dsv41: %s%sfn %s %v", p, h.name, fn.DType, fn.Shape)
				} else {
					f, _ := toF32(p+h.name+"fn", fn)
					h.dst.fn = &linear{kind: kindF32, rows: mix, cols: c.HCMult * c.Dim, f: f}
				}
			}
		}
	}

	a := &attention{ratio: c.CompressRatios[i], kvSource: c.isKVSource(i), idxSrc: c.isIndexSource(i)}
	a.rope = m.ropes[min(a.ratio, 1)]
	lin(&a.wqA, "attn.wq_a", c.QLoRA, c.Dim, kindFP8)
	vec(&a.qNorm, "attn.q_norm.weight", c.QLoRA)
	lin(&a.wqB, "attn.wq_b", c.Heads*c.HeadDim, c.QLoRA, kindFP8)
	lin(&a.wkv, "attn.wkv", c.HeadDim, c.Dim, kindFP8)
	vec(&a.kvNorm, "attn.kv_norm.weight", c.HeadDim)
	vec(&a.sink, "attn.attn_sink", c.Heads)
	lin(&a.woA, "attn.wo_a", c.OGroups*c.OLoRA, c.Heads*c.HeadDim/c.OGroups, kindFP8)
	lin(&a.woB, "attn.wo_b", c.Dim, c.OGroups*c.OLoRA, kindFP8)
	if err == nil && (a.wqA.kind != kindFP8 || a.woA.kind != kindFP8 || a.woB.kind != kindFP8) {
		err = fmt.Errorf("dsv41: layer %d: attention projections must be FP8", i)
	}
	if a.kvSource {
		if a.ratio == 0 {
			return nil, fmt.Errorf("dsv41: layer %d is a KV source without compression", i)
		}
		if a.ratio == 1 {
			lin(&a.compWKV, "attn.compressor.wkv", c.HeadDim, c.Dim, kindBF16)
		} else {
			// Softmax pooling runs in float32: the weights are widened.
			lin(&a.compWKV, "attn.compressor.wkv", c.HeadDim, c.Dim, kindBF16F32)
			lin(&a.compGate, "attn.compressor.wgate", c.HeadDim, c.Dim, kindBF16F32)
		}
		vec(&a.compNorm, "attn.compressor.norm.weight", c.HeadDim)
		lin(&a.idxWK, "attn.indexer.wk", c.IndexDim, c.HeadDim, kindBF16)
		vec(&a.idxKNorm, "attn.indexer.k_norm.weight", c.IndexDim)
	}
	if a.idxSrc {
		lin(&a.idxWqB, "attn.indexer.wq_b", c.IndexHeads*c.IndexDim, c.QLoRA, kindFP8)
		lin(&a.idxWeights, "attn.indexer.weights_proj", c.IndexHeads, c.Dim, kindBF16)
	}
	b.attn = a

	f := &moe{limit: c.SwiGLULimit, scoreFn: c.ScoreFunc, normTopK: c.NormTopK, scale: c.RouteScale, active: c.ActiveExperts}
	lin(&f.gate, "ffn.gate", c.RoutedExperts, c.Dim, kindBF16F32)
	vec(&f.bias, "ffn.gate.bias", c.RoutedExperts)
	for e := range c.RoutedExperts {
		var x expert
		q := fmt.Sprintf("ffn.experts.%d.", e)
		lin(&x.w1, q+"w1", c.MoEInter, c.Dim, kindFP4)
		lin(&x.w2, q+"w2", c.Dim, c.MoEInter, kindFP4)
		lin(&x.w3, q+"w3", c.MoEInter, c.Dim, kindFP4)
		f.experts = append(f.experts, x)
	}
	lin(&f.shared.w1, "ffn.shared_experts.w1", c.MoEInter, c.Dim, kindFP8)
	lin(&f.shared.w2, "ffn.shared_experts.w2", c.Dim, c.MoEInter, kindFP8)
	lin(&f.shared.w3, "ffn.shared_experts.w3", c.MoEInter, c.Dim, kindFP8)
	b.ffn = f

	if k := slices.Index(c.EngramLayers, i); k >= 0 && err == nil {
		e := &engram{index: k, dim: c.EngramDim}
		if tab, ok := w.engram[i]; ok {
			e.table = tab
		} else {
			var emb, sc *Tensor
			if emb, err = w.get(p + "engram.embed.weight"); err == nil {
				sc, err = w.get(p + "engram.embed.scale")
			}
			if err == nil && (emb.DType != "F8_E4M3" || !shapeIs(emb, c.EngramRows[k], c.EngramDim) ||
				sc.DType != "F8_E8M0" || !shapeIs(sc, c.EngramRows[k], c.EngramDim/32)) {
				err = fmt.Errorf("dsv41: layer %d: engram table %s %v, scales %s %v", i, emb.DType, emb.Shape, sc.DType, sc.Shape)
			}
			if err == nil {
				e.table = EngramTable{Rows: bytes.NewReader(emb.Data), Scales: bytes.NewReader(sc.Data)}
			}
		}
		cols := (c.EngramMaxNgram - 1) * c.EngramHeads
		lin(&e.wkv, "engram.wkv", c.Dim*(c.HCMult+1), cols*c.EngramDim, kindFP8)
		var qw, kw []float32
		vec(&qw, "engram.q_weight", c.HCMult*c.Dim)
		vec(&kw, "engram.k_weight", c.HCMult*c.Dim)
		if err == nil {
			e.qk = make([]float32, len(qw))
			for j := range qw {
				e.qk[j] = qw[j] * kw[j] // exact: products of bf16 values
			}
		}
		b.engram = e
	}
	if err != nil {
		return nil, err
	}
	return b, nil
}
