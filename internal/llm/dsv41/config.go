// Package dsv41 runs DeepSeek-V4.1-Flash: a mixture-of-experts transformer
// with hyper-connections, compressed sparse attention (CSA2) and Engram
// n-gram memory. docs/dev/deepseek-v41.md specifies what it computes; the
// reference is the developers' code in tools/llmref/dsv41/upstream.
//
// The reference leaves some arithmetic unspecified: the order of its
// float32 reductions (norms, softmax, float32 products) and its exp, pow
// and cos come from PyTorch. Here every such operation has one fixed order
// and portable functions (package num), so every node computes the same
// bits; against the reference the result agrees to rounding. Quantization
// and the FP8/FP4 products follow the reference's kernels bit for bit.
package dsv41

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
)

// Config is a model's shape, read from its Hugging Face config.json.
type Config struct {
	Vocab, Dim, MoEInter, Layers, Heads int
	HeadDim, RopeDim, QLoRA, OLoRA      int
	OGroups                             int
	RoutedExperts, ActiveExperts        int
	ScoreFunc                           string
	NormTopK                            bool
	RouteScale, SwiGLULimit             float32
	NormEps                             float32
	Window                              int
	// CompressRatios has one entry per layer, DSpark layers included: 0
	// is sliding window only, r compresses KV r-to-1.
	CompressRatios          []int
	KVSources, IndexSources []int
	RopeTheta               float64
	CompressRopeTheta       float64
	RopeFactor              float64
	BetaFast, BetaSlow      float64
	OriginalSeqLen          int
	IndexHeads, IndexDim    int
	IndexTopK               int
	CandidateSource         int // -1: none
	CandidateBlocks         int
	CandidateBlockSize      int
	HCMult, HCSinkhornIters int
	HCEps                   float32
	EngramLayers            []int
	EngramRows              []int
	EngramMaxNgram          int
	EngramHeads, EngramDim  int
	// EngramVocab is where each table's search for prime sizes starts,
	// EngramPad the token filling n-grams with no history, and
	// EngramCompressed the size of the compressed token map.
	EngramVocab, EngramPad, EngramCompressed int
	MTPLayers                                int
	// DSpark, the draft model (draft.go): BlockSize tokens drafted at once,
	// the first the last accepted, the rest Noise; it reads the stream's
	// mean at the input of the Targets layers. Its blocks' MoE has
	// DraftExperts experts, DraftActive of them per token.
	DraftBlock, DraftNoise, DraftRank int
	DraftTargets                      []int
	DraftExperts, DraftActive         int
	BOS, EOS                          int
	// MaxPositions is the longest context the model was trained for.
	MaxPositions int
}

type hfConfig struct {
	BOS          int `json:"bos_token_id"`
	EOS          int `json:"eos_token_id"`
	Quantization struct {
		QuantMethod     string `json:"quant_method"`
		WeightBlockSize []int  `json:"weight_block_size"`
		ScaleFmt        string `json:"scale_fmt"`
		ExpertDType     string `json:"expert_dtype"`
	} `json:"quantization_config"`
	Text struct {
		ModelType   string  `json:"model_type"`
		Vocab       int     `json:"vocab_size"`
		Hidden      int     `json:"hidden_size"`
		MoEInter    int     `json:"moe_intermediate_size"`
		Layers      int     `json:"num_hidden_layers"`
		Heads       int     `json:"num_attention_heads"`
		KVHeads     int     `json:"num_key_value_heads"`
		HeadDim     int     `json:"head_dim"`
		RopeDim     int     `json:"qk_rope_head_dim"`
		QLoRA       int     `json:"q_lora_rank"`
		OLoRA       int     `json:"o_lora_rank"`
		OGroups     int     `json:"o_groups"`
		Act         string  `json:"hidden_act"`
		SwiGLULimit float64 `json:"swiglu_limit"`
		NormEps     float64 `json:"rms_norm_eps"`
		MaxPos      int     `json:"max_position_embeddings"`
		RopeTheta   float64 `json:"rope_theta"`
		RopeScaling struct {
			Type     string  `json:"rope_type"`
			Factor   float64 `json:"factor"`
			BetaFast float64 `json:"beta_fast"`
			BetaSlow float64 `json:"beta_slow"`
			Original int     `json:"original_max_position_embeddings"`
		} `json:"rope_scaling"`
		RoutedExperts      int     `json:"n_routed_experts"`
		SharedExperts      int     `json:"n_shared_experts"`
		ActiveExperts      int     `json:"num_experts_per_tok"`
		ScoreFunc          string  `json:"scoring_func"`
		TopKMethod         string  `json:"topk_method"`
		NormTopK           bool    `json:"norm_topk_prob"`
		RouteScale         float64 `json:"routed_scaling_factor"`
		Window             int     `json:"sliding_window"`
		CompressRatios     []int   `json:"compress_ratios"`
		CompressRopeTheta  float64 `json:"compress_rope_theta"`
		KVSources          []int   `json:"kv_source_layer_ids"`
		IndexSources       []int   `json:"index_source_layer_ids"`
		IndexHeads         int     `json:"index_n_heads"`
		IndexDim           int     `json:"index_head_dim"`
		IndexTopK          int     `json:"index_topk"`
		CandidateSource    *int    `json:"candidate_source_layer_id"`
		CandidateBlocks    int     `json:"candidate_topk_blocks"`
		CandidateBlockSize int     `json:"candidate_block_size"`
		HCMult             int     `json:"hc_mult"`
		HCSinkhornIters    int     `json:"hc_sinkhorn_iters"`
		HCEps              float64 `json:"hc_eps"`
		EngramLayers       []int   `json:"engram_layer_ids"`
		EngramRows         []int   `json:"engram_num_embeddings"`
		EngramMaxNgram     int     `json:"engram_max_ngram_size"`
		EngramHeads        int     `json:"engram_n_heads"`
		EngramDim          int     `json:"engram_head_dim"`
		EngramVocab        int     `json:"engram_vocab_size"`
		EngramPad          int     `json:"engram_pad_token_id"`
		EngramCompressed   int     `json:"engram_compressed_vocab_size"`
		MTPLayers          int     `json:"num_nextn_predict_layers"`
		DSparkBlock        int     `json:"dspark_block_size"`
		DSparkNoise        int     `json:"dspark_noise_token_id"`
		DSparkTargets      []int   `json:"dspark_target_layer_ids"`
		DSparkRank         int     `json:"dspark_markov_rank"`
		DSparkExperts      int     `json:"dspark_n_routed_experts"`
		DSparkActive       int     `json:"dspark_num_experts_per_tok"`
	} `json:"text_config"`
}

// LoadConfig reads and checks a config.json.
func LoadConfig(path string) (*Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ParseConfig(b)
}

// ParseConfig parses a config.json of a deepseek_v41 model.
func ParseConfig(b []byte) (*Config, error) {
	var h hfConfig
	if err := json.Unmarshal(b, &h); err != nil {
		return nil, fmt.Errorf("dsv41 config: %w", err)
	}
	t := h.Text
	q := h.Quantization
	switch {
	case t.ModelType != "deepseek_v41_text":
		return nil, fmt.Errorf("dsv41 config: model type %q", t.ModelType)
	case q.QuantMethod != "fp8" || !slices.Equal(q.WeightBlockSize, []int{32, 32}) || q.ScaleFmt != "ue8m0" || q.ExpertDType != "fp4":
		return nil, fmt.Errorf("dsv41 config: quantization %+v not supported", q)
	case t.KVHeads != 1 || t.SharedExperts != 1 || t.Act != "silu" || t.TopKMethod != "noaux_tc":
		return nil, errors.New("dsv41 config: unsupported attention or MoE variant")
	case t.ScoreFunc != "sqrtsoftplus" && t.ScoreFunc != "sigmoid" && t.ScoreFunc != "softmax":
		return nil, fmt.Errorf("dsv41 config: scoring %q", t.ScoreFunc)
	case t.RopeScaling.Type != "yarn":
		return nil, fmt.Errorf("dsv41 config: rope scaling %q", t.RopeScaling.Type)
	case len(t.CompressRatios) != t.Layers+t.MTPLayers:
		return nil, errors.New("dsv41 config: compress_ratios does not cover every layer")
	case t.Hidden%32 != 0 || t.MoEInter%32 != 0 || t.HeadDim%32 != 0 || t.QLoRA%32 != 0 || (t.Heads*t.HeadDim/t.OGroups)%32 != 0:
		return nil, errors.New("dsv41 config: widths must be multiples of 32")
	case len(t.EngramLayers) != len(t.EngramRows):
		return nil, errors.New("dsv41 config: engram layers and tables differ in number")
	}
	c := &Config{
		Vocab: t.Vocab, Dim: t.Hidden, MoEInter: t.MoEInter, Layers: t.Layers, Heads: t.Heads,
		HeadDim: t.HeadDim, RopeDim: t.RopeDim, QLoRA: t.QLoRA, OLoRA: t.OLoRA, OGroups: t.OGroups,
		RoutedExperts: t.RoutedExperts, ActiveExperts: t.ActiveExperts, ScoreFunc: t.ScoreFunc,
		NormTopK: t.NormTopK, RouteScale: float32(t.RouteScale), SwiGLULimit: float32(t.SwiGLULimit),
		NormEps: float32(t.NormEps), Window: t.Window, CompressRatios: t.CompressRatios,
		KVSources: t.KVSources, IndexSources: t.IndexSources,
		RopeTheta: t.RopeTheta, CompressRopeTheta: t.CompressRopeTheta,
		RopeFactor: t.RopeScaling.Factor, BetaFast: t.RopeScaling.BetaFast, BetaSlow: t.RopeScaling.BetaSlow,
		OriginalSeqLen: t.RopeScaling.Original,
		MaxPositions:   t.MaxPos,
		IndexHeads:     t.IndexHeads, IndexDim: t.IndexDim, IndexTopK: t.IndexTopK,
		CandidateSource: -1, CandidateBlocks: t.CandidateBlocks, CandidateBlockSize: t.CandidateBlockSize,
		HCMult: t.HCMult, HCSinkhornIters: t.HCSinkhornIters, HCEps: float32(t.HCEps),
		EngramLayers: t.EngramLayers, EngramRows: t.EngramRows, EngramMaxNgram: t.EngramMaxNgram,
		EngramHeads: t.EngramHeads, EngramDim: t.EngramDim, MTPLayers: t.MTPLayers,
		EngramVocab: t.EngramVocab, EngramPad: t.EngramPad, EngramCompressed: t.EngramCompressed,
		BOS: h.BOS, EOS: h.EOS,
		DraftBlock: t.DSparkBlock, DraftNoise: t.DSparkNoise, DraftRank: t.DSparkRank, DraftTargets: t.DSparkTargets,
		DraftExperts: t.DSparkExperts, DraftActive: t.DSparkActive,
	}
	if c.DraftExperts == 0 {
		c.DraftExperts, c.DraftActive = c.RoutedExperts, c.ActiveExperts
	}
	if t.CandidateSource != nil {
		c.CandidateSource = *t.CandidateSource
	}
	return c, nil
}

// isKVSource reports whether layer i compresses its own KV.
func (c *Config) isKVSource(i int) bool { return i < c.Layers && slices.Contains(c.KVSources, i) }

// isIndexSource reports whether layer i runs its own indexer.
func (c *Config) isIndexSource(i int) bool { return i < c.Layers && slices.Contains(c.IndexSources, i) }
