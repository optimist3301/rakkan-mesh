package llm

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"

	"github.com/optimist3301/rakkan-mesh/internal/blob"
	"github.com/optimist3301/rakkan-mesh/internal/num"
)

// Hugging Face models: a config.json, weights in safetensors files (one,
// or shards), and a tokenizer.json. Supported architectures are the LLaMA
// family's: LlamaForCausalLM (Llama 2 and 3 without rope scaling, TinyLlama,
// SmolLM), MistralForCausalLM, Qwen2ForCausalLM (Qwen2, Qwen2.5) and
// Qwen3ForCausalLM (dense Qwen3).

// HFSource is a Hugging Face model's files.
type HFSource struct {
	Config []byte // config.json
	// GenerationConfig (generation_config.json) and TokenizerConfig
	// (tokenizer_config.json) are optional: stop tokens and the chat
	// format come from them.
	GenerationConfig []byte
	TokenizerConfig  []byte
	Tokenizer        []byte // tokenizer.json
	// Weights are the safetensors files, all shards in any order.
	Weights []SizedReaderAt
}

// SizedReaderAt is a file read in pieces, such as a stored blob.
type SizedReaderAt interface {
	io.ReaderAt
	Size() int64
}

type hfConfig struct {
	Architectures         []string        `json:"architectures"`
	ModelType             string          `json:"model_type"`
	HiddenSize            int             `json:"hidden_size"`
	IntermediateSize      int             `json:"intermediate_size"`
	NumHiddenLayers       int             `json:"num_hidden_layers"`
	NumAttentionHeads     int             `json:"num_attention_heads"`
	NumKeyValueHeads      int             `json:"num_key_value_heads"`
	HeadDim               int             `json:"head_dim"`
	VocabSize             int             `json:"vocab_size"`
	MaxPositionEmbeddings int             `json:"max_position_embeddings"`
	RmsNormEps            float64         `json:"rms_norm_eps"`
	RopeTheta             float64         `json:"rope_theta"`
	RopeScaling           json.RawMessage `json:"rope_scaling"`
	TieWordEmbeddings     bool            `json:"tie_word_embeddings"`
	HiddenAct             string          `json:"hidden_act"`
	SlidingWindow         *int            `json:"sliding_window"`
	UseSlidingWindow      *bool           `json:"use_sliding_window"`
	BosTokenID            *int            `json:"bos_token_id"`
	EosTokenID            json.RawMessage `json:"eos_token_id"`
}

// safetensors: an 8-byte little-endian header size, a JSON header naming
// each tensor's type, shape and byte range, then the data.
type stTensor struct {
	DType   string   `json:"dtype"`
	Shape   []int    `json:"shape"`
	Offsets [2]int64 `json:"data_offsets"`

	file SizedReaderAt
	base int64
}

func readSafetensors(f SizedReaderAt, into map[string]*stTensor) error {
	var hdr [8]byte
	if _, err := f.ReadAt(hdr[:], 0); err != nil {
		return fmt.Errorf("safetensors: %w", err)
	}
	n := binary.LittleEndian.Uint64(hdr[:])
	if n > 100<<20 || int64(n)+8 > f.Size() {
		return fmt.Errorf("safetensors: header of %d bytes", n)
	}
	raw := make([]byte, n)
	if _, err := f.ReadAt(raw, 8); err != nil {
		return fmt.Errorf("safetensors: %w", err)
	}
	var h map[string]json.RawMessage
	if err := json.Unmarshal(raw, &h); err != nil {
		return fmt.Errorf("safetensors header: %w", err)
	}
	base := 8 + int64(n)
	for name, v := range h {
		if name == "__metadata__" {
			continue
		}
		t := &stTensor{file: f, base: base}
		if err := json.Unmarshal(v, t); err != nil {
			return fmt.Errorf("safetensors tensor %s: %w", name, err)
		}
		if t.Offsets[0] < 0 || t.Offsets[1] < t.Offsets[0] || base+t.Offsets[1] > f.Size() {
			return fmt.Errorf("safetensors tensor %s: data outside the file", name)
		}
		if _, dup := into[name]; dup {
			return fmt.Errorf("safetensors: tensor %s in two files", name)
		}
		into[name] = t
	}
	return nil
}

func (t *stTensor) elems() int {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// raw reads a tensor's bytes, checking its type and element count.
func (t *stTensor) raw(name string, want int) ([]byte, int, error) {
	size := 0
	switch t.DType {
	case "BF16", "F16":
		size = 2
	case "F32":
		size = 4
	default:
		return nil, 0, fmt.Errorf("tensor %s: type %s not supported", name, t.DType)
	}
	if t.elems() != want || int64(want*size) != t.Offsets[1]-t.Offsets[0] {
		return nil, 0, fmt.Errorf("tensor %s: shape %v, want %d elements", name, t.Shape, want)
	}
	b := make([]byte, want*size)
	if _, err := t.file.ReadAt(b, t.base+t.Offsets[0]); err != nil {
		return nil, 0, fmt.Errorf("tensor %s: %w", name, err)
	}
	return b, size, nil
}

func f16ToF32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := int(h>>10) & 0x1f
	frac := uint32(h & 0x3ff)
	switch {
	case exp == 0x1f: // inf, NaN
		return math.Float32frombits(sign | 0x7f800000 | frac<<13)
	case exp == 0 && frac == 0:
		return math.Float32frombits(sign)
	case exp == 0: // subnormal: normalize
		e := -14
		for frac&0x400 == 0 {
			frac <<= 1
			e--
		}
		return math.Float32frombits(sign | uint32(e+127)<<23 | (frac&0x3ff)<<13)
	}
	return math.Float32frombits(sign | uint32(exp-15+127)<<23 | frac<<13)
}

// hfModel reads tensors by name in the target config's types.
type hfModel struct {
	c       Config
	tensors map[string]*stTensor
}

func (m *hfModel) has(name string) bool { _, ok := m.tensors[name]; return ok }

func (m *hfModel) vec(name string, n int) ([]float32, error) {
	t, ok := m.tensors[name]
	if !ok {
		return nil, fmt.Errorf("tensor %s missing", name)
	}
	b, size, err := t.raw(name, n)
	if err != nil {
		return nil, err
	}
	out := make([]float32, n)
	for i := range out {
		switch {
		case size == 4:
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		case t.DType == "BF16":
			out[i] = num.BF16(binary.LittleEndian.Uint16(b[2*i:]))
		default:
			out[i] = f16ToF32(binary.LittleEndian.Uint16(b[2*i:]))
		}
	}
	return out, nil
}

func (m *hfModel) mat(name string, n int) (Matrix, error) {
	if m.c.DType == "bf16" {
		t, ok := m.tensors[name]
		if !ok {
			return Matrix{}, fmt.Errorf("tensor %s missing", name)
		}
		if t.DType == "BF16" {
			b, _, err := t.raw(name, n)
			if err != nil {
				return Matrix{}, err
			}
			v := make([]uint16, n)
			for i := range v {
				v[i] = binary.LittleEndian.Uint16(b[2*i:])
			}
			return Matrix{BF16: v}, nil
		}
	}
	v, err := m.vec(name, n)
	if err != nil {
		return Matrix{}, err
	}
	if m.c.DType == "bf16" {
		// A model mixing types: round the rest to bfloat16.
		w := make([]uint16, n)
		for i, f := range v {
			w[i] = num.ToBF16(f)
		}
		return Matrix{BF16: w}, nil
	}
	return Matrix{F32: v}, nil
}

// permuteRows reorders each head's rows from Hugging Face's rotary layout,
// which rotates dimension j with j+head/2, to the engine's, which rotates
// adjacent pairs: row 2j takes row j, row 2j+1 takes row j+head/2. Applied
// to the query and key projections, their biases and their head norms, it
// leaves attention unchanged.
func permuteRows[T any](w []T, rows, head int) {
	n := len(w) / rows
	tmp := make([]T, head*n)
	for h := 0; h < rows; h += head {
		blk := w[h*n : (h+head)*n]
		copy(tmp, blk)
		for j := range head / 2 {
			copy(blk[2*j*n:], tmp[j*n:(j+1)*n])
			copy(blk[(2*j+1)*n:], tmp[(j+head/2)*n:(j+head/2+1)*n])
		}
	}
}

func permuteMatrix(m *Matrix, rows, head int) {
	if m.BF16 != nil {
		permuteRows(m.BF16, rows, head)
	} else {
		permuteRows(m.F32, rows, head)
	}
}

func permuteVec(v []float32, head int) { permuteRows(v, len(v), head) }

// hfConfigOf maps config.json to a Config, refusing what the engine would
// run differently from the model's definition.
func hfConfigOf(hc *hfConfig, weights map[string]*stTensor) (Config, error) {
	switch hc.ModelType {
	case "llama", "mistral", "qwen2", "qwen3":
	default:
		return Config{}, fmt.Errorf("model type %q is not supported (llama, mistral, qwen2, qwen3)", hc.ModelType)
	}
	if hc.HiddenAct != "" && hc.HiddenAct != "silu" {
		return Config{}, fmt.Errorf("activation %q is not supported", hc.HiddenAct)
	}
	if s := strings.TrimSpace(string(hc.RopeScaling)); s != "" && s != "null" {
		return Config{}, fmt.Errorf("rope scaling %s is not supported", s)
	}
	c := Config{
		Dim: hc.HiddenSize, Hidden: hc.IntermediateSize, Layers: hc.NumHiddenLayers,
		Heads: hc.NumAttentionHeads, KVHeads: hc.NumKeyValueHeads, Vocab: hc.VocabSize,
		SeqLen: hc.MaxPositionEmbeddings, RopeTheta: hc.RopeTheta, NormEps: hc.RmsNormEps,
		SharedClassifier: hc.TieWordEmbeddings,
	}
	if c.KVHeads == 0 {
		c.KVHeads = c.Heads
	}
	if hc.HeadDim > 0 && (c.Heads == 0 || hc.HeadDim != c.Dim/c.Heads) {
		c.HeadDim = hc.HeadDim
	}
	if c.RopeTheta == 0 {
		c.RopeTheta = 10000
	}
	if c.NormEps == 0 {
		c.NormEps = 1e-6
	}
	// Within a sliding window, attention is ordinary attention: cap the
	// context to it rather than mask.
	if w := hc.SlidingWindow; w != nil && *w > 0 && *w < c.SeqLen && (hc.UseSlidingWindow == nil || *hc.UseSlidingWindow) {
		c.SeqLen = *w
	}
	if _, ok := weights["lm_head.weight"]; !ok {
		c.SharedClassifier = true
	}
	_, c.QKVBias = weights["model.layers.0.self_attn.q_proj.bias"]
	_, c.QKNorm = weights["model.layers.0.self_attn.q_norm.weight"]
	if t, ok := weights["model.embed_tokens.weight"]; ok && t.DType == "BF16" {
		c.DType = "bf16"
	} else {
		c.DType = "f32"
	}
	if _, ok := weights["model.layers.0.self_attn.o_proj.bias"]; ok {
		return Config{}, errors.New("output projection biases are not supported")
	}
	return c, c.validate()
}

// ImportHF stores a Hugging Face model's parts and manifest, reading the
// weights one tensor at a time, and returns the model's ID.
func ImportHF(store *blob.Store, src HFSource, name string) (blob.ID, *Manifest, error) {
	var hc hfConfig
	if err := json.Unmarshal(src.Config, &hc); err != nil {
		return blob.ID{}, nil, fmt.Errorf("config.json: %w", err)
	}
	tensors := map[string]*stTensor{}
	for _, f := range src.Weights {
		if err := readSafetensors(f, tensors); err != nil {
			return blob.ID{}, nil, err
		}
	}
	c, err := hfConfigOf(&hc, tensors)
	if err != nil {
		return blob.ID{}, nil, err
	}
	if _, err := ReadBPE(src.Tokenizer, c.Vocab); err != nil {
		return blob.ID{}, nil, err
	}
	hm := &hfModel{c: c, tensors: tensors}
	m := &Manifest{Version: 1, Name: name, Config: c, TokenizerFormat: "hf"}
	m.Stop, m.BOS, m.Chat = hfTokens(&hc, src)

	put := func(b []byte) (string, error) {
		id, err := store.PutBytes(b)
		return id.String(), err
	}
	embed, err := hm.mat("model.embed_tokens.weight", c.Vocab*c.Dim)
	if err != nil {
		return blob.ID{}, nil, err
	}
	if m.Embed, err = put(appendMatrix(nil, embed)); err != nil {
		return blob.ID{}, nil, err
	}
	head := &Head{Wcls: embed}
	if head.RmsFinal, err = hm.vec("model.norm.weight", c.Dim); err != nil {
		return blob.ID{}, nil, err
	}
	if !c.SharedClassifier {
		if head.Wcls, err = hm.mat("lm_head.weight", c.Vocab*c.Dim); err != nil {
			return blob.ID{}, nil, err
		}
	}
	if m.Head, err = put(EncodeHead(c, head)); err != nil {
		return blob.ID{}, nil, err
	}
	embed, head = Matrix{}, nil // let them go before the layers

	for i := range c.Layers {
		l, err := hm.layer(i)
		if err != nil {
			return blob.ID{}, nil, err
		}
		id, err := put(EncodeLayer(c, l))
		if err != nil {
			return blob.ID{}, nil, err
		}
		m.Layers = append(m.Layers, id)
	}
	if m.Tokenizer, err = put(src.Tokenizer); err != nil {
		return blob.ID{}, nil, err
	}
	data, _ := json.Marshal(m)
	id, err := store.PutBytes(data)
	return id, m, err
}

func (hm *hfModel) layer(i int) (*Layer, error) {
	c := hm.c
	p := fmt.Sprintf("model.layers.%d.", i)
	d, q, kv, h, hs := c.Dim, c.QDim(), c.KVDim(), c.Hidden, c.HeadSize()
	l := &Layer{}
	var err error
	vec := func(dst *[]float32, name string, n int) {
		if err == nil {
			*dst, err = hm.vec(p+name, n)
		}
	}
	mat := func(dst *Matrix, name string, n int) {
		if err == nil {
			*dst, err = hm.mat(p+name, n)
		}
	}
	vec(&l.RmsAtt, "input_layernorm.weight", d)
	mat(&l.Wq, "self_attn.q_proj.weight", q*d)
	mat(&l.Wk, "self_attn.k_proj.weight", kv*d)
	mat(&l.Wv, "self_attn.v_proj.weight", kv*d)
	mat(&l.Wo, "self_attn.o_proj.weight", d*q)
	vec(&l.RmsFfn, "post_attention_layernorm.weight", d)
	mat(&l.W1, "mlp.gate_proj.weight", h*d)
	mat(&l.W2, "mlp.down_proj.weight", d*h)
	mat(&l.W3, "mlp.up_proj.weight", h*d)
	if c.QKVBias {
		vec(&l.Bq, "self_attn.q_proj.bias", q)
		vec(&l.Bk, "self_attn.k_proj.bias", kv)
		vec(&l.Bv, "self_attn.v_proj.bias", kv)
	}
	if c.QKNorm {
		vec(&l.QNorm, "self_attn.q_norm.weight", hs)
		vec(&l.KNorm, "self_attn.k_norm.weight", hs)
	}
	if err != nil {
		return nil, err
	}
	permuteMatrix(&l.Wq, q, hs)
	permuteMatrix(&l.Wk, kv, hs)
	if c.QKVBias {
		permuteVec(l.Bq, hs)
		permuteVec(l.Bk, hs)
	}
	if c.QKNorm {
		permuteVec(l.QNorm, hs)
		permuteVec(l.KNorm, hs)
	}
	return l, nil
}

// hfTokens finds stop tokens, the BOS token and the chat format.
func hfTokens(hc *hfConfig, src HFSource) (stop []int, bos *int, chat *ChatTemplate) {
	ids := func(raw json.RawMessage) []int {
		var one int
		var many []int
		if json.Unmarshal(raw, &one) == nil {
			return []int{one}
		}
		json.Unmarshal(raw, &many)
		return many
	}
	var gen struct {
		Eos json.RawMessage `json:"eos_token_id"`
	}
	if len(src.GenerationConfig) > 0 && json.Unmarshal(src.GenerationConfig, &gen) == nil {
		stop = ids(gen.Eos)
	}
	if len(stop) == 0 {
		stop = ids(hc.EosTokenID)
	}
	// Qwen prepends nothing; the LLaMA family prepends BOS.
	if (hc.ModelType == "llama" || hc.ModelType == "mistral") && hc.BosTokenID != nil {
		bos = hc.BosTokenID
	}
	var tc struct {
		ChatTemplate string `json:"chat_template"`
	}
	if len(src.TokenizerConfig) > 0 {
		json.Unmarshal(src.TokenizerConfig, &tc)
	}
	t := tc.ChatTemplate
	switch {
	case strings.Contains(t, "<|im_start|>") && strings.Contains(t, "enable_thinking"):
		// Qwen3, with thinking off: an empty thought.
		chat = &ChatTemplate{Prefix: "<|im_start|>user\n", Suffix: "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"}
	case strings.Contains(t, "<|im_start|>"):
		chat = &ChatTemplate{Prefix: "<|im_start|>user\n", Suffix: "<|im_end|>\n<|im_start|>assistant\n"}
		if strings.Contains(t, "You are Qwen, created by Alibaba Cloud") {
			chat.Prefix = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n" + chat.Prefix
		}
	case strings.Contains(t, "<|start_header_id|>"):
		chat = &ChatTemplate{Prefix: "<|start_header_id|>user<|end_header_id|>\n\n", Suffix: "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"}
	}
	return stop, bos, chat
}
