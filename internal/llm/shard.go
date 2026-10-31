package llm

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"

	"github.com/optimist3301/rakkan-mesh/internal/blob"
)

// A model is stored as blobs: the embedding table, each layer, the head
// and the tokenizer, listed by a manifest whose blob ID is the model's ID.
// A pipeline stage fetches only its layers.

// Manifest lists a model's parts.
type Manifest struct {
	Version   int      `json:"v"`
	Name      string   `json:"name,omitempty"`
	Config    Config   `json:"config"`
	Embed     string   `json:"embed"`
	Layers    []string `json:"layers"`
	Head      string   `json:"head"`
	Tokenizer string   `json:"tokenizer"`
	// TokenizerFormat is "llama2c" (also "") or "hf", a Hugging Face
	// tokenizer.json with byte-level BPE.
	TokenizerFormat string `json:"tokenizer_format,omitempty"`
	// BOS, if set, is prepended to prompts. Unset for a llama2c model
	// means token 1.
	BOS *int `json:"bos,omitempty"`
	// Stop lists tokens that end a generation; empty for a llama2c model
	// means tokens 1 and 2.
	Stop []int `json:"stop,omitempty"`
	// Chat, for chat-tuned models, wraps a prompt in their conversation
	// format when a generation asks for it.
	Chat *ChatTemplate `json:"chat,omitempty"`
}

// ChatTemplate turns a user message into a prompt: Prefix, the message,
// Suffix. Special tokens in them, such as <|im_start|>, are encoded as such.
type ChatTemplate struct {
	Prefix string `json:"prefix"`
	Suffix string `json:"suffix"`
}

func (m *Manifest) bos() int {
	switch {
	case m.BOS != nil:
		return *m.BOS
	case m.TokenizerFormat == "" || m.TokenizerFormat == "llama2c":
		return 1
	}
	return -1
}

func (m *Manifest) stop() []int {
	if len(m.Stop) == 0 && (m.TokenizerFormat == "" || m.TokenizerFormat == "llama2c") {
		return []int{1, 2}
	}
	return m.Stop
}

// maxManifest bounds a manifest: a thousand layer IDs and change.
const maxManifest = 256 << 10

// ParseManifest checks a manifest.
func ParseManifest(data []byte) (*Manifest, error) {
	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("model manifest: %w", err)
	}
	if m.Version != 1 {
		return nil, fmt.Errorf("model manifest version %d", m.Version)
	}
	if err := m.Config.validate(); err != nil {
		return nil, fmt.Errorf("model manifest: %w", err)
	}
	if len(m.Layers) != m.Config.Layers {
		return nil, fmt.Errorf("model manifest lists %d layers, config says %d", len(m.Layers), m.Config.Layers)
	}
	if f := m.TokenizerFormat; f != "" && f != "llama2c" && f != "hf" {
		return nil, fmt.Errorf("model manifest: tokenizer format %q", f)
	}
	for _, id := range append([]string{m.Embed, m.Head, m.Tokenizer}, m.Layers...) {
		if _, err := blob.ParseID(id); err != nil {
			return nil, fmt.Errorf("model manifest: %w", err)
		}
	}
	return &m, nil
}

func f32Bytes(parts ...[]float32) []byte {
	n := 0
	for _, p := range parts {
		n += len(p)
	}
	b := make([]byte, 0, 4*n)
	for _, p := range parts {
		for _, v := range p {
			b = binary.LittleEndian.AppendUint32(b, math.Float32bits(v))
		}
	}
	return b
}

// appendMatrix appends a matrix in its own type: float32 or bfloat16,
// little-endian.
func appendMatrix(b []byte, m Matrix) []byte {
	if m.BF16 != nil {
		for _, v := range m.BF16 {
			b = binary.LittleEndian.AppendUint16(b, v)
		}
		return b
	}
	return append(b, f32Bytes(m.F32)...)
}

// decoder takes tensors from a part's bytes in order.
type decoder struct {
	c    Config
	data []byte
	err  error
}

func (d *decoder) take(n int) []byte {
	if d.err != nil || n < 0 || n > len(d.data) {
		d.err = errors.New("model part has the wrong size")
		return nil
	}
	b := d.data[:n]
	d.data = d.data[n:]
	return b
}

func (d *decoder) vec(n int) []float32 { return floats(d.take(4 * n)) }

func (d *decoder) mat(n int) Matrix {
	if d.c.DType == "bf16" {
		b := d.take(2 * n)
		if b == nil {
			return Matrix{}
		}
		v := make([]uint16, n)
		for i := range v {
			v[i] = binary.LittleEndian.Uint16(b[2*i:])
		}
		return Matrix{BF16: v}
	}
	return Matrix{F32: floats(d.take(4 * n))}
}

func (d *decoder) done() error {
	if d.err == nil && len(d.data) != 0 {
		d.err = errors.New("model part has the wrong size")
	}
	return d.err
}

// EncodeLayer serializes a layer part.
func EncodeLayer(c Config, l *Layer) []byte {
	b := make([]byte, 0, LayerBytes(c))
	for _, t := range l.tensors(c) {
		if t.mat != nil {
			b = appendMatrix(b, *t.mat)
		} else {
			b = append(b, f32Bytes(*t.vec)...)
		}
	}
	return b
}

// LoadLayer decodes a layer part.
func LoadLayer(c Config, data []byte) (*Layer, error) {
	d := &decoder{c: c, data: data}
	l := &Layer{}
	for _, t := range l.tensors(c) {
		if t.mat != nil {
			*t.mat = d.mat(t.size)
		} else {
			*t.vec = d.vec(t.size)
		}
	}
	return l, d.done()
}

// EncodeHead serializes the head part; a shared classifier is not stored.
func EncodeHead(c Config, h *Head) []byte {
	b := f32Bytes(h.RmsFinal)
	if !c.SharedClassifier {
		b = appendMatrix(b, h.Wcls)
	}
	return b
}

// LoadEmbed decodes the embedding table.
func LoadEmbed(c Config, data []byte) (Matrix, error) {
	d := &decoder{c: c, data: data}
	m := d.mat(c.Vocab * c.Dim)
	return m, d.done()
}

// EmbedBytes is the embedding table's size.
func EmbedBytes(c Config) int64 { return int64(c.Vocab) * int64(c.Dim) * elemBytes(c, true) }

// LoadHead decodes the head; a shared classifier is the embedding table.
func LoadHead(c Config, data []byte, embed Matrix) (*Head, error) {
	d := &decoder{c: c, data: data}
	h := &Head{RmsFinal: d.vec(c.Dim), Wcls: embed}
	if !c.SharedClassifier {
		h.Wcls = d.mat(c.Vocab * c.Dim)
	}
	return h, d.done()
}

// Import stores a llama2.c checkpoint's parts and manifest, returning the
// model's ID.
func Import(store *blob.Store, ck *Checkpoint, tokenizer []byte, name string) (blob.ID, *Manifest, error) {
	c := ck.Config
	m := &Manifest{Version: 1, Name: name, Config: c}
	put := func(b []byte) (string, error) {
		id, err := store.PutBytes(b)
		return id.String(), err
	}
	var err error
	if m.Embed, err = put(appendMatrix(nil, ck.Embed)); err != nil {
		return blob.ID{}, nil, err
	}
	for _, l := range ck.Layers {
		id, err := put(EncodeLayer(c, l))
		if err != nil {
			return blob.ID{}, nil, err
		}
		m.Layers = append(m.Layers, id)
	}
	if m.Head, err = put(EncodeHead(c, ck.Head)); err != nil {
		return blob.ID{}, nil, err
	}
	if m.Tokenizer, err = put(tokenizer); err != nil {
		return blob.ID{}, nil, err
	}
	data, _ := json.Marshal(m)
	id, err := store.PutBytes(data)
	return id, m, err
}
