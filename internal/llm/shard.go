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

// Import stores a model's parts and manifest, returning the model's ID.
func Import(store *blob.Store, ck *Checkpoint, tokenizer []byte, name string) (blob.ID, *Manifest, error) {
	c := ck.Config
	m := &Manifest{Version: 1, Name: name, Config: c}
	put := func(b []byte) (string, error) {
		id, err := store.PutBytes(b)
		return id.String(), err
	}
	var err error
	if m.Embed, err = put(f32Bytes(ck.Embed)); err != nil {
		return blob.ID{}, nil, err
	}
	for _, l := range ck.Layers {
		var parts [][]float32
		for _, t := range l.tensors() {
			parts = append(parts, *t)
		}
		id, err := put(f32Bytes(parts...))
		if err != nil {
			return blob.ID{}, nil, err
		}
		m.Layers = append(m.Layers, id)
	}
	head := [][]float32{ck.Head.RmsFinal}
	if !c.SharedClassifier {
		head = append(head, ck.Head.Wcls)
	}
	if m.Head, err = put(f32Bytes(head...)); err != nil {
		return blob.ID{}, nil, err
	}
	if m.Tokenizer, err = put(tokenizer); err != nil {
		return blob.ID{}, nil, err
	}
	data, _ := json.Marshal(m)
	id, err := store.PutBytes(data)
	return id, m, err
}

// LoadLayer decodes a layer part.
func LoadLayer(c Config, data []byte) (*Layer, error) {
	sizes := c.layerSizes()
	total := 0
	for _, s := range sizes {
		total += s
	}
	if len(data) != 4*total {
		return nil, errors.New("layer part has the wrong size")
	}
	all := floats(data)
	l := &Layer{}
	for i, t := range l.tensors() {
		*t, all = all[:sizes[i]], all[sizes[i]:]
	}
	return l, nil
}

// LoadEmbed decodes the embedding table.
func LoadEmbed(c Config, data []byte) ([]float32, error) {
	if len(data) != 4*c.Vocab*c.Dim {
		return nil, errors.New("embedding part has the wrong size")
	}
	return floats(data), nil
}

// LoadHead decodes the head; a shared classifier is the embedding table.
func LoadHead(c Config, data []byte, embed []float32) (*Head, error) {
	want := c.Dim
	if !c.SharedClassifier {
		want += c.Vocab * c.Dim
	}
	if len(data) != 4*want {
		return nil, errors.New("head part has the wrong size")
	}
	all := floats(data)
	h := &Head{RmsFinal: all[:c.Dim], Wcls: embed}
	if !c.SharedClassifier {
		h.Wcls = all[c.Dim:]
	}
	return h, nil
}
