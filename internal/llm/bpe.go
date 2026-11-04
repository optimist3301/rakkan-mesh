package llm

import (
	"unicode/utf8"

	"github.com/optimist3301/rakkan-mesh/internal/llm/bpe"
)

// Codec turns text into tokens and tokens back into bytes.
type Codec interface {
	// Encode tokenizes text, without a BOS token. Special tokens written
	// in the text, such as <|im_start|>, become those tokens.
	Encode(text string) []int
	// Piece returns the bytes token adds after prev; a character may span
	// several tokens.
	Piece(prev, token int) []byte
}

// llama2cCodec adapts a llama2.c Tokenizer.
type llama2cCodec struct{ *Tokenizer }

func (c llama2cCodec) Encode(text string) []int { return c.Tokenizer.Encode(text, false) }

func (c llama2cCodec) Piece(prev, token int) []byte { return []byte(c.Decode(prev, token)) }

// BPE is Hugging Face byte-level BPE (package bpe).
type BPE = bpe.Tokenizer

// ReadBPE parses a Hugging Face tokenizer.json.
func ReadBPE(data []byte, vocab int) (*BPE, error) { return bpe.Read(data, vocab) }

// readCodec loads a manifest's tokenizer.
func readCodec(m *Manifest, data []byte) (Codec, error) {
	if m.TokenizerFormat == "hf" {
		return ReadBPE(data, m.Config.Vocab)
	}
	t, err := ReadTokenizer(data, m.Config.Vocab)
	if err != nil {
		return nil, err
	}
	return llama2cCodec{t}, nil
}

// validUTF8Prefix splits b into its longest prefix of whole characters and
// the incomplete rest.
func validUTF8Prefix(b []byte) (whole, rest []byte) {
	for cut := len(b); cut >= max(0, len(b)-3); cut-- {
		if utf8.Valid(b[:cut]) {
			return b[:cut], b[cut:]
		}
	}
	return b, nil // invalid, not merely incomplete: pass it on
}
