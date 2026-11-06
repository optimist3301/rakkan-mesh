package llm

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"unicode/utf8"
)

// Token IDs fixed by the LLaMA tokenizer.
const (
	TokenBOS = 1
	TokenEOS = 2
)

// Tokenizer is a byte-pair-encoding tokenizer in llama2.c's format:
// max_token_length (int32), then per token its score (float32), length
// (int32) and bytes. Spaces are literal (llama2.c replaced SentencePiece's
// "▁" with " "), and tokens 3..258 stand for raw bytes.
type Tokenizer struct {
	vocab  []string
	scores []float32
	index  map[string]int
}

// ReadTokenizer parses a tokenizer for a model with vocab tokens.
func ReadTokenizer(data []byte, vocab int) (*Tokenizer, error) {
	if len(data) < 4 {
		return nil, errors.New("tokenizer too short")
	}
	t := &Tokenizer{index: make(map[string]int, vocab)}
	off := 4
	for i := range vocab {
		if off+8 > len(data) {
			return nil, fmt.Errorf("tokenizer truncated at token %d", i)
		}
		score := math.Float32frombits(binary.LittleEndian.Uint32(data[off:]))
		n := int(int32(binary.LittleEndian.Uint32(data[off+4:])))
		off += 8
		if n < 0 || off+n > len(data) {
			return nil, fmt.Errorf("tokenizer truncated at token %d", i)
		}
		s := string(data[off : off+n])
		off += n
		t.vocab = append(t.vocab, s)
		t.scores = append(t.scores, score)
		if _, dup := t.index[s]; !dup {
			t.index[s] = i
		}
	}
	return t, nil
}

// Encode tokenizes text as llama2.c does: optional BOS, a leading space
// (SentencePiece's dummy prefix), a token per character (or per byte for
// characters missing from the vocabulary), then repeated merging of the
// adjacent pair whose merged token scores highest.
func (t *Tokenizer) Encode(text string, bos bool) []int {
	var tokens []int
	if bos {
		tokens = append(tokens, TokenBOS)
	}
	if text != "" {
		if id, ok := t.index[" "]; ok {
			tokens = append(tokens, id)
		}
	}
	for i := 0; i < len(text); {
		_, size := utf8.DecodeRuneInString(text[i:])
		if id, ok := t.index[text[i:i+size]]; ok {
			tokens = append(tokens, id)
		} else {
			// Byte fallback, as run.c: tokens 3..258 are the bytes. A
			// vocabulary without them (a toy one) drops the character
			// rather than naming tokens it lacks.
			for _, b := range []byte(text[i : i+size]) {
				if id := int(b) + 3; id < len(t.vocab) {
					tokens = append(tokens, id)
				}
			}
		}
		i += size
	}
	start := 0
	if bos {
		start = 1 // never merge into BOS
	}
	for {
		best, bestID, bestAt := float32(-1e10), -1, -1
		for i := start; i+1 < len(tokens); i++ {
			if id, ok := t.index[t.vocab[tokens[i]]+t.vocab[tokens[i+1]]]; ok && t.scores[id] > best {
				best, bestID, bestAt = t.scores[id], id, i
			}
		}
		if bestAt < 0 {
			return tokens
		}
		tokens[bestAt] = bestID
		tokens = append(tokens[:bestAt+1], tokens[bestAt+2:]...)
	}
}

// Decode returns the text of token, given the token before it.
func (t *Tokenizer) Decode(prev, token int) string {
	if token < 0 || token >= len(t.vocab) {
		return ""
	}
	piece := t.vocab[token]
	if prev == TokenBOS && len(piece) > 0 && piece[0] == ' ' {
		piece = piece[1:]
	}
	var b byte
	if n, _ := fmt.Sscanf(piece, "<0x%02X>", &b); n == 1 && len(piece) == 6 {
		return string([]byte{b})
	}
	return piece
}

// VocabSize is the number of tokens.
func (t *Tokenizer) VocabSize() int { return len(t.vocab) }
