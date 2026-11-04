package dsv41

import (
	"fmt"
	"math"
	"math/big"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"

	"github.com/optimist3301/rakkan-mesh/internal/llm/bpe"
)

// BuildEngramTables computes the Engram hash tables from the tokenizer and
// the config, as the reference's engram.py does with the tokenizers
// library, sympy and numpy (docs/dev/deepseek-v41.md, section 8).
func BuildEngramTables(c *Config, tok *bpe.Tokenizer) (*EngramTables, error) {
	if tok.Len() < c.Vocab {
		return nil, fmt.Errorf("dsv41: tokenizer of %d tokens for a vocabulary of %d", tok.Len(), c.Vocab)
	}
	tokenMap, compressed := compressedTokenMap(tok, c.Vocab)
	if compressed != c.EngramCompressed {
		// Every multiplier derives from this size: a mismatch rehashes
		// every n-gram, silently.
		return nil, fmt.Errorf("dsv41: the tokenizer compresses to %d ids, the model expects %d", compressed, c.EngramCompressed)
	}
	t := &EngramTables{MaxNgram: c.EngramMaxNgram, Heads: c.EngramHeads, Compressed: compressed, TokenMap: tokenMap}
	t.Pad = tokenMap[c.EngramPad]
	bound := max(1, (math.MaxInt64/int64(compressed))/2)
	seen := map[int64]bool{}
	for li, layer := range c.EngramLayers {
		var l engramLayer
		l.Layer, l.Rows = layer, c.EngramRows[li]
		var offset, total int64
		for range c.EngramMaxNgram - 1 {
			var sizes []int64
			current := int64(c.EngramVocab - 1)
			for range c.EngramHeads {
				current = nextPrime(current, seen)
				seen[current] = true
				sizes = append(sizes, current)
				l.Offsets = append(l.Offsets, offset)
				offset += current
				total += current
			}
			l.Primes = append(l.Primes, sizes)
		}
		if total != int64(l.Rows) {
			return nil, fmt.Errorf("dsv41: Engram layer %d: tables of %d rows, the model has %d", layer, total, l.Rows)
		}
		m, err := numpyIntegers(10007*uint64(layer), bound, c.EngramMaxNgram)
		if err != nil {
			return nil, err
		}
		for i := range m {
			m[i] = m[i]*2 + 1 // odd
		}
		l.Multipliers = m
		t.Layers = append(t.Layers, l)
	}
	return t, nil
}

// nextPrime is the smallest prime above start not yet seen. ProbablyPrime
// is exact below 2^64.
func nextPrime(start int64, seen map[int64]bool) int64 {
	for c := start + 1; ; c++ {
		if !seen[c] && big.NewInt(c).ProbablyPrime(0) {
			return c
		}
	}
}

// compressedTokenMap maps each of the first n token IDs onto a smaller id
// space where tokens that normalize alike collapse (build_compressed_token_map):
// " The", "the" and "THE" hash the same. IDs are given in order of first
// appearance.
func compressedTokenMap(tok *bpe.Tokenizer, n int) ([]int64, int) {
	ids := map[string]int64{}
	out := make([]int64, n)
	for id := range n {
		text := decodeLossy(tok, id)
		var key string
		if strings.ContainsRune(text, utf8.RuneError) {
			key = tok.Token(id) // part of a UTF-8 character: its raw form
		} else if key = normalizeEngram(text); key == "" {
			key = text
		}
		v, ok := ids[key]
		if !ok {
			v = int64(len(ids))
			ids[key] = v
		}
		out[id] = v
	}
	return out, len(ids)
}

// decodeLossy is the tokenizers library's decode of one token: added
// tokens as their content, others their bytes as UTF-8 with each invalid
// sequence replaced by U+FFFD.
func decodeLossy(tok *bpe.Tokenizer, id int) string {
	if tok.IsAdded(id) {
		return tok.Token(id)
	}
	return strings.ToValidUTF8(string(tok.Piece(0, id)), "�")
}

var (
	spaceRun = regexp.MustCompile(`[ \t\r\n]+`)
	sentinel = "" // private use: keeps a lone space from stripping to nothing
)

// normalizeEngram is the normalizer chain of build_compressed_token_map:
// NFKC, NFD, strip combining marks (all of category M: vowel signs of
// Indic scripts go too), lowercase, collapse runs of spaces,
// tabs and line breaks to a space, then strip surrounding whitespace -
// except a text of exactly one space, which stays.
func normalizeEngram(s string) string {
	s = norm.NFD.String(norm.NFKC.String(s))
	var b strings.Builder
	for _, r := range s {
		switch {
		case unicode.IsMark(r): // is_combining_mark: Mn, Mc and Me alike
		case r == 'İ':
			b.WriteString("i̇") // Rust's full lowercase mapping; NFD has split it anyway
		default:
			b.WriteRune(unicode.ToLower(r))
		}
	}
	s = spaceRun.ReplaceAllString(b.String(), " ")
	if s == " " {
		s = sentinel
	}
	s = strings.TrimFunc(s, unicode.IsSpace)
	return strings.ReplaceAll(s, sentinel, " ")
}
