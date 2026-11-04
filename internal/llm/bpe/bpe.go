// Package bpe is a byte-level byte-pair-encoding tokenizer as Hugging
// Face's tokenizers library runs it for GPT-2 style models (Qwen, Llama 3,
// DeepSeek): optional NFC normalization, splitting into words by the
// model's patterns, mapping each byte to a printable character, then
// merging adjacent symbols by the merges' priority. Added tokens (the
// special ones and others) are matched in the text first and encoded as
// themselves.
package bpe

import (
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strings"
	"sync"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// Tokenizer encodes text into token IDs and decodes them back.
type Tokenizer struct {
	vocab   map[string]int
	tokens  []string // by ID, byte-level strings or added tokens' content
	ranks   map[[2]string]int
	added   map[string]int // content -> ID
	addedBy []string       // added tokens, longest first
	special map[int]bool
	nfc     bool
	splits  []splitter // applied in order, each to every piece of the last
	ignore  bool       // ignore_merges: a word in the vocabulary is one token

	mu    sync.Mutex
	cache map[string][]int
}

type tokenizerJSON struct {
	Normalizer *struct {
		Type        string `json:"type"`
		Normalizers []struct {
			Type string `json:"type"`
		} `json:"normalizers"`
	} `json:"normalizer"`
	PreTokenizer json.RawMessage `json:"pre_tokenizer"`
	Model        struct {
		Type         string          `json:"type"`
		Vocab        map[string]int  `json:"vocab"`
		Merges       json.RawMessage `json:"merges"`
		IgnoreMerges bool            `json:"ignore_merges"`
		ByteFallback bool            `json:"byte_fallback"`
	} `json:"model"`
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
		LStrip  bool   `json:"lstrip"`
		RStrip  bool   `json:"rstrip"`
		Single  bool   `json:"single_word"`
	} `json:"added_tokens"`
}

// Read parses a Hugging Face tokenizer.json for a model of vocab tokens.
// Only byte-level BPE with supported normalization and splitting is
// accepted: anything else would tokenize differently from the model's
// training, silently.
func Read(data []byte, vocab int) (*Tokenizer, error) {
	var tj tokenizerJSON
	if err := json.Unmarshal(data, &tj); err != nil {
		return nil, fmt.Errorf("tokenizer.json: %w", err)
	}
	if tj.Model.Type != "BPE" || tj.Model.ByteFallback {
		return nil, fmt.Errorf("tokenizer.json: model %q is not byte-level BPE", tj.Model.Type)
	}
	t := &Tokenizer{vocab: tj.Model.Vocab, ranks: map[[2]string]int{}, added: map[string]int{}, special: map[int]bool{},
		ignore: tj.Model.IgnoreMerges, cache: map[string][]int{}}
	if n := tj.Normalizer; n != nil {
		switch {
		case n.Type == "NFC":
			t.nfc = true
		case n.Type == "Sequence" && len(n.Normalizers) == 0:
		case n.Type == "Sequence" && len(n.Normalizers) == 1 && n.Normalizers[0].Type == "NFC":
			t.nfc = true
		default:
			return nil, fmt.Errorf("tokenizer.json: normalizer %q not supported", n.Type)
		}
	}
	var err error
	if t.splits, err = splitters(tj.PreTokenizer); err != nil {
		return nil, err
	}

	// Merges are "a b" strings, or [a, b] pairs in newer files.
	var merges []json.RawMessage
	if err := json.Unmarshal(tj.Model.Merges, &merges); err != nil {
		return nil, fmt.Errorf("tokenizer.json merges: %w", err)
	}
	for i, raw := range merges {
		var pair [2]string
		var s string
		if json.Unmarshal(raw, &s) == nil {
			a, b, ok := strings.Cut(s, " ")
			if !ok {
				return nil, fmt.Errorf("tokenizer.json: merge %q", s)
			}
			pair = [2]string{a, b}
		} else if err := json.Unmarshal(raw, &pair); err != nil {
			return nil, fmt.Errorf("tokenizer.json merge %d: %w", i, err)
		}
		if _, dup := t.ranks[pair]; !dup {
			t.ranks[pair] = i
		}
	}
	for _, a := range tj.AddedTokens {
		if a.LStrip || a.RStrip || a.Single {
			return nil, fmt.Errorf("tokenizer.json: added token %q strips or matches whole words, not supported", a.Content)
		}
		t.added[a.Content] = a.ID
		t.addedBy = append(t.addedBy, a.Content)
		if a.Special {
			t.special[a.ID] = true
		}
	}
	slices.SortFunc(t.addedBy, func(a, b string) int { return len(b) - len(a) })

	n := vocab
	for _, id := range t.vocab {
		n = max(n, id+1)
	}
	for _, id := range t.added {
		n = max(n, id+1)
	}
	if n > vocab {
		return nil, fmt.Errorf("tokenizer has token %d beyond the model's vocabulary of %d", n-1, vocab)
	}
	t.tokens = make([]string, vocab)
	for s, id := range t.vocab {
		if id < 0 {
			return nil, errors.New("tokenizer.json: negative token ID")
		}
		t.tokens[id] = s
	}
	for s, id := range t.added {
		if id < 0 {
			return nil, errors.New("tokenizer.json: negative token ID")
		}
		t.tokens[id] = s
	}
	return t, nil
}

// Len is the number of token IDs (the model's vocabulary).
func (t *Tokenizer) Len() int { return len(t.tokens) }

// Token returns a token's string as tokenizer.json has it: the byte-level
// form for vocabulary tokens, the content for added ones (id_to_token).
func (t *Tokenizer) Token(id int) string { return t.tokens[id] }

// IsAdded reports whether a token is an added token.
func (t *Tokenizer) IsAdded(id int) bool {
	a, ok := t.added[t.tokens[id]]
	return ok && a == id
}

// Special reports whether a token is a special added token.
func (t *Tokenizer) Special(token int) bool { return t.special[token] }

// byteChars maps each byte to the printable character GPT-2's byte-level
// encoding uses for it, and charBytes back.
var byteChars, charBytes = func() ([256]rune, map[rune]byte) {
	var bc [256]rune
	cb := map[rune]byte{}
	n := 0
	for b := range 256 {
		if b >= '!' && b <= '~' || b >= 0xa1 && b <= 0xac || b >= 0xae && b <= 0xff {
			bc[b] = rune(b)
		} else {
			bc[b] = rune(256 + n)
			n++
		}
		cb[bc[b]] = byte(b)
	}
	return bc, cb
}()

// Encode tokenizes text, without a BOS token. Added tokens written in the
// text, such as <|im_start|>, become those tokens.
func (t *Tokenizer) Encode(text string) []int {
	var out []int
	for len(text) > 0 {
		// The next added token, earliest and then longest.
		at, which := len(text), ""
		for _, a := range t.addedBy {
			if i := strings.Index(text[:min(len(text), at+len(a))], a); i >= 0 && (i < at || i == at && len(a) > len(which)) {
				at, which = i, a
			}
		}
		out = t.encodeOrdinary(out, text[:at])
		if which == "" {
			break
		}
		out = append(out, t.added[which])
		text = text[at+len(which):]
	}
	return out
}

func (t *Tokenizer) encodeOrdinary(out []int, text string) []int {
	if text == "" {
		return out
	}
	if t.nfc {
		text = norm.NFC.String(text)
	}
	pieces := []string{text}
	for _, s := range t.splits {
		var next []string
		for _, p := range pieces {
			next = s.split(next, p)
		}
		pieces = next
	}
	for _, w := range pieces {
		out = append(out, t.word(w)...)
	}
	return out
}

// word encodes one word: its bytes as characters, merged by rank.
func (t *Tokenizer) word(w string) []int {
	t.mu.Lock()
	ids, ok := t.cache[w]
	t.mu.Unlock()
	if ok {
		return ids
	}
	var mapped strings.Builder
	for i := 0; i < len(w); i++ {
		mapped.WriteRune(byteChars[w[i]])
	}
	if id, ok := t.vocab[mapped.String()]; ok && t.ignore {
		return []int{id}
	}
	var syms []string
	for _, r := range mapped.String() {
		syms = append(syms, string(r))
	}
	for len(syms) > 1 {
		best, at := -1, -1
		for i := 0; i+1 < len(syms); i++ {
			if r, ok := t.ranks[[2]string{syms[i], syms[i+1]}]; ok && (best < 0 || r < best) {
				best, at = r, i
			}
		}
		if at < 0 {
			break
		}
		syms[at] += syms[at+1]
		syms = append(syms[:at+1], syms[at+2:]...)
	}
	ids = make([]int, 0, len(syms))
	for _, s := range syms {
		if id, ok := t.vocab[s]; ok {
			ids = append(ids, id)
		}
		// Every single byte is in a byte-level vocabulary, so a symbol is
		// missing only from a malformed one; it is dropped.
	}
	t.mu.Lock()
	if len(t.cache) > 1<<16 {
		clear(t.cache)
	}
	t.cache[w] = ids
	t.mu.Unlock()
	return ids
}

// Piece returns the bytes a token decodes to; a character may span
// several tokens. prev is unused: byte-level tokens decode alone.
func (t *Tokenizer) Piece(_, token int) []byte {
	if token < 0 || token >= len(t.tokens) {
		return nil
	}
	s := t.tokens[token]
	if t.IsAdded(token) {
		return []byte(s)
	}
	b := make([]byte, 0, len(s))
	for _, r := range s {
		if c, ok := charBytes[r]; ok {
			b = append(b, c)
		}
	}
	return b
}

// splitter is one Split pre-tokenizer with "Isolated" behaviour: the
// matches of its pattern become pieces, and so does each stretch of text
// between them.
type splitter interface {
	split(out []string, text string) []string
}

type regexSplitter struct{ re *regexp.Regexp }

func (s regexSplitter) split(out []string, text string) []string {
	at := 0
	for _, m := range s.re.FindAllStringIndex(text, -1) {
		if m[0] > at {
			out = append(out, text[at:m[0]])
		}
		if m[1] > m[0] {
			out = append(out, text[m[0]:m[1]])
		}
		at = m[1]
	}
	if at < len(text) {
		out = append(out, text[at:])
	}
	return out
}

// handSplitter matches a pattern Go's regexp cannot express (it has no
// lookahead): match returns the length of the match at rs[i], or 0.
type handSplitter struct{ match func(rs []rune, i int) int }

func (s handSplitter) split(out []string, text string) []string {
	rs := []rune(text)
	gap := 0
	for i := 0; i < len(rs); {
		n := s.match(rs, i)
		if n == 0 {
			i++
			continue
		}
		if i > gap {
			out = append(out, string(rs[gap:i]))
		}
		out = append(out, string(rs[i:i+n]))
		i += n
		gap = i
	}
	if gap < len(rs) {
		out = append(out, string(rs[gap:]))
	}
	return out
}

// Patterns with lookahead, matched by hand.
var handPatterns = map[string]func(rs []rune, i int) int{
	// Qwen2.
	`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`: func(rs []rune, i int) int {
		return matchGPT(rs, i, 1)
	},
	// Llama 3: digits three at a time.
	`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`: func(rs []rune, i int) int {
		return matchGPT(rs, i, 3)
	},
	// DeepSeek V3 and later, after splitting off digits and CJK runs.
	"[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+": matchDeepSeek,
}

// splitters reads a Sequence[Split..., ByteLevel(no regex)] pre-tokenizer.
func splitters(raw json.RawMessage) ([]splitter, error) {
	type pre struct {
		Type    string `json:"type"`
		Pattern struct {
			Regex  *string `json:"Regex"`
			String *string `json:"String"`
		} `json:"pattern"`
		Behavior      string `json:"behavior"`
		Invert        bool   `json:"invert"`
		UseRegex      bool   `json:"use_regex"`
		AddPrefixSpac bool   `json:"add_prefix_space"`
	}
	var seq struct {
		Type          string `json:"type"`
		Pretokenizers []pre  `json:"pretokenizers"`
	}
	if err := json.Unmarshal(raw, &seq); err != nil {
		return nil, fmt.Errorf("tokenizer.json pre_tokenizer: %w", err)
	}
	p := seq.Pretokenizers
	if seq.Type != "Sequence" || len(p) < 2 {
		return nil, errors.New("tokenizer.json: unsupported pre-tokenizer")
	}
	last := p[len(p)-1]
	if last.Type != "ByteLevel" || last.UseRegex || last.AddPrefixSpac {
		return nil, errors.New("tokenizer.json: unsupported pre-tokenizer: it must end with ByteLevel without its own regex")
	}
	var out []splitter
	for _, s := range p[:len(p)-1] {
		if s.Type != "Split" || s.Behavior != "Isolated" || s.Invert || s.Pattern.Regex == nil {
			return nil, fmt.Errorf("tokenizer.json: unsupported pre-tokenizer step %s/%s", s.Type, s.Behavior)
		}
		pattern := *s.Pattern.Regex
		if m, ok := handPatterns[pattern]; ok {
			out = append(out, handSplitter{m})
			continue
		}
		re, err := regexp.Compile(pattern)
		if err != nil {
			return nil, fmt.Errorf("tokenizer.json: unsupported splitting pattern %q", pattern)
		}
		out = append(out, regexSplitter{re})
	}
	return out, nil
}

func isL(r rune) bool  { return unicode.IsLetter(r) }
func isN(r rune) bool  { return unicode.IsNumber(r) }
func isM(r rune) bool  { return unicode.IsMark(r) }
func isP(r rune) bool  { return unicode.IsPunct(r) }
func isS(r rune) bool  { return unicode.IsSymbol(r) }
func isWS(r rune) bool { return unicode.IsSpace(r) }
func isNL(r rune) bool { return r == '\r' || r == '\n' }

// matchWhitespace matches the three whitespace alternatives these
// patterns end with, at a whitespace character:
//
//	\s*[\r\n]+   whitespace through the last line break
//	\s+(?!\S)    whitespace not before a non-space
//	\s+          whitespace
func matchWhitespace(rs []rune, i int) int {
	end := i
	for end < len(rs) && isWS(rs[end]) {
		end++
	}
	if end == i {
		return 0
	}
	for k := end - 1; k >= i; k-- {
		if isNL(rs[k]) {
			return k + 1 - i
		}
	}
	// Not before a non-space: at the end of the text, all of it; else all
	// but the last, which goes with the next word.
	if end == len(rs) || end-i == 1 {
		return end - i
	}
	return end - 1 - i
}

// matchGPT matches Qwen2's and Llama 3's pattern:
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d)   contractions
//	[^\r\n\p{L}\p{N}]?\p{L}+       a word, with one leading non-letter
//	\p{N}{1,digits}                digits
//	 ?[^\s\p{L}\p{N}]+[\r\n]*      punctuation, with one leading space
//	whitespace (matchWhitespace)
func matchGPT(rs []rune, i, digits int) int {
	r := rs[i]
	if r == '\'' {
		for _, c := range []string{"s", "t", "re", "ve", "m", "ll", "d"} {
			if i+1+len(c) <= len(rs) && strings.EqualFold(string(rs[i+1:i+1+len(c)]), c) {
				return 1 + len(c)
			}
		}
	}
	j := i
	if !isL(r) && !isN(r) && !isNL(r) && i+1 < len(rs) && isL(rs[i+1]) {
		j = i + 1
	}
	if j < len(rs) && isL(rs[j]) {
		for j < len(rs) && isL(rs[j]) {
			j++
		}
		return j - i
	}
	if isN(r) {
		j = i
		for j < len(rs) && j-i < digits && isN(rs[j]) {
			j++
		}
		return j - i
	}
	j = i
	if r == ' ' {
		j++
	}
	k := j
	for k < len(rs) && !isWS(rs[k]) && !isL(rs[k]) && !isN(rs[k]) {
		k++
	}
	if k > j {
		for k < len(rs) && isNL(rs[k]) {
			k++
		}
		return k - i
	}
	return matchWhitespace(rs, i)
}

const asciiPunct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

// matchDeepSeek matches DeepSeek's third splitting pattern:
//
//	[ASCII punctuation][A-Za-z]+          one punctuation mark and ASCII letters
//	[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+  letters and marks, with one leading other
//	 ?[\p{P}\p{S}]+[\r\n]*                punctuation and symbols, one leading space
//	whitespace (matchWhitespace)
//
// Characters none of these take (digits, which an earlier split isolates,
// control and format characters) are left between matches.
func matchDeepSeek(rs []rune, i int) int {
	r := rs[i]
	isLetterASCII := func(c rune) bool { return c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' }
	if r < 128 && strings.ContainsRune(asciiPunct, r) && i+1 < len(rs) && isLetterASCII(rs[i+1]) {
		j := i + 1
		for j < len(rs) && isLetterASCII(rs[j]) {
			j++
		}
		return j - i
	}
	isLM := func(c rune) bool { return isL(c) || isM(c) }
	j := i
	if !isNL(r) && !isL(r) && !isP(r) && !isS(r) && i+1 < len(rs) && isLM(rs[i+1]) {
		j = i + 1
	}
	if j < len(rs) && isLM(rs[j]) {
		for j < len(rs) && isLM(rs[j]) {
			j++
		}
		return j - i
	}
	j = i
	if r == ' ' {
		j++
	}
	k := j
	for k < len(rs) && (isP(rs[k]) || isS(rs[k])) {
		k++
	}
	if k > j {
		for k < len(rs) && isNL(rs[k]) {
			k++
		}
		return k - i
	}
	return matchWhitespace(rs, i)
}
