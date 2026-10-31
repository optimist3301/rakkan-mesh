package llm

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
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

// BPE is a byte-level byte-pair-encoding tokenizer as Hugging Face's
// tokenizers library runs it for GPT-2 style models (Qwen, Llama 3): NFC
// normalization, splitting into words by the model's pattern, mapping each
// byte to a printable character, then merging adjacent symbols by the
// merges' priority. Added tokens (the special ones and others) are matched
// in the text first and encoded as themselves.
type BPE struct {
	vocab   map[string]int
	tokens  []string // by ID, byte-level strings
	ranks   map[[2]string]int
	added   map[string]int // content -> ID
	addedBy []string       // added tokens, longest first
	special map[int]bool
	nfc     bool
	digits  int  // at most this many digits per word: 1 (Qwen) or 3 (Llama 3)
	ignore  bool // ignore_merges: a word in the vocabulary is one token

	mu    sync.Mutex
	cache map[string][]int
}

type tokenizerJSON struct {
	Normalizer *struct {
		Type string `json:"type"`
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
	} `json:"added_tokens"`
}

// The word-splitting patterns supported: Qwen2's, and Llama 3's, which
// differs in taking up to three digits at a time.
const (
	patternQwen   = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	patternLlama3 = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
)

// ReadBPE parses a Hugging Face tokenizer.json. Only byte-level BPE with a
// supported splitting pattern is accepted: anything else would tokenize
// differently from the model's training, silently.
func ReadBPE(data []byte, vocab int) (*BPE, error) {
	var tj tokenizerJSON
	if err := json.Unmarshal(data, &tj); err != nil {
		return nil, fmt.Errorf("tokenizer.json: %w", err)
	}
	if tj.Model.Type != "BPE" || tj.Model.ByteFallback {
		return nil, fmt.Errorf("tokenizer.json: model %q is not byte-level BPE", tj.Model.Type)
	}
	t := &BPE{vocab: tj.Model.Vocab, ranks: map[[2]string]int{}, added: map[string]int{}, special: map[int]bool{},
		ignore: tj.Model.IgnoreMerges, cache: map[string][]int{}}
	if tj.Normalizer != nil {
		if tj.Normalizer.Type != "NFC" {
			return nil, fmt.Errorf("tokenizer.json: normalizer %q", tj.Normalizer.Type)
		}
		t.nfc = true
	}
	pattern, err := splitPattern(tj.PreTokenizer)
	if err != nil {
		return nil, err
	}
	switch pattern {
	case patternQwen:
		t.digits = 1
	case patternLlama3:
		t.digits = 3
	default:
		return nil, fmt.Errorf("tokenizer.json: unsupported splitting pattern %q", pattern)
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

// splitPattern extracts the regex of a Sequence[Split, ByteLevel(no
// regex)] pre-tokenizer, the form Qwen and Llama 3 use.
func splitPattern(raw json.RawMessage) (string, error) {
	type pre struct {
		Type    string `json:"type"`
		Pattern struct {
			Regex string `json:"Regex"`
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
		return "", fmt.Errorf("tokenizer.json pre_tokenizer: %w", err)
	}
	p := seq.Pretokenizers
	if seq.Type != "Sequence" || len(p) != 2 || p[0].Type != "Split" || p[0].Behavior != "Isolated" || p[0].Invert ||
		p[1].Type != "ByteLevel" || p[1].UseRegex || p[1].AddPrefixSpac {
		return "", errors.New("tokenizer.json: unsupported pre-tokenizer")
	}
	return p[0].Pattern.Regex, nil
}

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

// Encode implements Codec.
func (t *BPE) Encode(text string) []int {
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

func (t *BPE) encodeOrdinary(out []int, text string) []int {
	if text == "" {
		return out
	}
	if t.nfc {
		text = norm.NFC.String(text)
	}
	for _, w := range t.split(text) {
		out = append(out, t.word(w)...)
	}
	return out
}

// word encodes one word: its bytes as characters, merged by rank.
func (t *BPE) word(w string) []int {
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

// Piece implements Codec.
func (t *BPE) Piece(_, token int) []byte {
	if token < 0 || token >= len(t.tokens) {
		return nil
	}
	s := t.tokens[token]
	if _, ok := t.added[s]; ok {
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

// Special reports whether a token is a special added token.
func (t *BPE) Special(token int) bool { return t.special[token] }

// split divides text into words by the model's pattern:
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d)   contractions
//	[^\r\n\p{L}\p{N}]?\p{L}+       a word, with one leading non-letter
//	\p{N}{1,digits}                digits
//	 ?[^\s\p{L}\p{N}]+[\r\n]*      punctuation, with one leading space
//	\s*[\r\n]+                     whitespace through the last line break
//	\s+(?!\S)                      whitespace not before a non-space
//	\s+                            whitespace
//
// Go's regexp has no lookahead, so the alternatives are matched by hand,
// in order, at each position, as a backtracking engine would.
func (t *BPE) split(text string) []string {
	rs := []rune(text)
	var out []string
	for i := 0; i < len(rs); {
		n := t.match(rs, i)
		out = append(out, string(rs[i:i+n]))
		i += n
	}
	return out
}

func isL(r rune) bool  { return unicode.IsLetter(r) }
func isN(r rune) bool  { return unicode.IsNumber(r) }
func isWS(r rune) bool { return unicode.IsSpace(r) }
func isNL(r rune) bool { return r == '\r' || r == '\n' }

// match returns the length of the word starting at rs[i].
func (t *BPE) match(rs []rune, i int) int {
	r := rs[i]
	at := func(j int) (rune, bool) {
		if j < len(rs) {
			return rs[j], true
		}
		return 0, false
	}
	// Contractions.
	if r == '\'' {
		for _, c := range []string{"s", "t", "re", "ve", "m", "ll", "d"} {
			if i+1+len(c) <= len(rs) && strings.EqualFold(string(rs[i+1:i+1+len(c)]), c) {
				return 1 + len(c)
			}
		}
	}
	// Letters, with one optional leading non-letter, non-digit, non-newline.
	j := i
	if !isL(r) && !isN(r) && !isNL(r) {
		if next, ok := at(i + 1); ok && isL(next) {
			j = i + 1
		}
	}
	if c, ok := at(j); ok && isL(c) {
		for j < len(rs) && isL(rs[j]) {
			j++
		}
		return j - i
	}
	// Digits.
	if isN(r) {
		j = i
		for j < len(rs) && j-i < t.digits && isN(rs[j]) {
			j++
		}
		return j - i
	}
	// Punctuation: optional space, then non-space non-letter non-digits,
	// then line breaks.
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
	// Whitespace.
	end := i
	for end < len(rs) && isWS(rs[end]) {
		end++
	}
	if end == i {
		return 1 // unreachable: every rune is matched above
	}
	// Through the last line break in the run.
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
