package llm

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Tests with a real model need RAKKAN_LLM_DIR holding stories15M.bin and
// tokenizer.bin from Karpathy's llama2.c / tinyllamas.
func realModel(t testing.TB) (*Checkpoint, *Tokenizer) {
	t.Helper()
	dir := os.Getenv("RAKKAN_LLM_DIR")
	if dir == "" {
		t.Skip("set RAKKAN_LLM_DIR to a directory with stories15M.bin and tokenizer.bin")
	}
	ckData, err := os.ReadFile(filepath.Join(dir, "stories15M.bin"))
	if err != nil {
		t.Fatal(err)
	}
	ck, err := ReadCheckpoint(ckData)
	if err != nil {
		t.Fatal(err)
	}
	tokData, err := os.ReadFile(filepath.Join(dir, "tokenizer.bin"))
	if err != nil {
		t.Fatal(err)
	}
	tok, err := ReadTokenizer(tokData, ck.Config.Vocab)
	if err != nil {
		t.Fatal(err)
	}
	return ck, tok
}

// generateLocal runs the whole model in one stage, greedily.
func generateLocal(ck *Checkpoint, tok *Tokenizer, prompt string, steps int) ([]int, string) {
	c := ck.Config
	stage := NewStage(c, ck.Layers, NewRope(c, c.SeqLen), c.SeqLen)
	tokens := tok.Encode(prompt, true)
	var out strings.Builder
	sampler := NewSampler(0, 0)
	for pos := 0; pos < steps && pos < c.SeqLen; pos++ {
		x := Embed(c, ck.Embed, tokens[pos])
		stage.Forward(x, pos)
		if pos+1 < len(tokens) {
			continue // still reading the prompt
		}
		next := sampler.Sample(ck.Head.Logits(c, x))
		if next == TokenBOS || next == TokenEOS {
			break
		}
		out.WriteString(tok.Decode(tokens[pos], next))
		tokens = append(tokens, next)
	}
	return tokens, out.String()
}

func TestTokenizerMatchesLlama(t *testing.T) {
	_, tok := realModel(t)
	got := tok.Encode("Once upon a time", true)
	want := []int{1, 9038, 2501, 263, 931} // <s> ▁Once ▁upon ▁a ▁time
	if len(got) != len(want) {
		t.Fatalf("tokens %v, want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("tokens %v, want %v", got, want)
		}
	}
	var text strings.Builder
	for i := 1; i < len(got); i++ {
		text.WriteString(tok.Decode(got[i-1], got[i]))
	}
	if text.String() != "Once upon a time" {
		t.Fatalf("decoded %q", text.String())
	}
	// Characters outside the vocabulary fall back to bytes and round-trip.
	emoji := tok.Encode("🦀", false)
	var back []byte
	for i, id := range emoji {
		prev := 0
		if i > 0 {
			prev = emoji[i-1]
		}
		back = append(back, tok.Decode(prev, id)...)
	}
	if strings.TrimSpace(string(back)) != "🦀" {
		t.Fatalf("emoji round trip: %q from %v", back, emoji)
	}
}

func TestRealModelGeneratesStory(t *testing.T) {
	ck, tok := realModel(t)
	_, text := generateLocal(ck, tok, "Once upon a time", 80)
	t.Logf("generated: %q", text)
	words := strings.Fields(text)
	common := 0
	for _, w := range words {
		switch strings.ToLower(strings.Trim(w, ".,!?\"")) {
		case "the", "a", "and", "was", "to", "she", "he", "her", "his", "they", "it", "there", "little":
			common++
		}
	}
	if len(words) < 30 || common*5 < len(words) {
		t.Fatalf("does not read like English: %q", text)
	}
}

func tokenizerBytesFromDir(t *testing.T) []byte {
	b, err := os.ReadFile(filepath.Join(os.Getenv("RAKKAN_LLM_DIR"), "tokenizer.bin"))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestRealCheckpointRoundTrip(t *testing.T) {
	realModel(t)
	data, _ := os.ReadFile(filepath.Join(os.Getenv("RAKKAN_LLM_DIR"), "stories15M.bin"))
	ck, _ := ReadCheckpoint(data)
	enc := ck.Encode()
	// Only the legacy rotary tables, written as zeros, may differ.
	rope := 4 * ck.Config.SeqLen * ck.Config.HeadSize()
	start := len(enc) - rope
	if !ck.Config.SharedClassifier {
		start -= 4 * ck.Config.Vocab * ck.Config.Dim
	}
	if len(enc) != len(data) || string(enc[:start]) != string(data[:start]) || string(enc[start+rope:]) != string(data[start+rope:]) {
		t.Fatal("re-encoded checkpoint differs from the original")
	}
}
