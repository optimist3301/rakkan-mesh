package llm

import (
	"os"
	"slices"
	"testing"

	"github.com/optimist3301/rakkan-mesh/internal/llm/dsv41"
)

func TestChatHistory(t *testing.T) {
	qwen := &ChatTemplate{Prefix: "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\n", Suffix: "<|im_end|>\n<|im_start|>assistant\n"}
	got, err := qwen.render([]Turn{{"hi", "hello"}, {"2+2?", "4"}}, "and 3+3?")
	if err != nil {
		t.Fatal(err)
	}
	want := "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello<|im_end|>\n" +
		"<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>\n" +
		"<|im_start|>user\nand 3+3?<|im_end|>\n<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("qwen:\n%q\nwant\n%q", got, want)
	}
	llama := &ChatTemplate{Prefix: "<|start_header_id|>user<|end_header_id|>\n\n", Suffix: "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"}
	got, _ = llama.render([]Turn{{"hi", "hello"}}, "bye")
	want = "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nhello<|eot_id|>" +
		"<|start_header_id|>user<|end_header_id|>\n\nbye<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
	if got != want {
		t.Fatalf("llama:\n%q\nwant\n%q", got, want)
	}
	if got, _ := llama.render(nil, "x"); got != llama.Prefix+"x"+llama.Suffix {
		t.Fatalf("no history: %q", got)
	}
	if _, err := (&ChatTemplate{Prefix: "U: ", Suffix: "\nA: "}).render([]Turn{{"a", "b"}}, "c"); err == nil {
		t.Fatal("an unknown format took a history")
	}
}

// DeepSeek's history goes through its own chat encoder, as messages.
func TestDeepSeekChatHistory(t *testing.T) {
	c := newCluster(t)
	coord := c.add(false)
	model, _ := importTinyDeepSeek(t, coord.blobs.Store())
	cm, err := coord.llm.coordinator(t.Context(), model, nil)
	if err != nil {
		t.Fatal(err)
	}
	msg := func(role, text string) *dsv41.Value {
		v := dsv41.NewObject()
		v.Set("role", dsv41.NewString(role))
		v.Set("content", dsv41.NewString(text))
		return v
	}
	text, _, err := dsv41.EncodeChat([]*dsv41.Value{msg("user", "hi"), msg("assistant", "hello"), msg("user", "bye")}, dsv41.ChatOptions{ThinkingMode: "chat"})
	if err != nil {
		t.Fatal(err)
	}
	got, err := cm.prompt(GenerateRequest{Prompt: "bye", Chat: true, History: []Turn{{"hi", "hello"}}})
	if err != nil {
		t.Fatal(err)
	}
	if want := cm.ds.tok.Encode(text); !slices.Equal(got, want) {
		t.Fatalf("tokens %v, want %v", got, want)
	}
	if _, err := cm.prompt(GenerateRequest{Prompt: "bye", History: []Turn{{"hi", "hello"}}}); err == nil {
		t.Error("a history without chat was taken")
	}
}

// A toy vocabulary without byte tokens drops characters it cannot spell;
// it used to name tokens past its end, and the node's handler panicked.
func TestTokenizerSmallVocabulary(t *testing.T) {
	data, err := os.ReadFile("../../sdk/js/test/fixtures/tiny-tokenizer.bin")
	if err != nil {
		t.Skip(err)
	}
	tok, err := ReadTokenizer(data, 50)
	if err != nil {
		t.Fatal(err)
	}
	for _, text := range []string{"The little dog", "Привет, мир!", "a\x00b"} {
		for _, id := range tok.Encode(text, true) {
			if id < 0 || id >= tok.VocabSize() {
				t.Fatalf("%q: token %d outside a vocabulary of %d", text, id, tok.VocabSize())
			}
		}
	}
}
