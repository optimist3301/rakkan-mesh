package llm

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/blob"
)

// Hugging Face models for these tests live in RAKKAN_HF_MODELS, one
// directory per model with its config.json, tokenizer files and
// safetensors, as downloaded from the Hub.
var hfModels = []string{"qwen2.5-0.5b-instruct", "qwen3-0.6b"}

type sizedFile struct {
	*os.File
	size int64
}

func (f sizedFile) Size() int64 { return f.size }

func openHF(t testing.TB, dir string) HFSource {
	t.Helper()
	read := func(name string, required bool) []byte {
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil && required {
			t.Fatal(err)
		}
		return b
	}
	src := HFSource{Config: read("config.json", true), GenerationConfig: read("generation_config.json", false),
		TokenizerConfig: read("tokenizer_config.json", false), Tokenizer: read("tokenizer.json", true)}
	paths, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	for _, p := range paths {
		f, err := os.Open(p)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { f.Close() })
		st, _ := f.Stat()
		src.Weights = append(src.Weights, sizedFile{f, st.Size()})
	}
	return src
}

// localModel is a stored model loaded whole into this process.
type localModel struct {
	m      *Manifest
	embed  Matrix
	head   *Head
	layers []*Layer
	tok    Codec
}

func loadStored(t testing.TB, store *blob.Store, id blob.ID) *localModel {
	t.Helper()
	read := func(s string) []byte {
		bid, _ := blob.ParseID(s)
		b, err := store.ReadAll(bid, 1<<34)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	m, err := ParseManifest(read(id.String()))
	if err != nil {
		t.Fatal(err)
	}
	lm := &localModel{m: m}
	if lm.embed, err = LoadEmbed(m.Config, read(m.Embed)); err != nil {
		t.Fatal(err)
	}
	if lm.head, err = LoadHead(m.Config, read(m.Head), lm.embed); err != nil {
		t.Fatal(err)
	}
	for _, l := range m.Layers {
		layer, err := LoadLayer(m.Config, read(l))
		if err != nil {
			t.Fatal(err)
		}
		lm.layers = append(lm.layers, layer)
	}
	if lm.tok, err = readCodec(m, read(m.Tokenizer)); err != nil {
		t.Fatal(err)
	}
	return lm
}

// logits runs tokens through the model and returns every position's.
func (lm *localModel) logits(tokens []int) [][]float32 {
	c := lm.m.Config
	stage := NewStage(c, lm.layers, NewRope(c, len(tokens)), len(tokens))
	var out [][]float32
	for pos, tok := range tokens {
		x := Embed(c, lm.embed, tok)
		stage.Forward(x, pos)
		out = append(out, lm.head.Logits(c, x))
	}
	return out
}

// chat generates greedily from a chat prompt, returning the text and the
// speed.
func (lm *localModel) chat(prompt string, maxTokens int) (string, float64) {
	text, _, speed := lm.chatTokens(prompt, maxTokens)
	return text, speed
}

func (lm *localModel) chatTokens(prompt string, maxTokens int) (string, []int, float64) {
	c := lm.m.Config
	tokens := lm.tok.Encode(lm.m.Chat.Prefix + prompt + lm.m.Chat.Suffix)
	n := len(tokens) + maxTokens
	stage := NewStage(c, lm.layers, NewRope(c, n), n)
	var text []byte
	var x []float32
	var out []int
	start := time.Now()
	for pos := 0; pos < n; pos++ {
		x = Embed(c, lm.embed, tokens[pos])
		stage.Forward(x, pos)
		if pos+1 < len(tokens) {
			continue
		}
		next := argmax(lm.head.Logits(c, x))
		if slices.Contains(lm.m.stop(), next) || pos+1 == n {
			break
		}
		text = append(text, lm.tok.Piece(tokens[pos], next)...)
		tokens = append(tokens, next)
		out = append(out, next)
	}
	return string(text), out, float64(len(tokens)) / time.Since(start).Seconds()
}

func TestHFModels(t *testing.T) {
	root := os.Getenv("RAKKAN_HF_MODELS")
	if root == "" {
		t.Skip("RAKKAN_HF_MODELS not set")
	}
	for _, name := range hfModels {
		dir := filepath.Join(root, name)
		if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
			continue
		}
		t.Run(name, func(t *testing.T) {
			store, err := blob.OpenStore(t.TempDir(), 1<<34)
			if err != nil {
				t.Fatal(err)
			}
			start := time.Now()
			id, m, err := ImportHF(store, openHF(t, dir), name)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("imported in %s: %+v, stop %v, chat %v", time.Since(start).Round(time.Millisecond), m.Config, m.Stop, m.Chat != nil)
			lm := loadStored(t, store, id)

			// Special tokens are recognized in text; text round-trips.
			tokens := lm.tok.Encode("<|im_start|>user\nНу привет, 世界! 1234<|im_end|>")
			if tokens[0] != 151644 || tokens[len(tokens)-1] != 151645 {
				t.Fatalf("special tokens not encoded as such: %v", tokens)
			}
			var back []byte
			for i, tok := range tokens {
				back = append(back, lm.tok.Piece(max(i-1, 0), tok)...)
			}
			if string(back) != "<|im_start|>user\nНу привет, 世界! 1234<|im_end|>" {
				t.Fatalf("round trip: %q", back)
			}

			// Logits agree with the independent numpy implementation.
			prompt := lm.tok.Encode(lm.m.Chat.Prefix + "What is the capital of France?" + lm.m.Chat.Suffix)
			compareWithReference(t, dir, lm, prompt)

			text, speed := lm.chat("What is the capital of France? Answer in one word.", 12)
			t.Logf("answer %q (%.1f tokens/s)", text, speed)
			if !strings.Contains(text, "Paris") {
				t.Fatalf("answer %q does not name Paris", text)
			}
			text, _ = lm.chat("Write one sentence about the sea.", 40)
			t.Logf("sentence %q", text)
		})
	}
}

// compareWithReference checks the engine's logits at every position of
// tokens against tools/llmref/hf_reference.py.
func compareWithReference(t *testing.T, dir string, lm *localModel, tokens []int) {
	t.Helper()
	script, _ := filepath.Abs("../../tools/llmref/hf_reference.py")
	if _, err := exec.LookPath("python3"); err != nil || exec.Command("python3", "-c", "import numpy").Run() != nil {
		t.Log("python3 with numpy not available: skipping the reference comparison")
		return
	}
	var ids []string
	for _, tok := range tokens {
		ids = append(ids, strconv.Itoa(tok))
	}
	out := filepath.Join(t.TempDir(), "logits.f32")
	cmd := exec.Command("python3", "-I", script, dir, strings.Join(ids, ","), out)
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("reference: %v", err)
	}
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	vocab := lm.m.Config.Vocab
	if len(raw) != 4*vocab*len(tokens) {
		t.Fatalf("reference wrote %d bytes", len(raw))
	}
	got := lm.logits(tokens)
	var worst, scale float64
	agree := 0
	for p := range tokens {
		ref := make([]float32, vocab)
		for i := range ref {
			ref[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*(p*vocab+i):]))
		}
		for i := range ref {
			worst = max(worst, math.Abs(float64(got[p][i]-ref[i])))
			scale = max(scale, math.Abs(float64(ref[i])))
		}
		if argmax(got[p]) == argmax(ref) {
			agree++
		}
	}
	t.Logf("logits vs reference over %d positions: max difference %.2e (largest logit %.1f), top token agrees at %d", len(tokens), worst, scale, agree)
	if worst > 1e-3*scale || agree < len(tokens)-1 {
		t.Fatalf("engine and reference disagree: max difference %.3g, top token agrees at %d of %d", worst, agree, len(tokens))
	}
}

// TestHFModelOverCluster runs a Qwen model split over three nodes and
// checks the text is the one a single process generates.
func TestHFModelOverCluster(t *testing.T) {
	root := os.Getenv("RAKKAN_HF_MODELS")
	if root == "" {
		t.Skip("RAKKAN_HF_MODELS not set")
	}
	dir := filepath.Join(root, hfModels[0])
	if _, err := os.Stat(dir); err != nil {
		t.Skip(err)
	}
	c := newCluster(t)
	coord := c.add(false)
	for range 3 {
		c.add(true)
	}
	model, _, err := ImportHF(coord.blobs.Store(), openHF(t, dir), "qwen")
	if err != nil {
		t.Fatal(err)
	}
	const question = "Name three colors of the rainbow."
	want, wantTokens, _ := loadStored(t, coord.blobs.Store(), model).chatTokens(question, 30)
	waitServers(t, coord, 3)

	var text strings.Builder
	var got []int
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Minute)
	defer cancel()
	stats, err := coord.llm.Generate(ctx, GenerateRequest{Model: model, Prompt: question, Chat: true, MaxTokens: 30, Stages: 3}, func(e Event) error {
		got = append(got, e.Token)
		text.WriteString(e.Text)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, wantTokens) || text.String() != want {
		t.Fatalf("distributed %q, local %q", text.String(), want)
	}
	t.Logf("%d tokens over %d stages in %s, loading the layers included: %q", stats.Tokens, len(stats.Stages), stats.Elapsed.Round(time.Millisecond), text.String())
	// Again, with the layers loaded.
	if stats, err = coord.llm.Generate(ctx, GenerateRequest{Model: model, Prompt: question, Chat: true, MaxTokens: 30, Stages: 3}, func(Event) error { return nil }); err != nil {
		t.Fatal(err)
	}
	t.Logf("loaded: %d tokens in %s, %.1f tokens/s", stats.Tokens, stats.Elapsed.Round(time.Millisecond), float64(stats.Tokens)/stats.Elapsed.Seconds())
}
