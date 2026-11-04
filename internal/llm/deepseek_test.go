package llm

import (
	"compress/gzip"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"slices"
	"sync/atomic"
	"testing"

	"github.com/libp2p/go-libp2p/core/peer"

	"github.com/optimist3301/rakkan-mesh/internal/blob"
	"github.com/optimist3301/rakkan-mesh/internal/gpu"
	"github.com/optimist3301/rakkan-mesh/internal/llm/dsv41"
	"github.com/optimist3301/rakkan-mesh/internal/membership"
)

const dsFixture = "testdata/dsv41-tiny"

// tinyTokenizer is DeepSeek's tokenizer cut to the tiny model's 512
// tokens: the specials, the 256 bytes and the merges making the rest.
// Any text encodes within them, and the Engram tables come out as from
// the whole tokenizer, which they read only the first 512 tokens of.
func tinyTokenizer(t *testing.T) []byte {
	t.Helper()
	f, err := os.Open("testdata/dsv41-tokenizer.json.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	z, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	data, err := io.ReadAll(z)
	if err != nil {
		t.Fatal(err)
	}
	var tj map[string]json.RawMessage
	if err := json.Unmarshal(data, &tj); err != nil {
		t.Fatal(err)
	}
	var model map[string]json.RawMessage
	json.Unmarshal(tj["model"], &model)
	var vocab map[string]int
	var merges [][2]string
	var added []map[string]any
	json.Unmarshal(model["vocab"], &vocab)
	json.Unmarshal(model["merges"], &merges)
	json.Unmarshal(tj["added_tokens"], &added)
	const n = 512
	for tok, id := range vocab {
		if id >= n {
			delete(vocab, tok)
		}
	}
	merges = slices.DeleteFunc(merges, func(m [2]string) bool {
		_, ok := vocab[m[0]+m[1]]
		_, a := vocab[m[0]]
		_, b := vocab[m[1]]
		return !ok || !a || !b
	})
	added = slices.DeleteFunc(added, func(a map[string]any) bool { return a["id"].(float64) >= n })
	model["vocab"], _ = json.Marshal(vocab)
	model["merges"], _ = json.Marshal(merges)
	tj["model"], _ = json.Marshal(model)
	tj["added_tokens"], _ = json.Marshal(added)
	out, _ := json.Marshal(tj)
	return out
}

func importTinyDeepSeek(t *testing.T, store *blob.Store) (blob.ID, []byte) {
	t.Helper()
	cfg, err := os.ReadFile(filepath.Join(dsFixture, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(filepath.Join(dsFixture, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	fi, _ := f.Stat()
	tok := tinyTokenizer(t)
	id, m, err := ImportDeepSeek(store, HFSource{Config: cfg, Tokenizer: tok, Weights: []SizedReaderAt{sizedFile{f, fi.Size()}}}, "tiny")
	if err != nil {
		t.Fatal(err)
	}
	if m.Engram[1] == "" || m.Engram[0] != "" {
		t.Fatalf("Engram tables %v", m.Engram)
	}
	return id, tok
}

// localDeepSeek generates greedily with the whole model in one process.
func localDeepSeek(t *testing.T, tokData []byte, prompt string, maxTokens int) []int {
	t.Helper()
	cfg, err := dsv41.LoadConfig(filepath.Join(dsFixture, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	tok, err := ReadBPE(tokData, cfg.Vocab)
	if err != nil {
		t.Fatal(err)
	}
	tables, err := dsv41.BuildEngramTables(cfg, tok)
	if err != nil {
		t.Fatal(err)
	}
	tensors, err := dsv41.ReadSafetensors(filepath.Join(dsFixture, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	m, err := dsv41.New(cfg, tensors, tables)
	if err != nil {
		t.Fatal(err)
	}
	tokens := append([]int{cfg.BOS}, tok.Encode(prompt)...)
	st := m.NewState()
	logits := st.Prefill(tokens)
	sampler := NewSampler(0, 0)
	var out []int
	for len(out) < maxTokens {
		next := sampler.Sample(logits)
		if next == cfg.EOS {
			break
		}
		out = append(out, next)
		logits = st.Decode(next)
	}
	return out
}

const dsPrompt = "The mesh runs models"

// DeepSeek split over three nodes generates what one process does.
func TestDeepSeekOverThreeNodes(t *testing.T) {
	c := newCluster(t)
	coord := c.add(false)
	var nodes []*llmNode
	for range 3 {
		nodes = append(nodes, c.add(true))
	}
	model, tok := importTinyDeepSeek(t, coord.blobs.Store())
	waitServers(t, coord, 3)
	want := localDeepSeek(t, tok, dsPrompt, 20)
	got, stats := generate(t, coord, GenerateRequest{Model: model, Prompt: dsPrompt, MaxTokens: 20, Stages: 3}, nil)
	if !equalTokens(got, want) {
		t.Fatalf("tokens %v, one process %v", got, want)
	}
	if len(stats.Stages) != 3 {
		t.Fatalf("stages %+v", stats.Stages)
	}
	for i, st := range stats.Stages {
		if i > 0 && st.From != stats.Stages[i-1].To || st.Node == stats.Stages[(i+1)%3].Node {
			t.Fatalf("stages %+v", stats.Stages)
		}
	}
	t.Logf("%d prompt and %d new tokens over %+v", stats.PromptTokens, stats.Tokens, stats.Stages)
}

// A stage failing mid-generation moves, its caches rebuilt by replaying
// its inputs, the shared compressed caches included.
func TestDeepSeekSurvivesStageFailure(t *testing.T) {
	c := newCluster(t)
	coord := c.add(false)
	nodes := []*llmNode{c.add(true), c.add(true), c.add(true), c.add(true)}
	model, tok := importTinyDeepSeek(t, coord.blobs.Store())
	waitServers(t, coord, 4)
	want := localDeepSeek(t, tok, dsPrompt, 20)
	for _, n := range nodes {
		var forwards atomic.Int64
		n.llm.failNext = func(Shard) bool { return forwards.Add(1) == 6 }
	}
	got, stats := generate(t, coord, GenerateRequest{Model: model, Prompt: dsPrompt, MaxTokens: 20, Stages: 3}, nil)
	if !equalTokens(got, want) {
		t.Fatalf("after failover: tokens %v, uninterrupted %v", got, want)
	}
	if stats.Failovers == 0 {
		t.Fatalf("no failover in %d tokens", len(got))
	}
	t.Logf("%d failovers; final stages %+v", stats.Failovers, stats.Stages)
}

// A guest's stage is witnessed on the whole Payload: a flipped bit of the
// shared state, not only of the stream, is caught.
func TestDeepSeekGuestStageIsWitnessed(t *testing.T) {
	c := newCluster(t)
	var err error
	if c.open, err = membership.IssuePolicy(c.root, membership.Policy{Network: c.net.ID, Seq: 1, Open: true, PowBits: 8}); err != nil {
		t.Fatal(err)
	}
	coord := c.add(false)
	member := c.addNode(true, false, 1<<30)
	guest := c.addNode(true, true, 2<<30)
	model, tok := importTinyDeepSeek(t, coord.blobs.Store())
	waitServers(t, coord, 2)
	want := localDeepSeek(t, tok, dsPrompt, 12)
	var forwards atomic.Int64
	guest.llm.tamper = func(_ Shard, out []byte) {
		if forwards.Add(1) == 4 {
			out[len(out)-1] ^= 1 // the last byte: past the stream and the mix
		}
	}
	got, stats := generate(t, coord, GenerateRequest{Model: model, Prompt: dsPrompt, MaxTokens: 12, Stages: 1}, nil)
	if !equalTokens(got, want) {
		t.Fatalf("tokens %v, one process %v", got, want)
	}
	if st := stats.Stages[0]; stats.Caught != 1 || st.Node != member.node.ID() {
		t.Fatalf("stage %+v, caught %d", st, stats.Caught)
	}
	if !coord.trust.Suspect(guest.node.ID()) {
		t.Fatal("cheater not marked")
	}
}

// testDevice picks a hardware GPU, else a software one.
func testDevice(t *testing.T) *gpu.Device {
	t.Helper()
	var dev *gpu.Device
	for _, d := range gpuDevices(t) {
		if dev == nil || d.Info().Type != "cpu" && dev.Info().Type == "cpu" {
			dev = d
		}
	}
	return dev
}

// A node with a GPU runs its DeepSeek stage there, next to a CPU node's,
// and the text is one process's.
func TestDeepSeekOnGPUNode(t *testing.T) {
	dev := testDevice(t)
	c := newCluster(t)
	coord := c.add(false)
	fast := c.addOpts(nodeOpts{serve: true, dev: dev, devBytes: 1 << 30, mem: 1 << 30, speed: 50})
	slow := c.addOpts(nodeOpts{serve: true, mem: 1 << 30, speed: 5})
	model, tok := importTinyDeepSeek(t, coord.blobs.Store())
	waitServers(t, coord, 2)
	want := localDeepSeek(t, tok, dsPrompt, 20)
	got, stats := generate(t, coord, GenerateRequest{Model: model, Prompt: dsPrompt, MaxTokens: 20, Stages: 2}, nil)
	if !equalTokens(got, want) {
		t.Fatalf("tokens %v, one process %v", got, want)
	}
	t.Logf("%s: stages %+v", dev.Info().Name, stats.Stages)
	nodes := map[peer.ID]bool{}
	for _, st := range stats.Stages {
		nodes[st.Node] = true
	}
	if len(stats.Stages) != 2 || !nodes[fast.node.ID()] || !nodes[slow.node.ID()] {
		t.Fatalf("stages %+v: want one on each node", stats.Stages)
	}
	if fast.llm.cfg.Device.Used() == 0 {
		t.Fatal("the GPU node's layers are not on its GPU")
	}
	fast.llm.mu.Lock()
	for _, l := range fast.llm.shards {
		if l.dsGPU == nil {
			t.Error("a DeepSeek shard of the GPU node runs on its CPU")
		}
	}
	fast.llm.mu.Unlock()

	// The GPU node fails mid-generation: its stage moves to the CPU node,
	// whose replay of the stage's inputs rebuilds the same caches.
	var forwards atomic.Int64
	fast.llm.failNext = func(Shard) bool { return forwards.Add(1) == 6 }
	got, stats = generate(t, coord, GenerateRequest{Model: model, Prompt: dsPrompt, MaxTokens: 20, Stages: 2}, nil)
	if !equalTokens(got, want) {
		t.Fatalf("after the GPU node failed: tokens %v, one process %v", got, want)
	}
	if stats.Failovers == 0 || stats.Stages[0].Node != slow.node.ID() {
		t.Fatalf("stages %+v, %d failovers: want the GPU stage moved to the CPU node", stats.Stages, stats.Failovers)
	}
}

// A guest's GPU stage is witnessed by a member's CPU: the bits agree, so
// the guest earns trust and is caught at nothing.
func TestDeepSeekGPUGuestWitnessedByCPU(t *testing.T) {
	dev := testDevice(t)
	c := newCluster(t)
	var err error
	if c.open, err = membership.IssuePolicy(c.root, membership.Policy{Network: c.net.ID, Seq: 1, Open: true, PowBits: 8}); err != nil {
		t.Fatal(err)
	}
	coord := c.add(false)
	member := c.addNode(true, false, 1<<30)
	// The guest offers more memory, so the stage goes to it first.
	guest := c.addOpts(nodeOpts{serve: true, guest: true, dev: dev, devBytes: 2 << 30, mem: 2 << 30, speed: 1})
	model, tok := importTinyDeepSeek(t, coord.blobs.Store())
	waitServers(t, coord, 2)
	want := localDeepSeek(t, tok, dsPrompt, 12)
	got, stats := generate(t, coord, GenerateRequest{Model: model, Prompt: dsPrompt, MaxTokens: 12, Stages: 1}, nil)
	if !equalTokens(got, want) {
		t.Fatalf("tokens %v, one process %v", got, want)
	}
	st := stats.Stages[0]
	if st.Node != guest.node.ID() || st.Witness != member.node.ID() || stats.Caught != 0 {
		t.Fatalf("stage %+v, caught %d", st, stats.Caught)
	}
	if guest.llm.cfg.Device.Used() == 0 {
		t.Fatal("the guest's stage is not on its GPU")
	}
	if coord.trust.Score(guest.node.ID()) == 0 {
		t.Fatal("witnessed GPU stages earned the guest no trust")
	}
}

// A GPU too small for the layers whole takes them without their experts,
// which run on the node's CPU; the text is the same.
func TestDeepSeekHostExperts(t *testing.T) {
	dev := testDevice(t)
	cfg, err := dsv41.LoadConfig(filepath.Join(dsFixture, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var whole, experts int64
	for i := range cfg.Layers {
		whole += dsv41.LayerBytes(cfg, i)
		experts += dsv41.ExpertBytes(cfg, i)
	}
	// Room for every layer's dense part and a stage's arena, not for the
	// layers whole.
	devBytes := whole - experts/2
	c := newCluster(t)
	coord := c.add(false)
	node := c.addOpts(nodeOpts{serve: true, dev: dev, devBytes: devBytes, mem: 1 << 30, speed: 50})
	model, tok := importTinyDeepSeek(t, coord.blobs.Store())
	waitServers(t, coord, 1)
	want := localDeepSeek(t, tok, dsPrompt, 20)
	got, stats := generate(t, coord, GenerateRequest{Model: model, Prompt: dsPrompt, MaxTokens: 20}, nil)
	if !equalTokens(got, want) {
		t.Fatalf("tokens %v, one process %v", got, want)
	}
	t.Logf("stages %+v", stats.Stages)
	node.llm.mu.Lock()
	defer node.llm.mu.Unlock()
	for _, l := range node.llm.shards {
		if l.dsGPU == nil || !l.dsGPU.HostExperts() {
			t.Fatalf("shard on the GPU %v, host experts %v", l.dsGPU != nil, l.dsGPU != nil && l.dsGPU.HostExperts())
		}
		if l.hostBytes == 0 {
			t.Fatal("the experts' RAM is not accounted")
		}
	}
}

// With the draft, the text is the one without it, at any temperature,
// whatever the draft gets right: all of a block, part of it (the stages
// rewinding as deep as the rest), or none. Fewer passes when it is right.
func TestDeepSeekDraftSameText(t *testing.T) {
	c := newCluster(t)
	coord := c.add(false)
	nodes := []*llmNode{c.add(true), c.add(true)}
	model, _ := importTinyDeepSeek(t, coord.blobs.Store())
	waitServers(t, coord, 2)
	for _, temp := range []float32{0, 0.9} {
		req := GenerateRequest{Model: model, Prompt: dsPrompt, MaxTokens: 24, Stages: 2, Temperature: temp, Seed: 7}
		want, plain := generate(t, coord, req, nil)
		req.Draft = true
		got, stats := generate(t, coord, req, nil)
		if !equalTokens(got, want) {
			t.Fatalf("temperature %v: with the draft %v, without %v", temp, got, want)
		}
		t.Logf("temperature %v, the draft as it is: %d passes (%d without), %d of %d drafted taken", temp, stats.Passes, plain.Passes, stats.Accepted, stats.Drafted)
		// Drafts right up to a point that moves: 0..k tokens taken a pass.
		step := 0
		coord.llm.redraft = func(tokens, proposal []int) []int {
			n := len(tokens) - stats.PromptTokens // tokens generated so far
			right := step % (len(proposal) + 1)
			step++
			out := make([]int, len(proposal))
			for i := range out {
				if i < right && n+i < len(want) {
					out[i] = want[n+i]
				} else {
					out[i] = (proposal[i] + 1) % 512
				}
			}
			return out
		}
		got, stats = generate(t, coord, req, nil)
		coord.llm.redraft = nil
		if !equalTokens(got, want) {
			t.Fatalf("temperature %v, drafts right in part: %v, without the draft %v", temp, got, want)
		}
		if stats.Accepted == 0 || stats.Passes >= plain.Passes {
			t.Fatalf("temperature %v: %d drafted taken in %d passes (%d without)", temp, stats.Accepted, stats.Passes, plain.Passes)
		}
		t.Logf("temperature %v, drafts right in part: %d passes (%d without), %d of %d drafted taken", temp, stats.Passes, plain.Passes, stats.Accepted, stats.Drafted)
	}
	_ = nodes
}

// Drafting over a GPU node and a CPU node, a stage failing mid-generation
// (its replacement replays only the positions kept), and a guest
// witnessed (the witness rewinding with it): the same text still.
func TestDeepSeekDraftFailoverAndWitness(t *testing.T) {
	dev := testDevice(t)
	c := newCluster(t)
	var err error
	if c.open, err = membership.IssuePolicy(c.root, membership.Policy{Network: c.net.ID, Seq: 1, Open: true, PowBits: 8}); err != nil {
		t.Fatal(err)
	}
	coord := c.add(false)
	gpuNode := c.addOpts(nodeOpts{serve: true, dev: dev, devBytes: 1 << 30, mem: 1 << 30, speed: 50})
	cpu := c.addOpts(nodeOpts{serve: true, mem: 1 << 30, speed: 5})
	spare := c.addOpts(nodeOpts{serve: true, mem: 1 << 30, speed: 1})
	model, _ := importTinyDeepSeek(t, coord.blobs.Store())
	waitServers(t, coord, 3)
	req := GenerateRequest{Model: model, Prompt: dsPrompt, MaxTokens: 24, Stages: 2, Temperature: 0.7, Seed: 3}
	want, _ := generate(t, coord, req, nil)
	step := 0
	coord.llm.redraft = func(tokens, proposal []int) []int {
		n := len(tokens) - 21
		right := step % (len(proposal) + 1)
		step++
		out := slices.Clone(proposal)
		for i := range right {
			if n+i < len(want) {
				out[i] = want[n+i]
			}
		}
		return out
	}
	var forwards atomic.Int64
	gpuNode.llm.failNext = func(Shard) bool { return forwards.Add(1) == 5 }
	req.Draft = true
	got, stats := generate(t, coord, req, nil)
	if !equalTokens(got, want) {
		t.Fatalf("with the draft and a failover: %v, without %v", got, want)
	}
	if stats.Failovers == 0 || stats.Accepted == 0 {
		t.Fatalf("%d failovers, %d drafted taken", stats.Failovers, stats.Accepted)
	}
	t.Logf("%d passes, %d taken of %d, %d failovers, stages %+v", stats.Passes, stats.Accepted, stats.Drafted, stats.Failovers, stats.Stages)
	_, _ = cpu, spare

	// A guest's stage, witnessed, through drafts taken and rejected.
	// The fastest, so the stage goes to it.
	guest := c.addOpts(nodeOpts{serve: true, guest: true, mem: 4 << 30, speed: 100})
	waitServers(t, coord, 4)
	step = 0
	got, stats = generate(t, coord, GenerateRequest{Model: model, Prompt: dsPrompt, MaxTokens: 24, Stages: 1, Temperature: 0.7, Seed: 3, Draft: true}, nil)
	if !equalTokens(got, want) {
		t.Fatalf("witnessed: %v, without the draft %v", got, want)
	}
	if st := stats.Stages[0]; st.Node != guest.node.ID() || st.Witness == "" || stats.Caught != 0 {
		t.Fatalf("stage %+v, caught %d", st, stats.Caught)
	}
}

// fixtureImage is the PNG of the fixture's image prompt.
func fixtureImage(t *testing.T) []byte {
	t.Helper()
	g, err := dsv41.ReadSafetensors(filepath.Join(dsFixture, "golden.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	return g["vl.png"].Data
}

// localDeepSeekImage generates greedily in one process from text, an
// image's span, then more text.
func localDeepSeekImage(t *testing.T, tokData []byte, before string, png []byte, after string, maxTokens int) []int {
	t.Helper()
	cfg, err := dsv41.LoadConfig(filepath.Join(dsFixture, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	tok, err := ReadBPE(tokData, cfg.Vocab)
	if err != nil {
		t.Fatal(err)
	}
	tables, err := dsv41.BuildEngramTables(cfg, tok)
	if err != nil {
		t.Fatal(err)
	}
	tensors, err := dsv41.ReadSafetensors(filepath.Join(dsFixture, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	m, err := dsv41.New(cfg, tensors, tables)
	if err != nil {
		t.Fatal(err)
	}
	vis, err := dsv41.NewVision(cfg, tensors)
	if err != nil {
		t.Fatal(err)
	}
	img, err := cfg.PrepareImage(png)
	if err != nil {
		t.Fatal(err)
	}
	emb, err := vis.Encode(img)
	if err != nil {
		t.Fatal(err)
	}
	span, err := vis.Span(img.ImageGrid, emb)
	if err != nil {
		t.Fatal(err)
	}
	tokens := append([]int{cfg.BOS}, tok.Encode(before)...)
	rows := make([][]float32, len(tokens))
	for _, r := range span {
		tokens = append(tokens, dsv41.ImagePosition)
		rows = append(rows, r)
	}
	for _, x := range tok.Encode(after) {
		tokens = append(tokens, x)
		rows = append(rows, nil)
	}
	st := m.NewState()
	logits := st.PrefillImages(tokens, rows)
	sampler := NewSampler(0, 0)
	var out []int
	for len(out) < maxTokens {
		next := sampler.Sample(logits)
		if next == cfg.EOS {
			break
		}
		out = append(out, next)
		logits = st.Decode(next)
	}
	return out
}

// A prompt with an image over two nodes, the draft proposing or not, gives
// what one process does; the coordinator encodes the image.
func TestDeepSeekImageOverNodes(t *testing.T) {
	c := newCluster(t)
	coord := c.add(false)
	for range 2 {
		c.add(true)
	}
	model, tok := importTinyDeepSeek(t, coord.blobs.Store())
	waitServers(t, coord, 2)
	png := fixtureImage(t)
	before, after := "Look at", " it"
	want := localDeepSeekImage(t, tok, before, png, after, 12)
	for _, draft := range []bool{false, true} {
		got, stats := generate(t, coord, GenerateRequest{
			Model: model, Prompt: before + dsv41.ImagePlaceholder + after, MaxTokens: 12, Stages: 2,
			Images: [][]byte{png}, Draft: draft,
		}, nil)
		if !equalTokens(got, want) {
			t.Fatalf("draft %v: tokens %v, one process %v", draft, got, want)
		}
		t.Logf("draft %v: %d prompt tokens (an image's span among them), %d new, %d passes", draft, stats.PromptTokens, stats.Tokens, stats.Passes)
	}
	// The places and the images must agree.
	if _, err := coord.llm.Generate(t.Context(), GenerateRequest{Model: model, Prompt: before, Images: [][]byte{png}}, func(Event) error { return nil }); err == nil {
		t.Error("an image without a place in the prompt was taken")
	}
}

// In the chat format, the images come before the message's text, each
// place the config's image token.
func TestDeepSeekChatImagePlaces(t *testing.T) {
	c := newCluster(t)
	coord := c.add(false)
	model, _ := importTinyDeepSeek(t, coord.blobs.Store())
	cm, err := coord.llm.coordinator(t.Context(), model, nil)
	if err != nil {
		t.Fatal(err)
	}
	cfg := cm.ds.head.Cfg
	tokens, err := cm.prompt(GenerateRequest{Prompt: "what is this", Chat: true, Images: make([][]byte, 2)})
	if err != nil {
		t.Fatal(err)
	}
	var places []int
	for i, x := range tokens {
		if x == cfg.ImageToken {
			places = append(places, i)
		}
	}
	text := cm.ds.tok.Encode("what is this")
	at := slices.Index(tokens, text[0])
	if len(places) != 2 || at < places[1] {
		t.Fatalf("image places %v, the text from %d, in %v", places, at, tokens)
	}
	if _, err := cm.prompt(GenerateRequest{Prompt: "a " + dsv41.ImagePlaceholder, Chat: true}); err == nil {
		t.Error("the placeholder in a chat message's text was taken")
	}
}

// A coordinator with a GPU encodes images on it: the same text.
func TestDeepSeekImageOnGPUCoordinator(t *testing.T) {
	dev := testDevice(t)
	c := newCluster(t)
	coord := c.addOpts(nodeOpts{dev: dev, devBytes: 1 << 30, mem: 1 << 30})
	c.add(true)
	model, tok := importTinyDeepSeek(t, coord.blobs.Store())
	waitServers(t, coord, 1)
	png := fixtureImage(t)
	before, after := "Look at", " it"
	want := localDeepSeekImage(t, tok, before, png, after, 12)
	got, _ := generate(t, coord, GenerateRequest{Model: model, Prompt: before + dsv41.ImagePlaceholder + after, MaxTokens: 12, Images: [][]byte{png}}, nil)
	if !equalTokens(got, want) {
		t.Fatalf("tokens %v, one process %v", got, want)
	}
	cm, err := coord.llm.coordinator(t.Context(), model, nil)
	if err != nil {
		t.Fatal(err)
	}
	if cm.ds.visionGPU == nil {
		t.Fatal("the image encoder is not on the coordinator's GPU")
	}
}
