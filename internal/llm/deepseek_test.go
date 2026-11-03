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
	guest := c.addOpts(nodeOpts{serve: true, guest: true, dev: dev, devBytes: 1 << 30, mem: 2 << 30, speed: 1})
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
