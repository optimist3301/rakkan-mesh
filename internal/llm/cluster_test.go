package llm

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"math"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/peer"

	"github.com/optimist3301/rakkan-mesh/internal/blob"
	"github.com/optimist3301/rakkan-mesh/internal/gpu"
	"github.com/optimist3301/rakkan-mesh/internal/membership"
	"github.com/optimist3301/rakkan-mesh/internal/mesh"
	"github.com/optimist3301/rakkan-mesh/internal/trust"
)

// tokenizerBytes builds a tokenizer in llama2.c's format.
func tokenizerBytes(vocab []string) []byte {
	b := binary.LittleEndian.AppendUint32(nil, 16)
	for i, s := range vocab {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(float32(-i)))
		b = binary.LittleEndian.AppendUint32(b, uint32(len(s)))
		b = append(b, s...)
	}
	return b
}

// syntheticTokenizer covers the synthetic model's 50 tokens: specials,
// then single characters.
func syntheticTokenizer() []byte {
	vocab := []string{"<unk>", "<s>", "</s>"}
	for _, r := range " abcdefghijklmnopqrstuvwxyz.,!?ABCDEFGHIJKLMNO" {
		vocab = append(vocab, string(r))
	}
	return tokenizerBytes(vocab[:50])
}

type llmNode struct {
	node  *mesh.Node
	blobs *blob.Service
	llm   *Service
	trust *trust.Service
}

type cluster struct {
	t    *testing.T
	root crypto.PrivKey
	net  *membership.Network
	base time.Time
	boot []peer.AddrInfo
	// open admits guests and gives every node a trust service.
	open *membership.SignedPolicy
}

func newCluster(t *testing.T) *cluster {
	root, n, err := membership.NewRootKey()
	if err != nil {
		t.Fatal(err)
	}
	return &cluster{t: t, root: root, net: n, base: time.Now()}
}

func (c *cluster) add(serve bool) *llmNode { return c.addNode(serve, false, 0) }

// addNode starts a member, or a guest proving work, serving stages with mem
// bytes (0: the default) if serve is set.
func (c *cluster) addNode(serve, guest bool, mem int64) *llmNode {
	return c.addOpts(nodeOpts{serve: serve, guest: guest, mem: mem, speed: 1})
}

// nodeOpts describes a test node: serving stages or not, member or guest,
// its memory for stages, its GPU, and the speed it advertises.
type nodeOpts struct {
	serve, guest bool
	mem          int64
	dev          *gpu.Device
	devBytes     int64
	speed        float64
}

func (c *cluster) addOpts(o nodeOpts) *llmNode {
	serve, guest, mem := o.serve, o.guest, o.mem
	t := c.t
	sk, _, _ := crypto.GenerateEd25519Key(rand.Reader)
	id, _ := peer.IDFromPrivateKey(sk)
	cfg := mesh.Config{Key: sk, Network: c.net, Bootstrap: c.boot, Policy: c.open,
		ListenAddrs: []string{"/ip4/127.0.0.1/udp/0/quic-v1"}, AuthTimeout: 2 * time.Second,
		HealthInterval: 300 * time.Millisecond, DiscoveryInterval: 500 * time.Millisecond, MaxBackoff: time.Second}
	if guest {
		pr, err := c.net.Solve(t.Context(), id, 8)
		if err != nil {
			t.Fatal(err)
		}
		cfg.Proof = &pr
	} else {
		cfg.Cert, _ = membership.Issue(c.root, nil, membership.Cert{Network: c.net.ID, Subject: id, Roles: []string{membership.RoleMember},
			NotBefore: c.base.Add(-time.Minute), NotAfter: c.base.Add(time.Hour)})
	}
	node, err := mesh.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	store, err := blob.OpenStore(t.TempDir(), 4<<30)
	if err != nil {
		t.Fatal(err)
	}
	n := &llmNode{node: node, blobs: blob.NewService(node, store, nil)}
	if c.open != nil {
		if n.trust, err = trust.New(trust.Config{Node: node}); err != nil {
			t.Fatal(err)
		}
	}
	if n.llm, err = NewService(ServiceConfig{Node: node, Blobs: n.blobs, Serve: serve, MemoryBytes: mem, Trust: n.trust,
		Device: o.dev, DeviceBytes: o.devBytes, Speed: o.speed}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { n.close() })
	if len(c.boot) == 0 {
		c.boot = []peer.AddrInfo{node.AddrInfo()}
	}
	return n
}

func (n *llmNode) close() {
	n.llm.Close()
	if n.trust != nil {
		n.trust.Close()
	}
	n.node.Close()
}

// waitServers waits until the coordinator sees k members serving stages.
func waitServers(t *testing.T, n *llmNode, k int) {
	t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	for len(n.llm.servers()) < k {
		if time.Now().After(deadline) {
			t.Fatalf("saw %d stage servers, want %d", len(n.llm.servers()), k)
		}
		time.Sleep(50 * time.Millisecond)
	}
}

func generate(t *testing.T, n *llmNode, req GenerateRequest, onToken func(int)) ([]int, *Stats) {
	t.Helper()
	var out []int
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	stats, err := n.llm.Generate(ctx, req, func(e Event) error {
		out = append(out, e.Token)
		if onToken != nil {
			onToken(len(out))
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return out, stats
}

// localTokens generates with the whole model in one process.
func localTokens(ck *Checkpoint, tok *Tokenizer, prompt string, maxTokens int) []int {
	c := ck.Config
	stage := NewStage(c, ck.Layers, NewRope(c, c.SeqLen), c.SeqLen)
	tokens := tok.Encode(prompt, true)
	sampler := NewSampler(0, 0)
	var out []int
	for pos := 0; len(out) < maxTokens && pos < c.SeqLen; pos++ {
		x := Embed(c, ck.Embed, tokens[pos])
		stage.Forward(x, pos)
		if pos+1 < len(tokens) {
			continue
		}
		next := sampler.Sample(ck.Head.Logits(c, x))
		if next == TokenBOS || next == TokenEOS {
			break
		}
		out = append(out, next)
		tokens = append(tokens, next)
	}
	return out
}

// talkative returns a synthetic model that generates n tokens from prompt
// without stopping, so failures can be injected mid-generation.
func talkative(t *testing.T, prompt string, n int) (*Checkpoint, []byte, *Tokenizer) {
	t.Helper()
	tokData := syntheticTokenizer()
	for seed := uint64(100); seed < 200; seed++ {
		ck := synthetic(seed)
		tok, _ := ReadTokenizer(tokData, ck.Config.Vocab)
		if len(localTokens(ck, tok, prompt, n)) == n {
			return ck, tokData, tok
		}
	}
	t.Fatal("no synthetic model generates long enough")
	return nil, nil, nil
}

func equalTokens(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestPipelineMatchesLocalGeneration(t *testing.T) {
	c := newCluster(t)
	coord := c.add(false) // holds the model's blobs, runs no layers
	s1, s2 := c.add(true), c.add(true)
	ck := synthetic(5)
	tokData := syntheticTokenizer()
	model, _, err := Import(coord.blobs.Store(), ck, tokData, "synthetic")
	if err != nil {
		t.Fatal(err)
	}
	tok, _ := ReadTokenizer(tokData, ck.Config.Vocab)
	waitServers(t, coord, 2)

	want := localTokens(ck, tok, "hello", 15)
	got, stats := generate(t, coord, GenerateRequest{Model: model, Prompt: "hello", MaxTokens: 15, Stages: 2}, nil)
	if !equalTokens(got, want) {
		t.Fatalf("pipeline tokens %v, local %v", got, want)
	}
	if len(stats.Stages) != 2 || stats.Stages[0].Node == stats.Stages[1].Node {
		t.Fatalf("stages %+v", stats.Stages)
	}
	for _, st := range stats.Stages {
		if st.Node == coord.node.ID() {
			t.Fatal("a stage ran on the coordinator, which does not serve")
		}
	}
	_ = s1
	_ = s2
}

func TestPipelineSurvivesStageFailure(t *testing.T) {
	c := newCluster(t)
	coord := c.add(false)
	nodes := []*llmNode{c.add(true), c.add(true), c.add(true)}
	ck, tokData, tok := talkative(t, "abc", 18)
	model, _, _ := Import(coord.blobs.Store(), ck, tokData, "synthetic")
	waitServers(t, coord, 3)
	want := localTokens(ck, tok, "abc", 18)

	// Every serving node crashes its stage on the 6th forward it handles.
	for _, n := range nodes {
		var forwards atomic.Int64
		n.llm.failNext = func(Shard) bool { return forwards.Add(1) == 6 }
	}
	got, stats := generate(t, coord, GenerateRequest{Model: model, Prompt: "abc", MaxTokens: 18, Stages: 2}, nil)
	if !equalTokens(got, want) {
		t.Fatalf("after failover: tokens %v, uninterrupted %v", got, want)
	}
	if stats.Failovers == 0 {
		t.Fatalf("no failover happened in %d tokens", len(got))
	}
	t.Logf("%d failovers; final stages %+v", stats.Failovers, stats.Stages)
}

func TestPipelineSurvivesNodeDeath(t *testing.T) {
	c := newCluster(t)
	coord := c.add(false)
	nodes := map[peer.ID]*llmNode{}
	for range 3 {
		n := c.add(true)
		nodes[n.node.ID()] = n
	}
	ck, tokData, tok := talkative(t, "dog", 18)
	model, _, _ := Import(coord.blobs.Store(), ck, tokData, "synthetic")
	waitServers(t, coord, 3)
	want := localTokens(ck, tok, "dog", 18)

	// Kill the node running the first stage after 5 tokens.
	var killed peer.ID
	got, stats := generate(t, coord, GenerateRequest{Model: model, Prompt: "dog", MaxTokens: 18, Stages: 2}, func(n int) {
		if n != 5 {
			return
		}
		for id, node := range nodes {
			if node.llm.advert().FreeMem < node.llm.cfg.MemoryBytes && killed == "" {
				// Busy with a stage: kill it.
				killed = id
				node.close()
			}
		}
	})
	if killed == "" {
		t.Fatal("no stage node found to kill")
	}
	if !equalTokens(got, want) {
		t.Fatalf("after node death: tokens %v, uninterrupted %v", got, want)
	}
	for _, st := range stats.Stages {
		if st.Node == killed {
			t.Fatal("dead node still listed as a stage")
		}
	}
	if stats.Failovers == 0 {
		t.Fatal("no failover recorded")
	}
}

func TestRealModelOverThreeNodes(t *testing.T) {
	ck, tok := realModel(t)
	c := newCluster(t)
	coord := c.add(false)
	for range 3 {
		c.add(true)
	}
	tokData := tokenizerBytesFromDir(t)
	model, _, err := Import(coord.blobs.Store(), ck, tokData, "stories15M")
	if err != nil {
		t.Fatal(err)
	}
	waitServers(t, coord, 3)
	want := localTokens(ck, tok, "Once upon a time", 60)
	var text strings.Builder
	prev := 0
	got, stats := generate(t, coord, GenerateRequest{Model: model, Prompt: "Once upon a time", MaxTokens: 60, Stages: 3}, nil)
	for _, id := range got {
		text.WriteString(tok.Decode(prev, id))
		prev = id
	}
	if !equalTokens(got, want) {
		t.Fatalf("distributed text differs from local:\n%q", text.String())
	}
	t.Logf("%d tokens over %d stages in %s: %q", stats.Tokens, len(stats.Stages), stats.Elapsed.Round(time.Millisecond), text.String())
}

func TestGuestStageIsWitnessed(t *testing.T) {
	c := newCluster(t)
	var err error
	if c.open, err = membership.IssuePolicy(c.root, membership.Policy{Network: c.net.ID, Seq: 1, Open: true, PowBits: 8}); err != nil {
		t.Fatal(err)
	}
	coord := c.add(false)
	member := c.addNode(true, false, 1<<30)
	// The guest offers more memory, so the stage goes to it first.
	guest := c.addNode(true, true, 2<<30)
	ck, tokData, tok := talkative(t, "abc", 18)
	model, _, _ := Import(coord.blobs.Store(), ck, tokData, "synthetic")
	waitServers(t, coord, 2)
	want := localTokens(ck, tok, "abc", 18)
	// Once cheating, the guest corrupts one activation on its 6th forward.
	var cheating atomic.Bool
	var forwards atomic.Int64
	guest.llm.tamper = func(_ Shard, out []byte) {
		if cheating.Load() && forwards.Add(1) == 6 {
			out[0] ^= 1 // the lowest bit of the first activation
		}
	}

	got, stats := generate(t, coord, GenerateRequest{Model: model, Prompt: "abc", MaxTokens: 18, Stages: 1}, nil)
	if !equalTokens(got, want) {
		t.Fatalf("tokens %v, local %v", got, want)
	}
	if st := stats.Stages[0]; st.Node != guest.node.ID() || st.Witness != member.node.ID() || stats.Caught != 0 {
		t.Fatalf("stage %+v, caught %d", st, stats.Caught)
	}
	if coord.trust.Score(guest.node.ID()) == 0 {
		t.Fatal("witnessed stages earned the guest no trust")
	}

	// The witness catches the corruption and takes over, and the text is
	// unaffected.
	cheating.Store(true)
	got, stats = generate(t, coord, GenerateRequest{Model: model, Prompt: "abc", MaxTokens: 18, Stages: 1}, nil)
	if !equalTokens(got, want) {
		t.Fatalf("after cheating: tokens %v, local %v", got, want)
	}
	if st := stats.Stages[0]; stats.Caught != 1 || st.Node != member.node.ID() {
		t.Fatalf("stage %+v, caught %d", st, stats.Caught)
	}
	if !coord.trust.Suspect(guest.node.ID()) {
		t.Fatal("cheater not marked")
	}
	deadline := time.Now().Add(5 * time.Second)
	for coord.node.IsMember(guest.node.ID()) {
		if time.Now().After(deadline) {
			t.Fatal("cheater still connected")
		}
		time.Sleep(20 * time.Millisecond)
	}
	// And gets no more stages.
	_, stats = generate(t, coord, GenerateRequest{Model: model, Prompt: "abc", MaxTokens: 5, Stages: 1}, nil)
	if stats.Stages[0].Node == guest.node.ID() {
		t.Fatal("cheater given a stage again")
	}
}
