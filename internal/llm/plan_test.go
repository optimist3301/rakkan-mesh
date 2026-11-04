package llm

import (
	"errors"
	"testing"
)

func TestPlanStages(t *testing.T) {
	gpuBig := stageNode{id: "gpu-big", speed: 100, layers: 20}
	gpuSmall := stageNode{id: "gpu-small", speed: 60, layers: 6}
	cpu := stageNode{id: "cpu", speed: 10, layers: 40}
	tiny := stageNode{id: "tiny", speed: 30, layers: 0} // too small for a layer
	nodes := []stageNode{cpu, tiny, gpuSmall, gpuBig}

	// Automatic: the fastest take what they can hold, the CPU the rest.
	got, err := planStages(uniform(32), nodes, 0)
	if err != nil {
		t.Fatal(err)
	}
	want := []span{{0, 20, "gpu-big"}, {20, 26, "gpu-small"}, {26, 32, "cpu"}}
	if !equalSpans(got, want) {
		t.Fatalf("automatic plan %v, want %v", got, want)
	}
	// A model the big GPU holds alone needs no other node.
	if got, _ := planStages(uniform(16), nodes, 0); !equalSpans(got, []span{{0, 16, "gpu-big"}}) {
		t.Fatalf("small model: %v", got)
	}
	// Too big for all of them together.
	if _, err := planStages(uniform(100), nodes, 0); !errors.Is(err, ErrNoStages) {
		t.Fatalf("oversized model: %v", err)
	}

	// Two stages: the two fastest, split by speed within what they hold.
	got, err = planStages(uniform(16), nodes, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !equalSpans(got, []span{{0, 10, "gpu-big"}, {10, 16, "gpu-small"}}) {
		t.Fatalf("two stages: %v", got)
	}
	// Every stage asked for gets a layer, however slow its node.
	slow := stageNode{id: "slow", speed: 0.01, layers: 10}
	if got, _ := planStages(uniform(4), []stageNode{gpuBig, slow}, 2); len(got) != 2 || got[1].node != "slow" || got[1].to-got[1].from != 1 {
		t.Fatalf("slow node in two stages: %v", got)
	}
	// Equal nodes share evenly.
	a, b, c := stageNode{id: "a", speed: 5, layers: 99}, stageNode{id: "b", speed: 5, layers: 99}, stageNode{id: "c", speed: 5, layers: 99}
	if got, _ := planStages(uniform(9), []stageNode{a, b, c}, 3); len(got) != 3 || got[0].to != 3 || got[1].to != 6 {
		t.Fatalf("equal nodes: %v", got)
	}
	// More stages than layers or nodes: as many as there can be.
	if got, _ := planStages(uniform(2), []stageNode{a, b, c}, 5); len(got) != 2 {
		t.Fatalf("two layers over five stages: %v", got)
	}
}

func equalSpans(a, b []span) bool {
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

func uniform(n int) []int64 {
	c := make([]int64, n)
	for i := range c {
		c[i] = 1
	}
	return c
}

// fits counts the layers from the first a node holds.
func fits(a advert, model string, c Config, positions int) int {
	fit := nodeFit(a, model, llamaShape{c}, positions)
	n := 0
	for n < c.Layers && fit(0, n+1) {
		n++
	}
	return n
}

func TestFits(t *testing.T) {
	c := Config{Dim: 64, Hidden: 128, Layers: 10, Heads: 4, KVHeads: 2, Vocab: 50, SeqLen: 256}
	lb, cb := LayerBytes(c), CacheBytes(c, 1, 100)
	cpu := advert{FreeMem: 3*(lb+cb) + 1}
	if n := fits(cpu, "m", c, 100); n != 3 {
		t.Fatalf("CPU node fits %d layers, want 3", n)
	}
	// Loaded layers of the model cost only their caches: with the first 4
	// loaded, the room for 3 whole layers holds their 4 caches and
	// (3lb - cb)/(lb + cb) layers more.
	cpu.Shards = []Shard{{"m", 0, 4}, {"other", 0, 9}}
	want := 4 + int((3*lb-cb+1)/(lb+cb))
	if n := fits(cpu, "m", c, 100); n != want {
		t.Fatalf("with 4 layers loaded: %d, want %d", n, want)
	}
	// Caches apart from weights: the scarcer bounds.
	apart := advert{Device: "gpu", FreeMem: 8 * lb, FreeCache: 5 * cb}
	if n := fits(apart, "m", c, 100); n != 5 {
		t.Fatalf("caches apart: fits %d, want 5", n)
	}
	// A GPU holding weights and caches is one pool.
	if n := fits(advert{Device: "gpu", FreeMem: 4 * (lb + cb)}, "m", c, 100); n != 4 {
		t.Fatalf("GPU node fits %d, want 4", n)
	}
	if n := fits(advert{FreeMem: 1 << 40}, "m", c, 100); n != c.Layers {
		t.Fatalf("huge node fits %d", n)
	}
}

// Layers of different sizes: each node holds what its memory takes, and
// parts kept on disk need room in the blob store.
func TestPlanUnequalLayers(t *testing.T) {
	sh := testShape{bytes: []int64{10, 50, 10, 10, 50, 10}, disk: []int64{0, 100, 0, 0, 0, 0}}
	big := stageNode{id: "big", speed: 10, fit: nodeFit(advert{FreeMem: 75, FreeDisk: 1000}, "m", sh, 1)}
	small := stageNode{id: "small", speed: 5, fit: nodeFit(advert{FreeMem: 80}, "m", sh, 1)}
	got, err := planStages(sh.bytes, []stageNode{big, small}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !equalSpans(got, []span{{0, 3, "big"}, {3, 6, "small"}}) {
		t.Fatalf("plan %v", got)
	}
	// The layer kept on disk goes only where the disk holds it.
	if _, err := planStages(sh.bytes, []stageNode{small, {id: "other", speed: 20, fit: nodeFit(advert{FreeMem: 1000}, "m", sh, 1)}}, 0); err == nil {
		t.Fatal("a plan without room for the table on disk")
	}
	// Two stages balanced by bytes read: the heavy layers split.
	even := stageNode{id: "a", speed: 1, fit: nodeFit(advert{FreeMem: 1000, FreeDisk: 1000}, "m", sh, 1)}
	other := even
	other.id = "b"
	if got, _ := planStages(sh.bytes, []stageNode{even, other}, 2); !equalSpans(got, []span{{0, 3, "a"}, {3, 6, "b"}}) {
		t.Fatalf("balanced plan %v", got)
	}
}

type testShape struct{ bytes, disk []int64 }

func (s testShape) numLayers() int                       { return len(s.bytes) }
func (s testShape) layerBytes(i int) int64               { return s.bytes[i] }
func (s testShape) diskBytes(i int) int64                { return s.disk[i] }
func (s testShape) readBytes(i int) int64                { return s.bytes[i] }
func (s testShape) hostBytes(int) int64                  { return 0 }
func (s testShape) cacheBytes(from, to, _ int) int64     { return int64(to - from) }
func (s testShape) gas(from, to, tokens int) int64       { return 0 }
func (s testShape) payloadBytes(tokens int) int          { return tokens }
func (s testShape) sizes(b []byte, n int) ([]int, error) { return nil, nil }

// hybridShape's layers keep part of themselves in RAM on a GPU node.
type hybridShape struct {
	testShape
	host []int64
}

func (s hybridShape) hostBytes(i int) int64 { return s.host[i] }

// A GPU node too small for whole layers takes them with their experts in
// its RAM, when that holds them.
func TestPlanHostExperts(t *testing.T) {
	sh := hybridShape{testShape{bytes: []int64{100, 100, 100, 100}, disk: make([]int64, 4)}, []int64{90, 90, 90, 90}}
	gpuNode := stageNode{id: "gpu", speed: 50, fit: nodeFit(advert{Device: "gpu", FreeMem: 40, FreeHost: 300}, "m", sh, 1)}
	cpu := stageNode{id: "cpu", speed: 5, fit: nodeFit(advert{FreeMem: 1000}, "m", sh, 1)}
	got, err := planStages(sh.bytes, []stageNode{gpuNode, cpu}, 0)
	if err != nil {
		t.Fatal(err)
	}
	// Three layers: 3·10 + caches on the GPU, 3·90 in RAM.
	if !equalSpans(got, []span{{0, 3, "gpu"}, {3, 4, "cpu"}}) {
		t.Fatalf("plan %v", got)
	}
	// Without RAM to spare the GPU node holds none.
	small := stageNode{id: "gpu", speed: 50, fit: nodeFit(advert{Device: "gpu", FreeMem: 40, FreeHost: 50}, "m", sh, 1)}
	if got, _ := planStages(sh.bytes, []stageNode{small, cpu}, 0); !equalSpans(got, []span{{0, 4, "cpu"}}) {
		t.Fatalf("plan without RAM: %v", got)
	}
}
