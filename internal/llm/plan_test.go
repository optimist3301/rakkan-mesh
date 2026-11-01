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
	got, err := planStages(32, nodes, 0)
	if err != nil {
		t.Fatal(err)
	}
	want := []span{{0, 20, "gpu-big"}, {20, 26, "gpu-small"}, {26, 32, "cpu"}}
	if !equalSpans(got, want) {
		t.Fatalf("automatic plan %v, want %v", got, want)
	}
	// A model the big GPU holds alone needs no other node.
	if got, _ := planStages(16, nodes, 0); !equalSpans(got, []span{{0, 16, "gpu-big"}}) {
		t.Fatalf("small model: %v", got)
	}
	// Too big for all of them together.
	if _, err := planStages(100, nodes, 0); !errors.Is(err, ErrNoStages) {
		t.Fatalf("oversized model: %v", err)
	}

	// Two stages: the two fastest, split by speed within what they hold.
	got, err = planStages(16, nodes, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !equalSpans(got, []span{{0, 10, "gpu-big"}, {10, 16, "gpu-small"}}) {
		t.Fatalf("two stages: %v", got)
	}
	// Every stage asked for gets a layer, however slow its node.
	slow := stageNode{id: "slow", speed: 0.01, layers: 10}
	if got, _ := planStages(4, []stageNode{gpuBig, slow}, 2); len(got) != 2 || got[1].node != "slow" || got[1].to-got[1].from != 1 {
		t.Fatalf("slow node in two stages: %v", got)
	}
	// Equal nodes share evenly.
	a, b, c := stageNode{id: "a", speed: 5, layers: 99}, stageNode{id: "b", speed: 5, layers: 99}, stageNode{id: "c", speed: 5, layers: 99}
	if got, _ := planStages(9, []stageNode{a, b, c}, 3); len(got) != 3 || got[0].to != 3 || got[1].to != 6 {
		t.Fatalf("equal nodes: %v", got)
	}
	// More stages than layers or nodes: as many as there can be.
	if got, _ := planStages(2, []stageNode{a, b, c}, 5); len(got) != 2 {
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

func TestFits(t *testing.T) {
	c := Config{Dim: 64, Hidden: 128, Layers: 10, Heads: 4, KVHeads: 2, Vocab: 50, SeqLen: 256}
	lb, cb := LayerBytes(c), CacheBytes(c, 1, 100)
	cpu := advert{FreeMem: 3*(lb+cb) + 1}
	if n := fits(cpu, "m", c, 100); n != 3 {
		t.Fatalf("CPU node fits %d layers, want 3", n)
	}
	// Loaded layers of the model cost only their caches: the 4 loaded take
	// 4 caches of the room for 3 whole layers, leaving 2 more.
	cpu.Shards = []Shard{{"m", 0, 4}, {"other", 0, 9}}
	if n := fits(cpu, "m", c, 100); n != 6 {
		t.Fatalf("with 4 layers loaded: %d, want 6", n)
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
