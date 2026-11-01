package llm

import (
	"cmp"
	"fmt"
	"slices"

	"github.com/libp2p/go-libp2p/core/peer"
)

// Planning a pipeline over nodes of different power: a CPU-only laptop, a
// workstation with a large GPU, a server with much RAM. Each node gossips
// its free memory (on its GPU, if it has one, and in RAM for caches) and
// the speed at which it streams weights through products (see
// MeasureSpeed). For one generation the stages run one after another, so
// a token costs the sum over stages of layers/speed: the fastest nodes
// should take as many layers as they can hold, and slower ones only what
// remains. Nodes too small for even one layer get none.

// stageNode is a node able to serve stages, as the planner sees it.
type stageNode struct {
	id     peer.ID
	speed  float64 // GB/s of weights
	layers int     // how many of the model's layers fit, for this generation
	free   int64   // memory, to break ties
}

// span is a range of layers assigned to a node.
type span struct {
	from, to int
	node     peer.ID
}

// fits returns how many layers of a model a node can hold for a
// generation of positions: weights and caches share its memory (its GPU's,
// if it has one) unless it advertises memory for caches apart. Layers of
// the model it has loaded already cost only caches.
func fits(a advert, model string, c Config, positions int) int {
	lb, cb := LayerBytes(c), CacheBytes(c, 1, positions)
	loaded := 0
	for _, sh := range a.Shards {
		if sh.Model == model {
			loaded += sh.To - sh.From
		}
	}
	var n int64
	if a.FreeCache > 0 {
		n = min(a.FreeMem/lb+int64(loaded), a.FreeCache/max(cb, 1))
	} else {
		n = (a.FreeMem + int64(loaded)*lb) / (lb + cb)
	}
	return int(max(min(n, int64(c.Layers)), 0))
}

func sortNodes(nodes []stageNode) {
	slices.SortFunc(nodes, func(a, b stageNode) int {
		if c := cmp.Compare(b.speed, a.speed); c != 0 {
			return c
		}
		if c := cmp.Compare(b.free, a.free); c != 0 {
			return c
		}
		return cmp.Compare(a.id, b.id)
	})
}

// planStages assigns a model's layers to nodes. With stages <= 0 the
// fastest nodes take as many layers as fit, which minimizes both compute
// time and hops. With stages > 0 the layers are spread over that many of
// the fastest nodes (fewer if there are fewer nodes or layers): a layer
// each, then the rest in proportion to their speed within what each holds,
// each layer going to the node whose stage would then finish soonest.
func planStages(layers int, nodes []stageNode, stages int) ([]span, error) {
	nodes = slices.DeleteFunc(slices.Clone(nodes), func(n stageNode) bool { return n.layers <= 0 })
	sortNodes(nodes)
	total := 0
	for _, n := range nodes {
		total += n.layers
	}
	var counts []int
	if stages <= 0 {
		left := layers
		for _, n := range nodes {
			k := min(n.layers, left)
			counts = append(counts, k)
			left -= k
		}
		if left > 0 {
			return nil, fmt.Errorf("%w: the model has %d layers, nodes serving stages hold %d", ErrNoStages, layers, total)
		}
	} else {
		nodes = nodes[:min(stages, len(nodes), layers)]
		counts = make([]int, len(nodes))
		// Each stage asked for gets a layer; the rest go by speed.
		for i := range counts {
			counts[i] = 1
		}
		for range layers - len(nodes) {
			best := -1
			var bestT float64
			for i, n := range nodes {
				if counts[i] >= n.layers {
					continue
				}
				t := float64(counts[i]+1) / max(n.speed, 1e-9)
				if best < 0 || t < bestT {
					best, bestT = i, t
				}
			}
			if best < 0 {
				held := 0
				for _, n := range nodes {
					held += n.layers
				}
				return nil, fmt.Errorf("%w: the model has %d layers, the %d fastest nodes hold %d", ErrNoStages, layers, len(nodes), held)
			}
			counts[best]++
		}
	}
	var out []span
	at := 0
	for i, k := range counts {
		if k > 0 {
			out = append(out, span{at, at + k, nodes[i].id})
			at += k
		}
	}
	if len(out) == 0 {
		return nil, ErrNoStages
	}
	return out, nil
}

// stageNodes lists the nodes able to serve a model's stages: members
// advertising so, and this node if it serves.
func (s *Service) stageNodes(model string, c Config, positions int, exclude map[peer.ID]bool) []stageNode {
	var out []stageNode
	add := func(p peer.ID, a advert) {
		if exclude[p] || s.cfg.Trust != nil && s.cfg.Trust.Suspect(p) {
			return
		}
		out = append(out, stageNode{id: p, speed: a.Speed, layers: fits(a, model, c, positions), free: a.FreeMem})
	}
	for p, a := range s.servers() {
		add(p, a)
	}
	if s.cfg.Serve {
		add(s.node.ID(), s.advert())
	}
	return out
}
