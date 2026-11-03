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
	id    peer.ID
	speed float64 // GB/s of weights
	// fit reports whether layers [from, to) fit the node for this
	// generation; without it, layers is how many do.
	fit    func(from, to int) bool
	layers int
	free   int64 // memory, to break ties
}

func (n stageNode) fits(from, to int) bool {
	if n.fit != nil {
		return n.fit(from, to)
	}
	return to-from <= n.layers
}

// span is a range of layers assigned to a node.
type span struct {
	from, to int
	node     peer.ID
}

// shape is what planning and accounting need to know of a model's layers,
// which need not be alike: DeepSeek-V4.1's differ in their attention and
// Engram, and a token reads only some of their experts.
type shape interface {
	numLayers() int
	// layerBytes is the memory layer i takes loaded, diskBytes what of it
	// stays on disk, readBytes what a token reads of it: what a node's
	// speed is measured against.
	layerBytes(i int) int64
	diskBytes(i int) int64
	readBytes(i int) int64
	// cacheBytes is the memory of a stage's caches for positions.
	cacheBytes(from, to, positions int) int64
	// gas prices a forward of tokens through layers [from, to).
	gas(from, to, tokens int) int64
	// payloadBytes bounds what passes between stages for tokens.
	payloadBytes(tokens int) int
}

// nodeFit tells which layer ranges of a model a node can hold for a
// generation of positions: weights and caches share its memory (its
// GPU's, if it has one) unless it advertises memory for caches apart.
// Layers of the model it has loaded already cost only caches.
func nodeFit(a advert, model string, sh shape, positions int) func(from, to int) bool {
	var loaded []Shard
	for _, s := range a.Shards {
		if s.Model == model {
			loaded = append(loaded, s)
		}
	}
	return func(from, to int) bool {
		var weights, disk int64
		for i := from; i < to; i++ {
			if !slices.ContainsFunc(loaded, func(s Shard) bool { return s.From <= i && i < s.To }) {
				weights += sh.layerBytes(i)
				disk += sh.diskBytes(i)
			}
		}
		cache := sh.cacheBytes(from, to, positions)
		if disk > 0 && disk > a.FreeDisk {
			return false
		}
		if a.FreeCache > 0 {
			return weights <= a.FreeMem && cache <= a.FreeCache
		}
		return weights+cache <= a.FreeMem
	}
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

// planStages assigns a model's layers, which cost[i] bytes each of reading
// per token, to nodes. With stages <= 0 the fastest nodes take as many
// layers as fit, which minimizes both compute time and hops. With stages >
// 0 the layers are spread over that many of the fastest nodes (fewer if
// there are fewer nodes or layers), a layer at least each, so that the
// slowest stage is as quick as can be: the stages of a token run one
// after another, but those of a prompt's tokens overlap.
func planStages(cost []int64, nodes []stageNode, stages int) ([]span, error) {
	layers := len(cost)
	nodes = slices.DeleteFunc(slices.Clone(nodes), func(n stageNode) bool {
		for i := range layers {
			if n.fits(i, i+1) {
				return false
			}
		}
		return true
	})
	sortNodes(nodes)
	// longest is how many layers from at a node takes, at most most.
	longest := func(n stageNode, at, most int) int {
		k := 0
		for k < most && n.fits(at, at+k+1) {
			k++
		}
		return k
	}
	var out []span
	if stages <= 0 {
		at := 0
		for _, n := range nodes {
			if at == layers {
				break
			}
			if k := longest(n, at, layers-at); k > 0 {
				out = append(out, span{at, at + k, n.id})
				at += k
			}
		}
		if at < layers || len(out) == 0 {
			return nil, fmt.Errorf("%w: the model has %d layers, nodes serving stages hold %d", ErrNoStages, layers, at)
		}
		return out, nil
	}
	nodes = nodes[:min(stages, len(nodes), layers)]
	if len(nodes) == 0 {
		return nil, ErrNoStages
	}
	// The quickest bound on a stage's time for which the layers can be
	// cut, in node order, into stages within it.
	sum := make([]int64, layers+1)
	for i, c := range cost {
		sum[i+1] = sum[i] + c
	}
	cut := func(bound float64) []span {
		var out []span
		at := 0
		for j, n := range nodes {
			left := len(nodes) - j - 1 // each later node needs a layer
			k := 0
			for at+k < layers-left && n.fits(at, at+k+1) && float64(sum[at+k+1]-sum[at])/max(n.speed, 1e-9) <= bound {
				k++
			}
			if k == 0 {
				return nil
			}
			out = append(out, span{at, at + k, n.id})
			at += k
		}
		if at < layers {
			return nil
		}
		return out
	}
	var bounds []float64
	for _, n := range nodes {
		for i := range layers {
			for j := i + 1; j <= layers; j++ {
				bounds = append(bounds, float64(sum[j]-sum[i])/max(n.speed, 1e-9))
			}
		}
	}
	slices.Sort(bounds)
	bounds = slices.Compact(bounds)
	i, _ := slices.BinarySearchFunc(bounds, true, func(b float64, _ bool) int {
		if cut(b) != nil {
			return 1
		}
		return -1
	})
	if i == len(bounds) {
		return nil, fmt.Errorf("%w: the model's %d layers do not fit the %d fastest nodes", ErrNoStages, layers, len(nodes))
	}
	return cut(bounds[i]), nil
}

// stageNodes lists the nodes able to serve a model's stages: members
// advertising so, and this node if it serves.
func (s *Service) stageNodes(model string, sh shape, positions int, exclude map[peer.ID]bool) []stageNode {
	var out []stageNode
	add := func(p peer.ID, a advert) {
		if exclude[p] || s.cfg.Trust != nil && s.cfg.Trust.Suspect(p) {
			return
		}
		out = append(out, stageNode{id: p, speed: a.Speed, fit: nodeFit(a, model, sh, positions), free: a.FreeMem})
	}
	for p, a := range s.servers() {
		add(p, a)
	}
	if s.cfg.Serve {
		add(s.node.ID(), s.advert())
	}
	return out
}
