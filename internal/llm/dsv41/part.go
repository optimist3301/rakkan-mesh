package dsv41

import (
	"slices"
	"strconv"
	"strings"
)

// TensorLayer returns the block a checkpoint tensor belongs to.
func TensorLayer(name string) (int, bool) {
	rest, ok := strings.CutPrefix(name, "layers.")
	if !ok {
		return 0, false
	}
	n, _, ok := strings.Cut(rest, ".")
	if !ok {
		return 0, false
	}
	l, err := strconv.Atoi(n)
	return l, err == nil && l >= 0
}

// IsHeadTensor reports whether a tensor belongs to the coordinator's part.
func IsHeadTensor(name string) bool {
	return name == "embed.weight" || name == "norm.weight" || name == "head.weight"
}

// Sizes, from the config alone: a node plans and accounts for parts
// before it has them, and a coordinator prices stages it does not run.

// matrix sizes by kind: bytes in a checkpoint, bytes held once loaded.
func fp8Bytes(rows, cols int) int64 {
	return int64(rows)*int64(cols) + int64((rows+31)/32)*int64(cols/32)
}

func fp4Bytes(rows, cols int) int64 {
	return int64(rows)*int64(cols)/2 + int64(rows)*int64(cols/32)
}

// bf16 matrices are held twice: the checkpoint's bytes and a copy.
func bf16Bytes(rows, cols int) int64 { return 4 * int64(rows) * int64(cols) }

// vectors and float32 matrices: at most float32 in the checkpoint and
// float32 once loaded.
func f32Bytes(n int) int64 { return 8 * int64(n) }

// layerSizes sums a block's weights: held is the memory a loaded block
// takes (an upper bound, the blob's size included), active the bytes a
// token reads, params the weights a token multiplies by. The Engram
// table is apart (EngramTableBytes): a token reads 24 of its rows.
func layerSizes(c *Config, i int) (held, active, params int64) {
	add := func(b, n int64, routed bool) {
		held += b
		if routed {
			// A token uses ActiveExperts of the RoutedExperts.
			active += b * int64(c.ActiveExperts) / int64(c.RoutedExperts)
			params += n * int64(c.ActiveExperts) / int64(c.RoutedExperts)
		} else {
			active += b
			params += n
		}
	}
	fp8 := func(r, k int) { add(fp8Bytes(r, k), int64(r)*int64(k), false) }
	bf := func(r, k int) { add(bf16Bytes(r, k), int64(r)*int64(k), false) }
	vec := func(n int) { add(f32Bytes(n), int64(n), false) }
	d, hc, hd := c.Dim, c.HCMult, c.HeadDim
	mix := (2 + hc) * hc
	vec(2 * d)                    // norms
	vec(2 * (mix*hc*d + mix + 3)) // hyper-connections
	fp8(c.QLoRA, d)
	vec(c.QLoRA)
	fp8(c.Heads*hd, c.QLoRA)
	fp8(hd, d)
	vec(hd + c.Heads)
	fp8(c.OGroups*c.OLoRA, c.Heads*hd/c.OGroups)
	fp8(d, c.OGroups*c.OLoRA)
	if c.isKVSource(i) {
		bf(hd, d)
		if c.CompressRatios[i] > 1 {
			bf(hd, d)
		}
		vec(hd)
		bf(c.IndexDim, hd)
		vec(c.IndexDim)
	}
	if c.isIndexSource(i) {
		fp8(c.IndexHeads*c.IndexDim, c.QLoRA)
		bf(c.IndexHeads, d)
	}
	bf(c.RoutedExperts, d)
	vec(c.RoutedExperts)
	for range c.RoutedExperts {
		n := 3 * int64(c.MoEInter) * int64(d)
		add(2*fp4Bytes(c.MoEInter, d)+fp4Bytes(d, c.MoEInter), n, true)
	}
	fp8(c.MoEInter, d)
	fp8(d, c.MoEInter)
	fp8(c.MoEInter, d)
	if k := slices.Index(c.EngramLayers, i); k >= 0 {
		fp8(d*(hc+1), (c.EngramMaxNgram-1)*c.EngramHeads*c.EngramDim)
		vec(3 * hc * d)
		rows := int64((c.EngramMaxNgram - 1) * c.EngramHeads)
		active += rows * (int64(c.EngramDim) + int64(c.EngramDim/32))
		params += rows * int64(c.EngramDim)
	}
	return held, active, params
}

// LayerBytes bounds the memory block i takes loaded, its Engram table
// aside, and so the size of its blob.
func LayerBytes(c *Config, i int) int64 { h, _, _ := layerSizes(c, i); return h }

// ActiveBytes is what block i reads of its weights per token: what a
// node's speed (bytes per second through products) is measured against.
func ActiveBytes(c *Config, i int) int64 { _, a, _ := layerSizes(c, i); return a }

// ActiveParams is how many weights block i multiplies a token by.
func ActiveParams(c *Config, i int) int64 { _, _, p := layerSizes(c, i); return p }

// EngramTableBytes is the size of block i's Engram table, 0 for none:
// the rows, then their scales.
func EngramTableBytes(c *Config, i int) int64 {
	k := slices.Index(c.EngramLayers, i)
	if k < 0 {
		return 0
	}
	n := int64(c.EngramRows[k]) * int64(c.EngramDim)
	return n + n/32
}

// HeadBytes bounds the memory of the coordinator's part, and so its blob.
func HeadBytes(c *Config) int64 {
	return bf16Bytes(c.Vocab, c.Dim)*2 + f32Bytes(c.Dim)
}

// CacheBytes bounds the caches of a stage running blocks [from, to) for
// a generation of positions: each block's window, the compressed caches of
// its KV sources and the copy of the one before it.
func CacheBytes(c *Config, from, to, positions int) int64 {
	entry := 4 * int64(c.HeadDim+c.IndexDim)
	var n int64
	for i := from; i < to; i++ {
		n += 4 * int64(c.Window) * int64(c.HeadDim)
		if r := c.CompressRatios[i]; c.isKVSource(i) {
			n += int64(positions/r)*entry + 8*int64(r)*int64(c.HeadDim)
		}
	}
	if src := sourceBefore(c, from); src >= 0 {
		n += int64(positions/c.CompressRatios[src]) * entry
	}
	return n
}
