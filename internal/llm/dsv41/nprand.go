package dsv41

import (
	"errors"
	"math/bits"
)

// numpyIntegers returns what numpy.random.default_rng(seed).integers(0,
// high, n, dtype=np.int64) returns: the Engram hash multipliers come from
// it. That is numpy's SeedSequence, its PCG64 (XSL-RR) generator, and
// Lemire's method for bounded integers, ported exactly. Only ranges above
// 2^32 are supported (numpy draws smaller ones 32 bits at a time).
func numpyIntegers(seed uint64, high int64, n int) ([]int64, error) {
	if high-1 <= 0xffffffff || high <= 0 {
		return nil, errors.New("numpyIntegers: range must exceed 2^32")
	}
	g := newPCG64(seedSequenceState(seed))
	rng := uint64(high - 1)
	out := make([]int64, n)
	for i := range out {
		out[i] = int64(g.boundedLemire(rng))
	}
	return out, nil
}

// SeedSequence constants (numpy/random/bit_generator.pyx).
const (
	ssInitA    = 0x43b0d7e5
	ssMultA    = 0x931e8875
	ssInitB    = 0x8b51f9dd
	ssMultB    = 0x58f38ded
	ssMixMultL = 0xca01f9dd
	ssMixMultR = 0x4973f715
	ssXShift   = 16
	ssPoolSize = 4
)

// seedSequenceState is SeedSequence(seed).generate_state(4, np.uint64).
func seedSequenceState(seed uint64) [4]uint64 {
	// The entropy as 32-bit words, least significant first; 0 is [0].
	var entropy []uint32
	if seed == 0 {
		entropy = []uint32{0}
	}
	for s := seed; s > 0; s >>= 32 {
		entropy = append(entropy, uint32(s))
	}
	hashConst := uint32(ssInitA)
	hashmix := func(v uint32) uint32 {
		v ^= hashConst
		hashConst *= ssMultA
		v *= hashConst
		return v ^ v>>ssXShift
	}
	mix := func(x, y uint32) uint32 {
		r := ssMixMultL*x - ssMixMultR*y
		return r ^ r>>ssXShift
	}
	var pool [ssPoolSize]uint32
	for i := range pool {
		var e uint32
		if i < len(entropy) {
			e = entropy[i]
		}
		pool[i] = hashmix(e)
	}
	for src := range pool {
		for dst := range pool {
			if src != dst {
				pool[dst] = mix(pool[dst], hashmix(pool[src]))
			}
		}
	}
	for src := ssPoolSize; src < len(entropy); src++ {
		for dst := range pool {
			pool[dst] = mix(pool[dst], hashmix(entropy[src]))
		}
	}
	// generate_state: 8 words cycling through the pool, paired little-end
	// first into 4 uint64s.
	var words [8]uint32
	h := uint32(ssInitB)
	for i := range words {
		v := pool[i%ssPoolSize]
		v ^= h
		h *= ssMultB
		v *= h
		words[i] = v ^ v>>ssXShift
	}
	var out [4]uint64
	for i := range out {
		out[i] = uint64(words[2*i]) | uint64(words[2*i+1])<<32
	}
	return out
}

// pcg64 is numpy's PCG64: a 128-bit LCG with the XSL-RR output function.
type pcg64 struct{ hi, lo, incHi, incLo uint64 }

const (
	pcgMultHi = 2549297995355413924
	pcgMultLo = 4865540595714422341
)

// newPCG64 seeds as numpy does from SeedSequence's four words: the first
// two are the state's high and low halves, the last two the stream.
func newPCG64(v [4]uint64) *pcg64 {
	g := &pcg64{}
	// inc = (initseq << 1) | 1, initseq = v[2]:v[3]
	g.incHi = v[2]<<1 | v[3]>>63
	g.incLo = v[3]<<1 | 1
	g.step()
	var c uint64
	g.lo, c = bits.Add64(g.lo, v[1], 0)
	g.hi, _ = bits.Add64(g.hi, v[0], c)
	g.step()
	return g
}

// step advances the state: state = state·mult + inc (mod 2^128).
func (g *pcg64) step() {
	hi, lo := bits.Mul64(g.lo, pcgMultLo)
	hi += g.hi*pcgMultLo + g.lo*pcgMultHi
	var c uint64
	lo, c = bits.Add64(lo, g.incLo, 0)
	hi, _ = bits.Add64(hi, g.incHi, c)
	g.hi, g.lo = hi, lo
}

func (g *pcg64) next() uint64 {
	g.step()
	return bits.RotateLeft64(g.hi^g.lo, -int(g.hi>>58))
}

// boundedLemire returns a uniform integer in [0, rng] (rng < 2^64 - 1).
func (g *pcg64) boundedLemire(rng uint64) uint64 {
	excl := rng + 1
	hi, lo := bits.Mul64(g.next(), excl)
	if lo < excl {
		threshold := (^uint64(0) - rng) % excl
		for lo < threshold {
			hi, lo = bits.Mul64(g.next(), excl)
		}
	}
	return hi
}
