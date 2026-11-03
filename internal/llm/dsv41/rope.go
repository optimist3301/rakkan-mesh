package dsv41

import (
	"math"
	"sync"

	"github.com/optimist3301/rakkan-mesh/internal/num"
)

// rope holds a layer kind's rotary frequencies (precompute_freqs_cis):
// layers without compression use the base theta without YaRN; layers with
// compression the compressed theta with YaRN.
type rope struct {
	freqs []float32 // per channel pair, float32 as the reference keeps them

	mu    sync.Mutex
	cache map[int][]float32 // position -> cos, sin interleaved
}

func newRope(c *Config, compressed bool) *rope {
	dim := c.RopeDim
	base, original := c.RopeTheta, 0
	if compressed {
		base, original = c.CompressRopeTheta, c.OriginalSeqLen
	}
	f := make([]float32, dim/2)
	for i := range f {
		e := float32(2*i) / float32(dim)
		p := float32(num.Exp(num.Log(float64(float32(base))) * float64(e))) // base ** e, in float32
		f[i] = 1 / p
	}
	if original > 0 {
		// YaRN: dimensions whose wavelength fits the training context keep
		// their frequency, those well beyond are divided by the factor, and
		// a linear ramp fades between them.
		corr := func(rot float64) float64 {
			return float64(dim) * num.Log(float64(original)/(rot*2*math.Pi)) / (2 * num.Log(base))
		}
		low := max(math.Floor(corr(c.BetaFast)), 0)
		high := min(math.Ceil(corr(c.BetaSlow)), float64(dim-1))
		span := float32(max(high-low, 1e-3))
		factor := float32(c.RopeFactor)
		for i := range f {
			ramp := min(max((float32(i)-float32(low))/span, 0), 1)
			smooth := 1 - ramp
			f[i] = float32(float32(f[i]/factor)*(1-smooth)) + float32(f[i]*smooth)
		}
	}
	return &rope{freqs: f, cache: map[int][]float32{}}
}

// at returns cos and sin of every pair's angle at a position, interleaved.
func (r *rope) at(pos int) []float32 {
	r.mu.Lock()
	defer r.mu.Unlock()
	if cs, ok := r.cache[pos]; ok {
		return cs
	}
	cs := make([]float32, 2*len(r.freqs))
	for i, f := range r.freqs {
		a := float32(float32(pos) * f)
		s, c := num.Sincos(float64(a))
		cs[2*i], cs[2*i+1] = float32(c), float32(s)
	}
	if len(r.cache) > 1<<16 {
		clear(r.cache)
	}
	r.cache[pos] = cs
	return cs
}

// apply rotates x (len 2·pairs, consecutive pairs as complex numbers) by
// the angles at pos, the inverse rotation if inverse; results in bf16.
func (r *rope) apply(x []float32, pos int, inverse bool) {
	cs := r.at(pos)
	for i := 0; i < len(x); i += 2 {
		a, b := x[i], x[i+1]
		c, s := cs[i], cs[i+1]
		if inverse {
			s = -s
		}
		x[i] = bf16(float32(a*c) - float32(b*s))
		x[i+1] = bf16(float32(a*s) + float32(b*c))
	}
}
