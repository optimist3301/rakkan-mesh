package llm

// Sampler picks the next token from logits. Temperature 0 picks the most
// likely token; otherwise tokens are drawn from softmax(logits/temperature)
// with a seeded generator, so a seed reproduces a generation exactly.
type Sampler struct {
	Temperature float32
	state       uint64
}

func NewSampler(temperature float32, seed uint64) *Sampler {
	if seed == 0 {
		seed = 0x9e3779b97f4a7c15
	}
	return &Sampler{Temperature: temperature, state: seed}
}

// next is xorshift64*: a deterministic stream of uniform values in [0, 1).
func (s *Sampler) next() float32 {
	s.state ^= s.state >> 12
	s.state ^= s.state << 25
	s.state ^= s.state >> 27
	return float32((s.state*0x2545f4914f6cdd1d)>>40) / (1 << 24)
}

func (s *Sampler) Sample(logits []float32) int {
	if s.Temperature <= 0 {
		return argmax(logits)
	}
	p := make([]float32, len(logits))
	for i, v := range logits {
		p[i] = v / s.Temperature
	}
	softmax(p)
	r := s.next()
	var cum float32
	for i, v := range p {
		if cum += v; r < cum {
			return i
		}
	}
	return len(p) - 1
}

func argmax(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}
