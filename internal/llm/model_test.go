package llm

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestCheckpointRoundTrip(t *testing.T) {
	ck := synthetic(3)
	back, err := ReadCheckpoint(ck.Encode())
	if err != nil {
		t.Fatal(err)
	}
	if string(back.Encode()) != string(ck.Encode()) || back.Config != ck.Config {
		t.Fatal("checkpoint changed in a round trip")
	}
}

// synthetic returns a small random model with grouped-query attention
// (two query heads per key/value head), which the real test model lacks.
func synthetic(seed uint64) *Checkpoint {
	c := Config{Dim: 32, Hidden: 72, Layers: 3, Heads: 4, KVHeads: 2, Vocab: 50, SeqLen: 24, SharedClassifier: false}
	rng := rand.New(rand.NewPCG(seed, seed))
	r := func(n int, scale float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64() * scale)
		}
		return v
	}
	ck := &Checkpoint{Config: c, Embed: r(c.Vocab*c.Dim, 1), Head: &Head{RmsFinal: r(c.Dim, 1), Wcls: r(c.Vocab*c.Dim, 0.2)}}
	for range c.Layers {
		l := &Layer{}
		for i, size := range c.layerSizes() {
			scale := 0.2
			if i == 0 || i == 5 { // norm weights
				scale = 1
			}
			*l.tensors()[i] = r(size, scale)
		}
		ck.Layers = append(ck.Layers, l)
	}
	return ck
}

// naiveLogits is an independent float64 implementation of the forward
// pass over a whole token sequence, written from the architecture's
// definition rather than from the engine, for comparison.
func naiveLogits(ck *Checkpoint, tokens []int) []float64 {
	c := ck.Config
	hs := c.HeadSize()
	f := func(v []float32) []float64 {
		out := make([]float64, len(v))
		for i, x := range v {
			out[i] = float64(x)
		}
		return out
	}
	matvec := func(w []float32, x []float64, rows int) []float64 {
		out := make([]float64, rows)
		for i := range out {
			for j, xv := range x {
				out[i] += float64(w[i*len(x)+j]) * xv
			}
		}
		return out
	}
	norm := func(x []float64, w []float32) []float64 {
		ss := 0.0
		for _, v := range x {
			ss += v * v
		}
		inv := 1 / math.Sqrt(ss/float64(len(x))+1e-5)
		out := make([]float64, len(x))
		for i := range x {
			out[i] = float64(w[i]) * x[i] * inv
		}
		return out
	}
	rope := func(v []float64, pos int) {
		for i := 0; i < len(v); i += 2 {
			theta := float64(pos) * math.Pow(10000, -float64(i%hs)/float64(hs))
			a, b := v[i], v[i+1]
			v[i], v[i+1] = a*math.Cos(theta)-b*math.Sin(theta), a*math.Sin(theta)+b*math.Cos(theta)
		}
	}
	type kv struct{ k, v [][]float64 }
	caches := make([]kv, c.Layers)
	var x []float64
	for pos, tok := range tokens {
		x = f(ck.Embed[tok*c.Dim : (tok+1)*c.Dim])
		for li, l := range ck.Layers {
			xn := norm(x, l.RmsAtt)
			q, k, v := matvec(l.Wq, xn, c.Dim), matvec(l.Wk, xn, c.KVDim()), matvec(l.Wv, xn, c.KVDim())
			rope(q, pos)
			rope(k, pos)
			caches[li].k = append(caches[li].k, k)
			caches[li].v = append(caches[li].v, v)
			att := make([]float64, c.Dim)
			for h := range c.Heads {
				g := h / (c.Heads / c.KVHeads)
				scores := make([]float64, pos+1)
				mx := math.Inf(-1)
				for t := range scores {
					for i := range hs {
						scores[t] += q[h*hs+i] * caches[li].k[t][g*hs+i]
					}
					scores[t] /= math.Sqrt(float64(hs))
					mx = max(mx, scores[t])
				}
				sum := 0.0
				for t := range scores {
					scores[t] = math.Exp(scores[t] - mx)
					sum += scores[t]
				}
				for t := range scores {
					for i := range hs {
						att[h*hs+i] += scores[t] / sum * caches[li].v[t][g*hs+i]
					}
				}
			}
			o := matvec(l.Wo, att, c.Dim)
			for i := range x {
				x[i] += o[i]
			}
			xn = norm(x, l.RmsFfn)
			a, b := matvec(l.W1, xn, c.Hidden), matvec(l.W3, xn, c.Hidden)
			for i := range a {
				a[i] = a[i] / (1 + math.Exp(-a[i])) * b[i]
			}
			o = matvec(l.W2, a, c.Dim)
			for i := range x {
				x[i] += o[i]
			}
		}
	}
	return matvec(ck.Head.Wcls, norm(x, ck.Head.RmsFinal), c.Vocab)
}

func TestForwardMatchesNaiveReference(t *testing.T) {
	ck := synthetic(1)
	c := ck.Config
	tokens := []int{1, 7, 33, 2, 49, 12, 12, 30, 5, 18}
	stage := NewStage(c, ck.Layers, NewRope(c))
	for pos, tok := range tokens {
		x := Embed(c, ck.Embed, tok)
		stage.Forward(x, pos)
		got := ck.Head.Logits(c, x)
		want := naiveLogits(ck, tokens[:pos+1])
		worst := 0.0
		for i := range got {
			worst = max(worst, math.Abs(float64(got[i])-want[i]))
		}
		if worst > 1e-3 {
			t.Fatalf("position %d: logits differ from the reference by %g", pos, worst)
		}
		if argmax(got) != argmaxF64(want) {
			t.Fatalf("position %d: predicted %d, reference %d", pos, argmax(got), argmaxF64(want))
		}
	}
}

func argmaxF64(v []float64) int {
	best := 0
	for i := range v {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

// Splitting the layers across stages, as nodes would, changes nothing.
func TestStagesComposeExactly(t *testing.T) {
	ck := synthetic(2)
	c := ck.Config
	rope := NewRope(c)
	whole := NewStage(c, ck.Layers, rope)
	split := []*Stage{NewStage(c, ck.Layers[:1], rope), NewStage(c, ck.Layers[1:], rope)}
	for pos, tok := range []int{1, 4, 9, 16, 25} {
		a, b := Embed(c, ck.Embed, tok), Embed(c, ck.Embed, tok)
		whole.Forward(a, pos)
		for _, s := range split {
			s.Forward(b, pos)
		}
		for i := range a {
			if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
				t.Fatalf("position %d: split pipeline differs", pos)
			}
		}
	}
}
