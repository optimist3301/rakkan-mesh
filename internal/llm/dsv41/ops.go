package dsv41

import (
	"cmp"
	"math"
	"slices"

	"github.com/optimist3301/rakkan-mesh/internal/num"
)

// Elementwise and reduction operations, each with a fixed order of
// arithmetic, every product rounded before it is added (num's rule against
// fused multiply-adds). Values the reference keeps in bfloat16 are float32
// here, rounded with bf16 wherever the reference rounds.

func bf16(f float32) float32 { return num.RoundBF16(f) }

// sum32 adds v in a fixed order: 32 running sums (element i to sum i mod
// 32), combined by halves.
func sum32(v []float32) float32 {
	var s [32]float32
	for i, x := range v {
		s[i&31] += x
	}
	for w := 16; w >= 1; w /= 2 {
		for j := range w {
			s[j] += s[j+w]
		}
	}
	return s[0]
}

// meanSquare is mean(x²): the squares rounded, summed in Dot32's order.
func meanSquare(x []float32) float32 { return num.Dot32(x, x) / float32(len(x)) }

// rsqrt is 1/sqrt(v), each step rounded: sqrt correctly rounded, then the
// division.
func rsqrt(v float32) float32 { return 1 / float32(math.Sqrt(float64(v))) }

// rmsNorm sets out to bf16(w · (x · rsqrt(mean(x²) + eps))).
func rmsNorm(out, x, w []float32, eps float32) {
	r := rsqrt(meanSquare(x) + eps)
	for i, v := range x {
		out[i] = bf16(float32(w[i] * float32(v*r)))
	}
}

func sigmoid(x float32) float32 { return 1 / (1 + num.Exp32(-x)) }

// silu is x·sigmoid(x), as x / (1 + e^-x).
func silu(x float32) float32 { return x / (1 + num.Exp32(-x)) }

// softplus is log(1 + e^x), x itself above 20 (PyTorch's threshold), in
// float32 (num.Log1p32) as a GPU can compute it too.
func softplus(x float32) float32 {
	if x > 20 {
		return x
	}
	return num.Log1p32(num.Exp32(x))
}

func sqrt32(x float32) float32 { return float32(math.Sqrt(float64(x))) }

// topK returns the indices of the k largest values, largest first; among
// equal values the lower index wins (docs/dev/deepseek-v41.md, section 4).
// NaNs are not expected.
func topK(v []float32, k int) []int {
	idx := make([]int, len(v))
	for i := range idx {
		idx[i] = i
	}
	slices.SortStableFunc(idx, func(a, b int) int { return cmp.Compare(v[b], v[a]) })
	return idx[:k]
}

var negInf = float32(math.Inf(-1))

// splitSinkhorn turns hyper-connection mixing logits into pre [hc], post
// [hc] and comb [hc·hc] (row-major, comb[j·hc+k]), comb made doubly
// stochastic, as the reference's hc_split_sinkhorn kernel does.
func splitSinkhorn(mix, scale, base []float32, hc, iters int, eps float32, pre, post, comb []float32) {
	for j := range hc {
		pre[j] = sigmoid(float32(mix[j]*scale[0])+base[j]) + eps
		post[j] = 2 * sigmoid(float32(mix[hc+j]*scale[1])+base[hc+j])
	}
	for i := range hc * hc {
		comb[i] = float32(mix[2*hc+i]*scale[2]) + base[2*hc+i]
	}
	for j := range hc {
		row := comb[j*hc : (j+1)*hc]
		m := slices.Max(row)
		var s float32
		for k := range row {
			row[k] = num.Exp32(row[k] - m)
			s += row[k]
		}
		for k := range row {
			row[k] = row[k]/s + eps
		}
	}
	colNorm := func() {
		for k := range hc {
			var s float32
			for j := range hc {
				s += comb[j*hc+k]
			}
			for j := range hc {
				comb[j*hc+k] /= s + eps
			}
		}
	}
	colNorm()
	for range iters - 1 {
		for j := range hc {
			var s float32
			for k := range hc {
				s += comb[j*hc+k]
			}
			for k := range hc {
				comb[j*hc+k] /= s + eps
			}
		}
		colNorm()
	}
}

// hcPre collapses the hc copies of the stream (h, [hc][dim]) into out:
// bf16(Σ_c p_c·h_c), summed in copy order.
func hcPre(out, h, p []float32) {
	dim := len(out)
	for d := range dim {
		var s float32
		for c := range p {
			s += float32(p[c] * h[c*dim+d])
		}
		out[d] = bf16(s)
	}
}

// hcPost sets the stream h to copy k = bf16(post_k·y + Σ_j comb[j,k]·res_j).
func hcPost(h, y, res, post, comb []float32) {
	dim, hc := len(y), len(post)
	for k := range hc {
		for d := range dim {
			var s float32
			for j := range hc {
				s += float32(comb[j*hc+k] * res[j*dim+d])
			}
			h[k*dim+d] = bf16(float32(post[k]*y[d]) + s)
		}
	}
}
