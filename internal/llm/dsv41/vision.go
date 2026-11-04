package dsv41

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"strings"
	"sync"

	"github.com/optimist3301/rakkan-mesh/internal/num"
)

// The image encoder (vision.py): a ViT over one image's patches, with
// full attention among them and a two-dimensional RoPE (a patch's row in
// the first half of each head's angles, its column in the second), then
// the aligner, which merges Downsample×Downsample patches into one of the
// model's positions. Every weight is bf16 and every step rounds to bf16
// where PyTorch's bf16 tensors do; the sums inside a step (products,
// norms, softmax) have this engine's fixed order, so the result agrees
// with the reference to rounding and is the same on every node.

// Vision is the image encoder, the aligner and the span embeddings.
type Vision struct {
	cfg                 *Config
	patch               *bf16Linear
	blocks              []visionBlock
	norm                []float32
	align1, align2      *bf16Linear
	start, end, newLine []float32 // the span's delimiters, bf16 values
	invFreq             []float32 // per angle pair of a half head
	// afterBlock, for tests, sees the stream after each block.
	afterBlock func(i int, x []float32)
}

type visionBlock struct {
	norm1, norm2 []float32
	qkv, o       *bf16Linear
	w1, w2       *bf16Linear
}

// bf16Linear is a PyTorch nn.Linear in bf16: y = bf16(x·Wᵀ + b), the sum in
// float32 (num.DotBF16's order), the bias added to it before the rounding.
type bf16Linear struct {
	w       []uint16
	b       []float32 // nil: none
	out, in int
}

// IsVisionTensor reports whether a checkpoint tensor is the image path's
// (the encoder, the aligner, the span embeddings), not the gates' biases.
func IsVisionTensor(name string) bool {
	return strings.HasPrefix(name, "vision.") || strings.HasPrefix(name, "aligner.") || strings.HasPrefix(name, "image_")
}

// NewVision builds the image encoder from its tensors.
func NewVision(cfg *Config, tensors map[string]*Tensor) (*Vision, error) {
	v := &cfg.Vision
	if v.Layers == 0 {
		return nil, errors.New("dsv41: the model has no image encoder")
	}
	w := &weights{t: tensors, used: map[string]bool{}}
	vis := &Vision{cfg: cfg}
	lin := func(name string, out, in int, bias bool) (*bf16Linear, error) {
		t, err := w.get(name + ".weight")
		if err != nil {
			return nil, err
		}
		if t.DType != "BF16" || !shapeIs(t, out, in) {
			return nil, fmt.Errorf("dsv41: %s: %s %v, want BF16 [%d, %d]", name, t.DType, t.Shape, out, in)
		}
		l := &bf16Linear{w: toBF16(t), out: out, in: in}
		if bias {
			if l.b, err = w.vec(name+".bias", out); err != nil {
				return nil, err
			}
		}
		return l, nil
	}
	p, d := v.Patch, v.Dim
	var err error
	if vis.patch, err = lin("vision.patch_embed.proj", d, 3*p*p, true); err != nil {
		return nil, err
	}
	for i := range v.Layers {
		pre := fmt.Sprintf("vision.blocks.%d.", i)
		var b visionBlock
		if b.norm1, err = w.vec(pre+"norm1.weight", d); err != nil {
			return nil, err
		}
		if b.norm2, err = w.vec(pre+"norm2.weight", d); err != nil {
			return nil, err
		}
		if b.qkv, err = lin(pre+"attn.wqkv", 3*d, d, true); err != nil {
			return nil, err
		}
		if b.o, err = lin(pre+"attn.wo", d, d, true); err != nil {
			return nil, err
		}
		if b.w1, err = lin(pre+"mlp.w1", 2*v.Inter, d, false); err != nil {
			return nil, err
		}
		if b.w2, err = lin(pre+"mlp.w2", d, v.Inter, false); err != nil {
			return nil, err
		}
		vis.blocks = append(vis.blocks, b)
	}
	if vis.norm, err = w.vec("vision.norm.weight", d); err != nil {
		return nil, err
	}
	r := v.Downsample
	if vis.align1, err = lin("aligner.w1", cfg.Dim, d*r*r, true); err != nil {
		return nil, err
	}
	if vis.align2, err = lin("aligner.w2", cfg.Dim, cfg.Dim, true); err != nil {
		return nil, err
	}
	for _, e := range []struct {
		name string
		to   *[]float32
	}{{"image_start", &vis.start}, {"image_end", &vis.end}, {"image_newline", &vis.newLine}} {
		if *e.to, err = w.vec(e.name, cfg.Dim); err != nil {
			return nil, err
		}
		for i, x := range *e.to {
			(*e.to)[i] = bf16(x) // a bf16 parameter
		}
	}
	var unused []string
	for name := range tensors {
		if IsVisionTensor(name) && !w.used[name] {
			unused = append(unused, name)
		}
	}
	if len(unused) > 0 {
		return nil, fmt.Errorf("dsv41: unexpected vision tensors: %v", unused)
	}
	// get_vision_cos_sin: 1 / theta^(2i/half) in float32 for the half of a
	// head each of rows and columns take.
	half := d / v.Heads / 2
	vis.invFreq = make([]float32, half/2)
	for i := range vis.invFreq {
		e := float32(2*i) / float32(half)
		vis.invFreq[i] = 1 / float32(num.Exp(num.Log(float64(float32(v.RopeTheta)))*float64(e)))
	}
	return vis, nil
}

// apply sets out (n rows of l.out) from x (n rows of l.in).
func (l *bf16Linear) apply(out, x []float32, n int) {
	num.MatMulBF16(out, l.w, x, n, l.out)
	for i := range n {
		o := out[i*l.out : (i+1)*l.out]
		for j := range o {
			if l.b != nil {
				o[j] += l.b[j]
			}
			o[j] = bf16c(o[j])
		}
	}
}

// Encode runs the encoder and the aligner on an image: its LLMH·LLMW rows
// of the model's width, bf16, in reading order.
func (vis *Vision) Encode(img *Image) ([]float32, error) {
	x, err := vis.ViT(img)
	if err != nil {
		return nil, err
	}
	return vis.align(x, img.VitH, img.VitW), nil
}

// ViT is the encoder alone (ViT.forward): VitH·VitW rows of its width.
func (vis *Vision) ViT(img *Image) ([]float32, error) {
	v := &vis.cfg.Vision
	n, d, p := img.VitH*img.VitW, v.Dim, v.Patch
	if n == 0 || len(img.Patches) != n*3*p*p {
		return nil, fmt.Errorf("dsv41: %d patch values for a %d×%d grid", len(img.Patches), img.VitH, img.VitW)
	}
	x := make([]float32, n*d)
	vis.patch.apply(x, img.Patches, n)
	cos, sin := vis.angles(img.VitH, img.VitW)
	h := make([]float32, n*d)
	qkv := make([]float32, n*3*d)
	att := make([]float32, n*d)
	y := make([]float32, n*d)
	g := make([]float32, n*2*v.Inter)
	a := make([]float32, n*v.Inter)
	for bi := range vis.blocks {
		b := &vis.blocks[bi]
		for i := range n {
			rmsNorm(h[i*d:(i+1)*d], x[i*d:(i+1)*d], b.norm1, 1e-6)
		}
		b.qkv.apply(qkv, h, n)
		vis.rotate(qkv, n, cos, sin)
		vis.attention(att, qkv, n)
		b.o.apply(y, att, n)
		for i := range x {
			x[i] = bf16c(x[i] + y[i])
		}
		for i := range n {
			rmsNorm(h[i*d:(i+1)*d], x[i*d:(i+1)*d], b.norm2, 1e-6)
		}
		b.w1.apply(g, h, n)
		in := v.Inter
		for i := range n {
			gate, up := g[i*2*in:i*2*in+in], g[i*2*in+in:(i+1)*2*in]
			for j := range in {
				a[i*in+j] = bf16c(bf16c(silu(gate[j])) * up[j])
			}
		}
		b.w2.apply(y, a, n)
		for i := range x {
			x[i] = bf16c(x[i] + y[i])
		}
		if vis.afterBlock != nil {
			vis.afterBlock(bi, x)
		}
	}
	for i := range n {
		rmsNorm(x[i*d:(i+1)*d], x[i*d:(i+1)*d], vis.norm, 1e-6)
	}
	return x, nil
}

// angles are each patch's cos and sin, float32: its row's angles for the
// first quarter of the head's values, its column's for the second.
func (vis *Vision) angles(rows, cols int) (cos, sin []float32) {
	q := len(vis.invFreq)
	cos, sin = make([]float32, rows*cols*2*q), make([]float32, rows*cols*2*q)
	for r := range rows {
		for c := range cols {
			o := (r*cols + c) * 2 * q
			for i, f := range vis.invFreq {
				for k, pos := range [2]int{r, c} {
					s, co := num.Sincos(float64(float32(float32(pos) * f)))
					cos[o+k*q+i], sin[o+k*q+i] = float32(co), float32(s)
				}
			}
		}
	}
	return cos, sin
}

// rotate applies apply_rotary to the queries and keys in qkv: each head's
// halves x1, x2 become x1·cos − x2·sin, x2·cos + x1·sin, in bf16.
func (vis *Vision) rotate(qkv []float32, n int, cos, sin []float32) {
	v := &vis.cfg.Vision
	d, hd := v.Dim, v.Dim/v.Heads
	half := hd / 2
	for i := range n {
		cs, sn := cos[i*half:(i+1)*half], sin[i*half:(i+1)*half]
		for part := range 2 { // q, k
			row := qkv[i*3*d+part*d : i*3*d+(part+1)*d]
			for h := range v.Heads {
				x := row[h*hd : (h+1)*hd]
				for j := range half {
					x1, x2 := x[j], x[half+j]
					x[j] = bf16c(float32(x1*cs[j]) - float32(x2*sn[j]))
					x[half+j] = bf16c(float32(x2*cs[j]) + float32(x1*sn[j]))
				}
			}
		}
	}
}

// attention is scaled_dot_product_attention over all n patches, per head:
// scores in float32 scaled by 1/√head, a softmax, the values' weighted sum
// divided by the softmax's, then bf16.
func (vis *Vision) attention(out, qkv []float32, n int) {
	v := &vis.cfg.Vision
	d, hd := v.Dim, v.Dim/v.Heads
	scale := float32(1 / math.Sqrt(float64(hd)))
	// Keys and values head by head, contiguous.
	ks, vs := make([]float32, n*d), make([]float32, n*d)
	for h := range v.Heads {
		for j := range n {
			copy(ks[(h*n+j)*hd:], qkv[j*3*d+d+h*hd:][:hd])
			copy(vs[(h*n+j)*hd:], qkv[j*3*d+2*d+h*hd:][:hd])
		}
	}
	jobs := make(chan int, v.Heads*n)
	for h := range v.Heads {
		for i := range n {
			jobs <- h*n + i
		}
	}
	close(jobs)
	var wg sync.WaitGroup
	for range min(runtime.GOMAXPROCS(0), v.Heads*n) {
		wg.Go(func() {
			s := make([]float32, n)
			acc := make([]float32, hd)
			for job := range jobs {
				h, i := job/n, job%n
				q := qkv[i*3*d+h*hd:][:hd]
				k, vv := ks[h*n*hd:(h+1)*n*hd], vs[h*n*hd:(h+1)*n*hd]
				m := float32(math.Inf(-1))
				for j := range n {
					s[j] = num.Dot32(q, k[j*hd:(j+1)*hd]) * scale
					m = max(m, s[j])
				}
				var sum float32
				clear(acc)
				for j := range n {
					e := num.Exp32(s[j] - m)
					sum += e
					pe := bf16c(e) // multiplied with the values in bf16
					for c, x := range vv[j*hd : (j+1)*hd] {
						acc[c] += float32(pe * x)
					}
				}
				o := out[i*d+h*hd:][:hd]
				inv := 1 / sum
				for c := range o {
					o[c] = bf16c(acc[c] * inv)
				}
			}
		})
	}
	wg.Wait()
}

// align is the Aligner: the rows×cols grid of encoder rows padded with
// zeros to multiples of Downsample, each Downsample² block unfolded
// (channel, then row, then column within the block), then
// bf16(w2·gelu(w1·x)).
func (vis *Vision) align(x []float32, rows, cols int) []float32 {
	feat, L := vis.unfold(x, rows, cols)
	dim := vis.cfg.Dim
	h := make([]float32, L*dim)
	vis.align1.apply(h, feat, L)
	for i, t := range h {
		h[i] = bf16c(gelu(t))
	}
	out := make([]float32, L*dim)
	vis.align2.apply(out, h, L)
	return out
}

// unfold is the aligner's input: L rows of Downsample² encoder rows each.
func (vis *Vision) unfold(x []float32, rows, cols int) ([]float32, int) {
	v := &vis.cfg.Vision
	r, d := v.Downsample, v.Dim
	br, bc := ceilDiv(rows, r), ceilDiv(cols, r)
	L, in := br*bc, d*r*r
	feat := make([]float32, L*in)
	for bi := range br {
		for bj := range bc {
			f := feat[(bi*bc+bj)*in:]
			for ki := range r {
				for kj := range r {
					pr, pc := bi*r+ki, bj*r+kj
					if pr >= rows || pc >= cols {
						continue
					}
					src := x[(pr*cols+pc)*d:]
					for c := range d {
						f[c*r*r+ki*r+kj] = src[c]
					}
				}
			}
		}
	}
	return feat, L
}

// bf16c rounds to bf16 after making a NaN canonical: processors and GPUs
// make NaNs of different bits.
func bf16c(x float32) float32 { return bf16(num.Canonical(x)) }

// gelu is PyTorch's exact GELU in float32: x·0.5·(1 + erf(x·√½)).
func gelu(x float32) float32 {
	e := float32(num.Erf(float64(float32(x * math.Sqrt2 / 2))))
	return float32(x*0.5) * (1 + e)
}

// Span is the inputs of an image span's positions (merge_image_embeddings):
// the delimiters' embeddings around the encoded rows, a line's end after
// each row of LLMW. emb is Encode's.
func (vis *Vision) Span(g ImageGrid, emb []float32) ([][]float32, error) {
	d := vis.cfg.Dim
	if len(emb) != g.LLMH*g.LLMW*d {
		return nil, fmt.Errorf("dsv41: %d encoded values for a %d×%d grid", len(emb), g.LLMH, g.LLMW)
	}
	rows := make([][]float32, 0, g.Tokens())
	rows = append(rows, vis.start)
	for i := range g.LLMH {
		for j := range g.LLMW {
			k := i*g.LLMW + j
			rows = append(rows, emb[k*d:(k+1)*d])
		}
		rows = append(rows, vis.newLine)
	}
	return append(rows, vis.end), nil
}
