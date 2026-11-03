package dsv41

import (
	"bytes"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/optimist3301/rakkan-mesh/internal/gpu"
	"github.com/optimist3301/rakkan-mesh/internal/num"
)

func gpuDevices(t *testing.T) []*gpu.Device {
	t.Helper()
	infos, err := gpu.Devices()
	if err != nil || len(infos) == 0 {
		t.Skipf("no Vulkan devices: %v", err)
	}
	var out []*gpu.Device
	for _, info := range infos {
		d, err := gpu.Open(info)
		if err != nil {
			t.Fatalf("%s: %v", info.Name, err)
		}
		t.Cleanup(d.Close)
		out = append(out, d)
	}
	return out
}

// payloadDiff names the first field where two payloads differ.
func payloadDiff(c *Config, a, b *Payload) string {
	for i := range a.H {
		if math.Float32bits(a.H[i]) != math.Float32bits(b.H[i]) {
			return fmt.Sprintf("stream[%d] (copy %d, %d): GPU %v, CPU %v", i, i/c.Dim, i%c.Dim, a.H[i], b.H[i])
		}
	}
	for i := range a.PreMix {
		if math.Float32bits(a.PreMix[i]) != math.Float32bits(b.PreMix[i]) {
			return fmt.Sprintf("pre_mix[%d]: GPU %v, CPU %v", i, a.PreMix[i], b.PreMix[i])
		}
	}
	return fmt.Sprintf("shared state: GPU source %d kv %v idx %v topk %v cand %v; CPU source %d kv %v idx %v topk %v cand %v",
		a.Source, a.CompKV != nil, a.IndexK != nil, a.TopK, a.Candidates, b.Source, b.CompKV != nil, b.IndexK != nil, b.TopK, b.Candidates)
}

// A part's blocks on any GPU give the CPU's Payloads bit for bit, at every
// position of the prompt and of the generation, split anyhow.
func TestGPUStagesMatchCPU(t *testing.T) {
	cfg, tensors, tables := fixtureParts(t)
	head, err := NewHead(cfg, layerTensors(tensors, -1, -1), tables)
	if err != nil {
		t.Fatal(err)
	}
	_, g := loadFixture(t)
	tokens := g.ints(t, "tokens")
	prompt := promptLen
	for _, d := range gpuDevices(t) {
		for _, run := range []struct {
			cuts []int
			host bool
		}{{[]int{0, 1, 2, 3, 4, 5, 6, 7, 8}, false}, {[]int{0, 8}, false}, {[]int{0, 3, 6, 8}, false}, {[]int{0, 8}, true}, {[]int{0, 3, 6, 8}, true}} {
			cuts := run.cuts
			t.Run(fmt.Sprintf("%s/%v/host experts %v", d.Info().Name, cuts, run.host), func(t *testing.T) {
				var cpu []*Stage
				var gpus []*GPUStage
				for i := range len(cuts) - 1 {
					part, err := NewPart(cfg, layerTensors(tensors, cuts[i], cuts[i+1]), cuts[i], cuts[i+1], nil)
					if err != nil {
						t.Fatal(err)
					}
					cpu = append(cpu, part.NewStage())
					// The GPU's own copy of the part, its host weights let go
					// once uploaded (all but the experts kept on the host).
					gpart, err := NewPart(cfg, layerTensors(tensors, cuts[i], cuts[i+1]), cuts[i], cuts[i+1], nil)
					if err != nil {
						t.Fatal(err)
					}
					gp, err := gpart.ToGPU(d, GPUOptions{HostExperts: run.host})
					if err != nil {
						t.Fatal(err)
					}
					t.Cleanup(gp.Free)
					gp.DropHostWeights()
					st, err := gp.NewStage(len(tokens))
					if err != nil {
						t.Fatal(err)
					}
					t.Cleanup(st.Close)
					gpus = append(gpus, st)
				}
				for pos := range tokens {
					prefill := 0
					if pos < prompt {
						prefill = prompt
					}
					p, err := head.Embed(tokens[:pos+1], prefill)
					if err != nil {
						t.Fatal(err)
					}
					for i := range cpu {
						in, _ := AppendPayload(nil, cfg, p)
						gp, _, err := DecodePayload(in, cfg, len(tokens))
						if err != nil {
							t.Fatal(err)
						}
						if err := cpu[i].Forward(p, pos); err != nil {
							t.Fatal(err)
						}
						if err := gpus[i].Forward(gp, pos); err != nil {
							t.Fatalf("GPU layers %d..%d, position %d: %v", cuts[i], cuts[i+1], pos, err)
						}
						want, _ := AppendPayload(nil, cfg, p)
						got, err := AppendPayload(nil, cfg, gp)
						if err != nil {
							t.Fatal(err)
						}
						if !bytes.Equal(got, want) {
							t.Fatalf("layers %d..%d, position %d: %s", cuts[i], cuts[i+1], pos, payloadDiff(cfg, gp, p))
						}
					}
				}
			})
		}
	}
}

// awkwardVec mixes ordinary values with what the fixture's stay clear of:
// beyond the SwiGLU limit, huge and tiny, zeros of both signs, around
// softplus's threshold.
func awkwardVec(r *rand.Rand, n int, scale float64) []float32 {
	v := make([]float32, n)
	for i := range v {
		switch r.IntN(12) {
		case 0:
			v[i] = float32(r.NormFloat64() * 40)
		case 1:
			v[i] = float32(r.NormFloat64() * 1e6)
		case 2:
			v[i] = float32(r.NormFloat64() * 1e-6)
		case 3:
			v[i] = float32(math.Copysign(0, -1))
		case 4:
			v[i] = 20 + float32(r.IntN(3)-1)*float32(math.Ldexp(1, -19))
		default:
			v[i] = float32(r.NormFloat64() * scale)
		}
	}
	return v
}

// The elementwise kernels on inputs the fixture does not reach.
func TestGPUKernelsOnExtremes(t *testing.T) {
	r := rand.New(rand.NewPCG(21, 22))
	const n = 4096
	for _, d := range gpuDevices(t) {
		name := d.Info().Name
		a, err := d.NewArena(16*n, 8*n)
		if err != nil {
			t.Fatal(err)
		}
		run := func(in [][]float32, outs []int, ops func(b *gpu.Batch)) [][]float32 {
			b := a.Begin()
			for i, v := range in {
				b.Write(i*n, v)
			}
			ops(b)
			res := make([][]float32, len(outs))
			for i, o := range outs {
				res[i] = make([]float32, n)
				b.Read(o, res[i])
			}
			if err := b.Submit(); err != nil {
				t.Fatal(err)
			}
			return res
		}
		check := func(what string, got, want []float32) {
			t.Helper()
			for i := range want {
				if !d.Info().Denormals && isDenormal(want[i]) {
					continue // a device flushing denormals differs there
				}
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("%s %s [%d]: GPU %v, CPU %v", name, what, i, got[i], want[i])
				}
			}
		}
		for round := range 4 {
			g, u, w := awkwardVec(r, n, 3), awkwardVec(r, n, 3), awkwardVec(r, n, 1)
			// The expert activation, weighted, over 4 slots of n/4.
			res := run([][]float32{g, u, w[:4]}, []int{0}, func(b *gpu.Batch) { b.SiluMulDS(n/4, 0, n, 10, true, 2*n, 4, n/4) })
			want := make([]float32, n)
			for i := range want {
				gi, ui := min(g[i], 10), min(max(u[i], -10), 10)
				want[i] = bf16(float32(w[i/(n/4)] * float32(silu(gi)*ui)))
			}
			check(fmt.Sprintf("silu·mul round %d", round), res[0], want)

			// FP4 and FP8 quantization in place.
			for _, e4 := range []bool{false, true} {
				res = run([][]float32{g}, []int{0}, func(b *gpu.Batch) { b.FP4QuantInPlace(0, n, e4) })
				want = slices.Clone(g)
				kind := num.FP4ScaleE8M0
				if e4 {
					kind = num.FP4ScaleE4M3
				}
				num.FP4QuantInPlace(want, kind)
				check(fmt.Sprintf("FP4 (E4M3 scales %v)", e4), res[0], want)
			}
			res = run([][]float32{g}, []int{0}, func(b *gpu.Batch) { b.ActQuantInPlace(0, n) })
			want = slices.Clone(g)
			num.ActQuantInPlace(want)
			check("act_quant", res[0], want)

			// RMS normalization in 8 segments, and the gate's scores.
			res = run([][]float32{g, w}, []int{2 * n}, func(b *gpu.Batch) { b.DSNorm(0, 2*n, n, n/8, 8, n/8, 1e-20) })
			for s := range 8 {
				rmsNorm(want[s*n/8:(s+1)*n/8], g[s*n/8:(s+1)*n/8], w[:n/8], 1e-20)
			}
			check("rms_norm", res[0], want)
			for _, sig := range []bool{false, true} {
				res = run([][]float32{g, w}, []int{2 * n, 3 * n}, func(b *gpu.Batch) { b.MoEScores(n, 0, n, sig, 2*n, 3*n) })
				biased := make([]float32, n)
				for i, v := range g {
					if sig {
						want[i] = sigmoid(v)
					} else {
						want[i] = sqrt32(softplus(v))
					}
					biased[i] = want[i] + w[i]
				}
				check(fmt.Sprintf("gate scores (sigmoid %v)", sig), res[0], want)
				check("biased scores", res[1], biased)
			}
		}
		a.Free()
	}
}

func isDenormal(f float32) bool { b := math.Float32bits(f); return b>>23&0xff == 0 && b&0x7fffff != 0 }

// The GPU's selection is topK's, ties and signed zeros included, at sizes
// where ranking every pair would not do.
func TestGPUSelectMatchesTopK(t *testing.T) {
	r := rand.New(rand.NewPCG(31, 32))
	for _, d := range gpuDevices(t) {
		for _, n := range []int{1, 7, 300, 5000, 70000} {
			a, err := d.NewArena(3*n+16, 2*n+16)
			if err != nil {
				t.Fatal(err)
			}
			v := make([]float32, n)
			for i := range v {
				switch r.IntN(6) {
				case 0:
					v[i] = float32(math.Inf(-1))
				case 1:
					v[i] = float32(math.Copysign(0, -1))
				case 2:
					v[i] = 0
				default:
					v[i] = float32(r.IntN(50) - 25) // many ties
				}
			}
			for _, k := range []int{0, 1, min(5, n), min(512, n), n} {
				visible := n / 2
				b := a.Begin()
				b.Write(0, v)
				b.SelectSorted(n, k, 0, n, 2*n, visible)
				got := make([]float32, k)
				b.Read(n, got)
				if err := b.Submit(); err != nil {
					t.Fatal(err)
				}
				want := topK(v, k)
				slices.Sort(want)
				for i, x := range want {
					if x >= visible {
						want[i] = -1
					}
					if g := int(int32(math.Float32bits(got[i]))); g != want[i] {
						t.Fatalf("%s n=%d k=%d: entry %d: GPU %d, CPU %d", d.Info().Name, n, k, i, g, want[i])
					}
				}
			}
			a.Free()
		}
	}
}
