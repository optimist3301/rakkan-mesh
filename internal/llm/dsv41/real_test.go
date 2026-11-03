package dsv41

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/optimist3301/rakkan-mesh/internal/gpu"
)

// Real layers of DeepSeek-V4.1-Flash against the reference on real
// weights: the published shard of a layer, and the reference's outputs for
// it from tools/llmref/dsv41/layer_check.py. The files are large (6.9 GiB a
// layer), so the test runs only when RAKKAN_DSV41_REAL names the directory
// holding them: the shards as fetch.py stores them, and layer<N>.safetensors.
//
//	cd tools/llmref/dsv41 && python layer_check.py 2 $DIR/layer2.safetensors
//	RAKKAN_DSV41_REAL=$DIR go test -run RealLayers ./internal/llm/dsv41
//
// A whole block cannot agree bit for bit: where the reference leaves the
// order of operations unspecified (softmax sums and exp, float32 and bf16
// products) the last bit of some values differs, and the FP8 and FP4
// quantization before every product and of every cached key turns that
// into a different rounding step, a different block scale. So attention
// and MoE are each checked on the reference's own input, where they agree
// to rounding: nearly every cached value identical, the rest one FP4 step
// off; the same compressed positions selected except for a swap at the
// top-k boundary where such a step moved a score; the same experts. A
// mistake in the algorithm shows as errors of the order of the values
// themselves, and as selections that differ everywhere.
func TestRealLayers(t *testing.T) {
	dir := os.Getenv("RAKKAN_DSV41_REAL")
	if dir == "" {
		t.Skip("RAKKAN_DSV41_REAL not set")
	}
	cfg, err := LoadConfig("../../../tools/llmref/dsv41/upstream/config.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, layer := range []int{2, 20} {
		t.Run(fmt.Sprintf("layer%d", layer), func(t *testing.T) { checkRealLayer(t, cfg, dir, layer) })
	}
}

func checkRealLayer(t *testing.T, cfg *Config, dir string, layer int) {
	gpath := filepath.Join(dir, fmt.Sprintf("layer%d.safetensors", layer))
	if _, err := os.Stat(gpath); err != nil {
		t.Skipf("no reference outputs: %v", err)
	}
	gt, err := ReadSafetensors(gpath)
	if err != nil {
		t.Fatal(err)
	}
	g := golden(gt)
	tensors, err := ReadSafetensors(filepath.Join(dir, fmt.Sprintf("model-%05d-of-00048.safetensors", layer+3)))
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{Cfg: cfg, ropes: [2]*rope{newRope(cfg, false), newRope(cfg, true)}}
	b, err := m.loadBlock(&weights{t: tensors, used: map[string]bool{}}, layer)
	if err != nil {
		t.Fatal(err)
	}

	decode := 0
	for k := range g {
		if strings.HasPrefix(k, "topk.decode") {
			decode++
		}
	}
	T := len(g.ints(t, "tokens"))
	P := T - decode
	D, width := cfg.Dim, cfg.HCMult*cfg.Dim
	prefill := func(st *State, pos int) {
		st.prefillN = 0
		if pos < P {
			st.prefillN = P
		}
	}

	// Attention alone, on the reference's input.
	attnIn, attnRef := g.f32(t, "trace.attn_in"), g.f32(t, "trace.attn")
	topkPrefill := g.ints(t, "topk.prefill")
	k := len(topkPrefill) / P
	st := m.NewState()
	var attn, attnSameIdx diff
	idxBad, candBad := 0, 0
	for pos := range T {
		prefill(st, pos)
		var sh shared
		y := st.attention(b, attnIn[pos*D:(pos+1)*D], pos, &sh)
		attn.add(y, attnRef[pos*D:(pos+1)*D])
		same := true
		// Golden indices are offset by the window rows before them: the
		// prompt length when prefilling, the window after.
		want, offset := []int(nil), cfg.Window
		if pos < P {
			want, offset = slices.Clone(topkPrefill[pos*k:(pos+1)*k]), P
		} else {
			want = g.ints(t, fmt.Sprintf("topk.decode%d", pos-P))
		}
		for i := range want {
			if want[i] >= 0 {
				want[i] -= offset
			}
		}
		if !slices.Equal(sh.topk, want) {
			same = false
			idxBad++
			// The selections may differ only at the boundary: a few entries.
			if swapped := len(sh.topk) - common(sh.topk, want); swapped > k/100 {
				t.Errorf("position %d: %d of %d selected entries differ", pos, swapped, k)
			}
		}
		if same {
			attnSameIdx.add(y, attnRef[pos*D:(pos+1)*D])
		}
		if layer == cfg.CandidateSource {
			var c []int
			if pos < P {
				all := g.ints(t, "candidates.prefill")
				w := len(all) / P
				c = all[pos*w : (pos+1)*w]
			} else {
				c = g.ints(t, fmt.Sprintf("candidates.decode%d", pos-P))
			}
			got := make([]int, len(sh.candidates))
			for i, v := range sh.candidates {
				if v {
					got[i] = 1
				}
			}
			if !slices.Equal(got, c) {
				candBad++
			}
		}
	}
	t.Logf("attention: %s; selections differ at %d of %d positions", attn, idxBad, T)
	t.Logf("attention where the selections agree: %s", attnSameIdx)
	if idxBad > T/100 {
		t.Errorf("selections differ at %d of %d positions", idxBad, T)
	}
	if candBad > T/100 {
		t.Errorf("candidate pools differ at %d of %d positions", candBad, T)
	}
	if attnSameIdx.maxL2 > 0.03 {
		t.Errorf("attention's error exceeds rounding: %s", attnSameIdx)
	}
	for _, c := range []struct {
		name string
		got  [][]float32
	}{{"cache.compress_kv", st.compKV[layer]}, {"cache.index_k", st.indexK[layer]}} {
		ref := g.f32(t, c.name)
		if n := len(ref) / len(c.got[0]); n != len(c.got) {
			t.Errorf("%s: %d entries, reference %d", c.name, len(c.got), n)
			continue
		}
		var d diff
		for i, e := range c.got {
			d.add(e, ref[i*len(e):(i+1)*len(e)])
		}
		t.Logf("%s: %s", c.name, d)
		if d.n-d.same > d.n/10000 {
			t.Errorf("%s differs: %s", c.name, d)
		}
	}

	// MoE alone, on the reference's input.
	ffnIn, ffnRef := g.f32(t, "trace.ffn_in"), g.f32(t, "trace.ffn")
	expRef := g.ints(t, "trace.experts")
	active := cfg.ActiveExperts
	st = m.NewState()
	var ffn diff
	expBad := 0
	for pos := range T {
		var exp []int
		st.trace = func(name string, v []float32) {
			if name == "experts" {
				for _, e := range v {
					exp = append(exp, int(e))
				}
			}
		}
		y := st.moe(b.ffn, ffnIn[pos*D:(pos+1)*D])
		ffn.add(y, ffnRef[pos*D:(pos+1)*D])
		if !slices.Equal(exp, expRef[pos*active:(pos+1)*active]) {
			expBad++
		}
	}
	t.Logf("MoE: %s; experts differ at %d of %d positions", ffn, expBad, T)
	if expBad > T/200 {
		t.Errorf("experts differ at %d of %d positions", expBad, T)
	}
	if ffn.maxL2 > 0.01 {
		t.Errorf("MoE's error exceeds rounding: %s", ffn)
	}

	// The whole block, for the record: its own attention input and caches.
	input, out := g.f32(t, "input"), g.f32(t, "out")
	st = m.NewState()
	var whole diff
	for pos := range T {
		prefill(st, pos)
		h := slices.Clone(input[pos*width : (pos+1)*width])
		preMix := make([]float32, cfg.HCMult)
		preMix[0] = 1
		var sh shared
		st.block(b, h, preMix, pos, &sh)
		whole.add(h, out[pos*width:(pos+1)*width])
	}
	t.Logf("whole block: %s", whole)
	if whole.maxL2 > 0.5 {
		t.Errorf("the block's error exceeds what rounding compounds to: %s", whole)
	}
}

// common counts the entries two selections share.
func common(a, b []int) int {
	n := 0
	for _, x := range a {
		if x >= 0 && slices.Contains(b, x) {
			n++
		}
	}
	return n
}

// TestRealLayersGPU runs the real layers on a GPU (not a software one) and
// on the CPU, position by position on the reference's input, and wants
// the same bits: the stream, the mix, the selections and the cache
// entries.
func TestRealLayersGPU(t *testing.T) {
	dir := os.Getenv("RAKKAN_DSV41_REAL")
	if dir == "" {
		t.Skip("RAKKAN_DSV41_REAL not set")
	}
	var dev *gpu.Device
	for _, d := range gpuDevices(t) {
		if d.Info().Type != "cpu" {
			dev = d
			break
		}
	}
	if dev == nil {
		t.Skip("no hardware GPU")
	}
	cfg, err := LoadConfig("../../../tools/llmref/dsv41/upstream/config.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, layer := range []int{2, 20} {
		t.Run(fmt.Sprintf("layer%d", layer), func(t *testing.T) {
			g, err := ReadSafetensors(filepath.Join(dir, fmt.Sprintf("layer%d.safetensors", layer)))
			if err != nil {
				t.Skipf("no reference inputs: %v", err)
			}
			tensors, err := ReadSafetensors(filepath.Join(dir, fmt.Sprintf("model-%05d-of-00048.safetensors", layer+3)))
			if err != nil {
				t.Fatal(err)
			}
			part, err := NewPart(cfg, layerTensors(tensors, layer, layer+1), layer, layer+1, nil)
			if err != nil {
				t.Fatal(err)
			}
			gp, err := part.ToGPU(dev)
			if err != nil {
				t.Fatal(err)
			}
			defer gp.Free()
			input := golden(g).f32(t, "input")
			decode := 0
			for k := range g {
				if strings.HasPrefix(k, "topk.decode") {
					decode++
				}
			}
			T := len(golden(g).ints(t, "tokens"))
			P := T - decode
			gs, err := gp.NewStage(T)
			if err != nil {
				t.Fatal(err)
			}
			defer gs.Close()
			cs := part.NewStage()
			width := cfg.HCMult * cfg.Dim
			hashes := make([][]int64, len(cfg.EngramLayers))
			for i := range hashes {
				hashes[i] = make([]int64, (cfg.EngramMaxNgram-1)*cfg.EngramHeads)
			}
			var cpuTime, gpuTime time.Duration
			for pos := range T {
				prefill := 0
				if pos < P {
					prefill = P
				}
				mk := func() *Payload {
					p := &Payload{Prefill: prefill, H: slices.Clone(input[pos*width : (pos+1)*width]), PreMix: make([]float32, cfg.HCMult),
						Hashes: hashes, Source: sourceBefore(cfg, layer)}
					p.PreMix[0] = 1
					return p
				}
				pc, pg := mk(), mk()
				start := time.Now()
				if err := cs.Forward(pc, pos); err != nil {
					t.Fatal(err)
				}
				cpuTime += time.Since(start)
				start = time.Now()
				if err := gs.Forward(pg, pos); err != nil {
					t.Fatal(err)
				}
				gpuTime += time.Since(start)
				want, _ := AppendPayload(nil, cfg, pc)
				got, _ := AppendPayload(nil, cfg, pg)
				if !bytes.Equal(got, want) {
					t.Fatalf("position %d: %s", pos, payloadDiff(cfg, pg, pc))
				}
			}
			t.Logf("%d positions bit for bit; CPU %s, GPU (%s) %s per position", T, (cpuTime / time.Duration(T)).Round(time.Millisecond),
				dev.Info().Name, (gpuTime / time.Duration(T)).Round(time.Millisecond))
		})
	}
}
