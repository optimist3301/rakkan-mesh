package dsv41

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
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
	needMemory(t, 12)
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

// needMemory skips a test the machine has not the memory for: running
// out takes more down than the test (the OOM killer also takes the
// desktop session).
func needMemory(t *testing.T, gib int) {
	t.Helper()
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return
	}
	for _, line := range strings.Split(string(b), "\n") {
		var kb int64
		if _, err := fmt.Sscanf(line, "MemAvailable: %d kB", &kb); err == nil {
			if kb>>20 < int64(gib) {
				t.Skipf("%d GiB of memory available, the test needs %d", kb>>20, gib)
			}
			return
		}
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
// entries. The GPU runs the layer whole, then with its experts on the CPU,
// one after the other, so that each has the device to itself.
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
	// The layer on the host, its copy on the GPU (in RAM too for an
	// integrated one) and a part being uploaded.
	needMemory(t, 18)
	// Go would let the heap grow to twice the live layer before collecting;
	// layers come one after the other here.
	defer debug.SetMemoryLimit(debug.SetMemoryLimit(12 << 30))
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
			input := golden(g).f32(t, "input")
			decode := 0
			for k := range g {
				if strings.HasPrefix(k, "topk.decode") {
					decode++
				}
			}
			T := len(golden(g).ints(t, "tokens"))
			P := T - decode
			width := cfg.HCMult * cfg.Dim
			hashes := make([][]int64, len(cfg.EngramLayers))
			for i := range hashes {
				hashes[i] = make([]int64, (cfg.EngramMaxNgram-1)*cfg.EngramHeads)
			}
			mk := func(pos int) *Payload {
				prefill := 0
				if pos < P {
					prefill = P
				}
				p := &Payload{Prefill: prefill, H: slices.Clone(input[pos*width : (pos+1)*width]), PreMix: make([]float32, cfg.HCMult),
					Hashes: hashes, Source: sourceBefore(cfg, layer)}
				p.PreMix[0] = 1
				return p
			}
			per := func(d time.Duration) time.Duration { return (d / time.Duration(T)).Round(time.Millisecond) }

			cs := part.NewStage()
			want := make([][]byte, T)
			var elapsed time.Duration
			for pos := range T {
				p := mk(pos)
				start := time.Now()
				if err := cs.Forward(p, pos); err != nil {
					t.Fatal(err)
				}
				elapsed += time.Since(start)
				want[pos], _ = AppendPayload(nil, cfg, p)
			}
			t.Logf("CPU: %s per position", per(elapsed))
			for _, o := range []GPUOptions{{}, {HostExperts: true}} {
				gp, err := part.ToGPU(dev, o)
				if err != nil {
					t.Fatal(err)
				}
				gs, err := gp.NewStage(T)
				if err != nil {
					t.Fatal(err)
				}
				// The prompt in batches, the generation position by position.
				var prefill, decoding time.Duration
				for pos := 0; pos < T; {
					n := 1
					if pos < P {
						n = min(GPUBatch, P-pos)
					}
					ps := make([]*Payload, n)
					for j := range ps {
						ps[j] = mk(pos + j)
					}
					start := time.Now()
					if err := gs.ForwardBatch(ps, pos); err != nil {
						t.Fatal(err)
					}
					if pos < P {
						prefill += time.Since(start)
					} else {
						decoding += time.Since(start)
					}
					for j, p := range ps {
						if got, _ := AppendPayload(nil, cfg, p); !bytes.Equal(got, want[pos+j]) {
							ref, _, _ := DecodePayload(want[pos+j], cfg, T)
							t.Fatalf("host experts %v, position %d: %s", o.HostExperts, pos+j, payloadDiff(cfg, p, ref))
						}
					}
					pos += n
				}
				t.Logf("GPU (%s), host experts %v: %d positions bit for bit; prompt in batches of %d %s, generation %s per position; %d MiB on the GPU",
					dev.Info().Name, o.HostExperts, T, GPUBatch, (prefill / time.Duration(P)).Round(time.Millisecond),
					(decoding / time.Duration(T-P)).Round(time.Millisecond), gp.Bytes()>>20)
				gs.Close()
				gp.Free()
			}
			part, tensors, cs = nil, nil, nil
			runtime.GC()
			debug.FreeOSMemory()
		})
	}
}

// The image encoder on its published weights (shard 1), against the
// reference's outputs (tools/llmref/dsv41/vision_check.py): the picture
// prepared to the bit, the encoder to rounding, and on a GPU the CPU's
// bits.
func TestRealVision(t *testing.T) {
	dir := os.Getenv("RAKKAN_DSV41_REAL")
	if dir == "" {
		t.Skip("RAKKAN_DSV41_REAL not set")
	}
	needMemory(t, 8)
	defer debug.SetMemoryLimit(debug.SetMemoryLimit(6 << 30))
	gpath := filepath.Join(dir, "vision.safetensors")
	if _, err := os.Stat(gpath); err != nil {
		t.Skipf("no reference outputs: %v", err)
	}
	gt, err := ReadSafetensors(gpath)
	if err != nil {
		t.Fatal(err)
	}
	g := golden(gt)
	cfg, err := LoadConfig("../../../tools/llmref/dsv41/upstream/config.json")
	if err != nil {
		t.Fatal(err)
	}
	tensors, err := ReadSafetensors(filepath.Join(dir, "model-00001-of-00048.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	// The span's embeddings are in shard 2, beside the token embeddings.
	shard2, err := ReadSafetensors(filepath.Join(dir, "model-00002-of-00048.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"image_start", "image_end", "image_newline"} {
		x := *shard2[name]
		x.Data = bytes.Clone(x.Data)
		tensors[name] = &x
	}
	shard2 = nil
	runtime.GC()
	vis, err := NewVision(cfg, tensors)
	if err != nil {
		t.Fatal(err)
	}
	tensors = nil
	runtime.GC()

	img, err := cfg.PrepareImage(g["png"].Data)
	if err != nil {
		t.Fatal(err)
	}
	if grid := []int{img.VitH, img.VitW, img.LLMH, img.LLMW}; !slices.Equal(grid, g.ints(t, "grid")) {
		t.Fatalf("grid %v, reference %v", grid, g.ints(t, "grid"))
	}
	want := g.f32(t, "patches")
	for i := range want {
		if img.Patches[i] != want[i] {
			t.Fatalf("patch value %d: %v, reference %v", i, img.Patches[i], want[i])
		}
	}
	n := img.VitH * img.VitW
	vis.afterBlock = func(i int, x []float32) {
		if ref, ok := g[fmt.Sprintf("block%d", i)]; ok {
			var df diff
			r, _ := toF32("block", ref)
			df.add(x, r)
			t.Logf("block %2d: %s", i, df)
		}
	}
	start := time.Now()
	cpu, err := vis.Encode(img)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("CPU: %d patches in %v", n, time.Since(start))
	vis.afterBlock = nil
	var de diff
	de.add(cpu, g.f32(t, "embeds"))
	t.Logf("aligner:  %s", de)
	if de.maxL2 > 0.05 {
		t.Errorf("the encoder disagrees with the reference beyond rounding: %s", de)
	}
	for _, dev := range gpuDevices(t) {
		if strings.HasPrefix(dev.Info().Name, "llvmpipe") && len(gpuDevices(t)) > 1 {
			continue // slow, and the other device checks the kernels
		}
		gv, err := vis.ToGPU(dev)
		if err != nil {
			t.Fatal(err)
		}
		start := time.Now()
		got, err := gv.Encode(img)
		gv.Free()
		if err != nil {
			t.Fatal(err)
		}
		for i := range cpu {
			if math.Float32bits(got[i]) != math.Float32bits(cpu[i]) {
				t.Fatalf("%s: value %d: %v, CPU %v", dev.Info().Name, i, got[i], cpu[i])
			}
		}
		t.Logf("%s: the CPU's bits, in %v", dev.Info().Name, time.Since(start))
	}
}
