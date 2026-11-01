package llm

import (
	"cmp"
	"math"
	"testing"

	"github.com/optimist3301/rakkan-mesh/internal/gpu"
)

// gpuDevices opens every Vulkan device, software ones included, or skips.
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
			t.Fatal(err)
		}
		t.Cleanup(d.Close)
		out = append(out, d)
	}
	return out
}

// A model's layers on a GPU give the CPU's bits, in every weight type and
// architecture variant: the whole layer runs there.
func TestGPUStagesMatchCPU(t *testing.T) {
	configs := []Config{
		{Dim: 64, Hidden: 128, Layers: 3, Heads: 4, KVHeads: 2, Vocab: 50, SeqLen: 96, QKVBias: true},
		{Dim: 64, Hidden: 96, Layers: 2, Heads: 4, KVHeads: 2, Vocab: 50, SeqLen: 96, HeadDim: 32, QKNorm: true, RopeTheta: 1e6, NormEps: 1e-6},
		{Dim: 96, Hidden: 160, Layers: 2, Heads: 3, KVHeads: 3, Vocab: 50, SeqLen: 96},
	}
	for _, d := range gpuDevices(t) {
		for ci, c := range configs {
			for _, dtype := range []string{"", "bf16", "q8_0", "q4_m"} {
				ck := randomModel(c, 9)
				if dtype != "" {
					if err := ck.Quantize(dtype); err != nil {
						t.Fatal(err)
					}
				}
				qc := ck.Config
				onGPU := randomModel(c, 9)
				if dtype != "" {
					onGPU.Quantize(dtype)
				}
				for _, l := range onGPU.Layers {
					if _, err := l.toDevice(qc, d); err != nil {
						t.Fatal(err)
					}
					if l.Wq.onHost() {
						t.Fatal("host copy kept")
					}
				}
				rope := NewRope(qc, qc.SeqLen)
				cpu, dev := NewStage(qc, ck.Layers, rope, 80), NewStage(qc, onGPU.Layers, rope, 80)
				if dev.gpu == nil {
					t.Fatalf("stage not on the GPU: %v", dev.gpuErr)
				}
				// A prompt longer than one submission takes, then a token
				// at a time.
				var prompt []int
				for i := range 70 {
					prompt = append(prompt, (i*7+3)%c.Vocab)
				}
				var a, b []float32
				for _, tok := range prompt {
					a = append(a, Embed(qc, ck.Embed, tok)...)
				}
				b = append([]float32(nil), a...)
				if err := cpu.ForwardMany(a, 0, len(prompt)); err != nil {
					t.Fatal(err)
				}
				if err := dev.ForwardMany(b, 0, len(prompt)); err != nil {
					t.Fatal(err)
				}
				same := func(pos int) {
					for i := range a {
						if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
							t.Fatalf("%s, config %d, %s, position %d, element %d: GPU %v, CPU %v", d.Info().Name, ci, cmp.Or(dtype, "f32"), pos, i, b[i], a[i])
						}
					}
				}
				same(0)
				for pos, tok := range []int{16, 25, 36, 49, 7, 3} {
					a, b = Embed(qc, ck.Embed, tok), Embed(qc, ck.Embed, tok)
					if err := cpu.Forward(a, len(prompt)+pos); err != nil {
						t.Fatal(err)
					}
					if err := dev.Forward(b, len(prompt)+pos); err != nil {
						t.Fatal(err)
					}
					same(len(prompt) + pos)
				}
				dev.Close()
				for _, l := range onGPU.Layers {
					l.release(qc)
				}
			}
		}
		if d.Used() != 0 {
			t.Fatalf("%s: %d bytes leaked", d.Info().Name, d.Used())
		}
		t.Logf("%s: stages match the CPU in every weight type", d.Info().Name)
	}
}

// A pipeline over nodes of different power: the GPU node takes the layers
// its memory holds, a small CPU node none, a large CPU node the rest; the
// text is exactly the single-process one.
func TestHeterogeneousPipeline(t *testing.T) {
	devs := gpuDevices(t)
	var dev *gpu.Device
	for _, d := range devs {
		if d.Info().Type != "cpu" || dev == nil {
			dev = d
		}
	}
	c := newCluster(t)
	coord := c.add(false)
	ck, tokData, tok := talkative(t, "abc", 18)
	ck.Config.SeqLen = 64
	lb := LayerBytes(ck.Config)
	cb := CacheBytes(ck.Config, 1, 64)
	fast := c.addOpts(nodeOpts{serve: true, dev: dev, devBytes: 2*lb + lb/2, mem: 1 << 30, speed: 50})
	tiny := c.addOpts(nodeOpts{serve: true, mem: (lb + cb) / 2, speed: 20})
	big := c.addOpts(nodeOpts{serve: true, mem: 1 << 30, speed: 5})
	model, _, err := Import(coord.blobs.Store(), ck, tokData, "synthetic")
	if err != nil {
		t.Fatal(err)
	}
	waitServers(t, coord, 3)
	want := localTokens(ck, tok, "abc", 18)

	got, stats := generate(t, coord, GenerateRequest{Model: model, Prompt: "abc", MaxTokens: 18}, nil)
	if !equalTokens(got, want) {
		t.Fatalf("tokens %v, local %v", got, want)
	}
	t.Logf("stages: %+v", stats.Stages)
	if len(stats.Stages) != 2 || stats.Stages[0].Node != fast.node.ID() || stats.Stages[0].To != 2 || stats.Stages[1].Node != big.node.ID() {
		t.Fatalf("plan %+v: want layers 0-1 on the GPU node, the rest on the large CPU node", stats.Stages)
	}
	for _, st := range stats.Stages {
		if st.Node == tiny.node.ID() {
			t.Fatal("a node too small for a layer got one")
		}
	}
	if fast.llm.cfg.Device.Used() == 0 {
		t.Fatal("the GPU node's layers are not on its GPU")
	}
}
