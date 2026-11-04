package gpu

import (
	"fmt"
	"slices"
	"time"
)

// Profiling, for working on the kernels' speed: while a profile runs,
// every dispatch is submitted on its own and timed from the host, so the
// time of each kind of kernel adds up (with a submission's cost each,
// tens of microseconds: small kernels look slower than they are).

var kindNames = map[kernelKind]string{
	kindF32:        "F32",
	kindBF16:       "BF16",
	kindQ8:         "Q8",
	kindQ4:         "Q4",
	kindQuantize:   "Quantize",
	kindRMSNorm:    "RMSNorm",
	kindAddVec:     "AddVec",
	kindRope:       "Rope",
	kindCopy:       "Copy",
	kindScores:     "Scores",
	kindSoftmax:    "Softmax",
	kindWSum:       "WSum",
	kindResidual:   "Residual",
	kindSiluMul:    "SiluMul",
	kindFP8:        "FP8",
	kindFP4:        "FP4",
	kindActFP8:     "ActFP8",
	kindMathTest:   "MathTest",
	kindDSNorm:     "DSNorm",
	kindHCSplit:    "HCSplit",
	kindHCPre:      "HCPre",
	kindHCPost:     "HCPost",
	kindRopeDS:     "RopeDS",
	kindActQuant:   "ActQuant",
	kindFP4Quant:   "FP4Quant",
	kindRoundScale: "RoundScale",
	kindCompress:   "Compress",
	kindIdxScores:  "IdxScores",
	kindCandPool:   "CandPool",
	kindTopK:       "TopK",
	kindSparseAttn: "SparseAttn",
	kindMoEScore:   "MoEScore",
	kindMoEWeights: "MoEWeights",
	kindFP4Experts: "FP4Experts",
	kindSiluMulDS:  "SiluMulDS",
	kindMoESum:     "MoESum",
	kindEngramGate: "EngramGate",
	kindSelect:     "Select",
	kindCompact:    "Compact",
	kindFP8Batch:   "FP8Batch",
	kindHCMean:     "HCMean",
	kindVMatMul:    "VMatMul",
	kindVAttn16:    "VAttn16",
	kindVAttn32:    "VAttn32",
	kindVAttn64:    "VAttn64",
	kindVRope:      "VRope",
	kindVAdd:       "VAdd",
	kindVSiluMul:   "VSiluMul",
	kindPoll:       "Copies",
}

// kindPoll stands for the copies and other commands before a dispatch.
const kindPoll = kindAllCount

// ProfileEntry is the time a kind of kernel took.
type ProfileEntry struct {
	Kernel string
	Calls  int
	Time   time.Duration
}

type profile struct {
	calls map[kernelKind]int
	time  map[kernelKind]time.Duration
}

// StartProfile starts timing the device's kernels.
func (d *Device) StartProfile() {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.prof = &profile{calls: map[kernelKind]int{}, time: map[kernelKind]time.Duration{}}
}

// StopProfile ends the profile and returns its entries, the slowest first.
func (d *Device) StopProfile() []ProfileEntry {
	d.mu.Lock()
	defer d.mu.Unlock()
	p := d.prof
	d.prof = nil
	if p == nil {
		return nil
	}
	var out []ProfileEntry
	for k, t := range p.time {
		name := kindNames[k]
		if name == "" {
			name = fmt.Sprint(int(k))
		}
		out = append(out, ProfileEntry{name, p.calls[k], t})
	}
	slices.SortFunc(out, func(a, b ProfileEntry) int { return int(b.Time - a.Time) })
	return out
}

// profiled runs the commands recorded so far, then record (one dispatch)
// alone, timing each; the batch then goes on recording.
func (b *Batch) profiled(k kernelKind, record func()) {
	d := b.a.d
	run := func(kind kernelKind) {
		start := time.Now()
		if err := d.submit(); err != nil {
			b.err = err
			return
		}
		d.prof.time[kind] += time.Since(start)
		d.prof.calls[kind]++
		begin := vkCommandBufferBeginInfo{sType: stCommandBufferBegin, flags: commandBufferOneTimeOnly}
		b.err = check("begin commands", vk.BeginCommandBuffer(b.cb, &begin))
	}
	run(kindPoll)
	if b.err != nil {
		return
	}
	record()
	run(k)
}
