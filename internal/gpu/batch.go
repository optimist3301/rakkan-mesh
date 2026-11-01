package gpu

import (
	"errors"
	"fmt"
	"unsafe"
)

// Arena is device memory for a computation: activations, caches and
// tables, addressed in 32-bit words. Values move between it and the host
// through a host-visible staging buffer.
type Arena struct {
	d            *Device
	buf, staging *buffer
	set          uint64
	words        int
	qxOff, xsOff int // scratch for quantized inputs; -1 if none
	stagingWords int
	bytes        int64
}

// NewArena allocates an arena of words, and staging for stagingWords
// moved per submission.
func (d *Device) NewArena(words, stagingWords int) (*Arena, error) {
	a := &Arena{d: d, words: words, stagingWords: stagingWords, qxOff: -1, xsOff: -1}
	if uint64(4*words) > d.info.maxRange {
		return nil, fmt.Errorf("gpu: arena of %d MiB exceeds the device's binding limit of %d MiB", 4*words>>20, d.info.maxRange>>20)
	}
	var err error
	if a.buf, err = d.newBuffer(uint64(4*max(words, 1)), memDeviceLocal); err != nil {
		return nil, err
	}
	if a.staging, err = d.newBuffer(uint64(4*max(stagingWords, 1)), memHostVisible|memHostCoherent); err != nil {
		d.freeBuffer(a.buf)
		return nil, err
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if a.set, err = d.newSet(d.setPool, 1, a.buf); err != nil {
		d.freeBuffer(a.buf)
		d.freeBuffer(a.staging)
		return nil, err
	}
	a.bytes = int64(a.buf.size)
	if d.host != nil {
		d.used += a.bytes
	}
	return a, nil
}

// SetQuantScratch reserves room for an input quantized to Q8_0: the bytes
// at word qxOff (a quarter of the input's length) and the scales at word
// xsOff (a thirty-second).
func (a *Arena) SetQuantScratch(qxOff, xsOff int) { a.qxOff, a.xsOff = qxOff, xsOff }

// Bytes is the device memory the arena holds.
func (a *Arena) Bytes() int64 { return a.bytes }

// Free releases the arena.
func (a *Arena) Free() {
	d := a.d
	d.mu.Lock()
	defer d.mu.Unlock()
	vk.FreeDescriptorSets(d.dev, d.setPool, 1, &a.set)
	d.freeBuffer(a.buf)
	d.freeBuffer(a.staging)
	if a != d.host {
		d.used -= a.bytes
	}
}

// Vectors is a buffer of floats on a device, such as a layer's norm
// weights and biases, read by kernels from descriptor set 0.
type Vectors struct {
	d     *Device
	buf   *buffer
	set   uint64
	bytes int64
}

// UploadVectors copies floats to the device.
func (d *Device) UploadVectors(v []float32) (*Vectors, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	buf, err := d.upload(floatBytes(v))
	if err != nil {
		return nil, err
	}
	// The layout of set 0 has two bindings; vectors fill both.
	set, err := d.newSet(d.setPool, 0, buf, buf)
	if err != nil {
		d.freeBuffer(buf)
		return nil, err
	}
	vs := &Vectors{d: d, buf: buf, set: set, bytes: int64(buf.size)}
	d.used += vs.bytes
	return vs, nil
}

// Free releases the vectors.
func (v *Vectors) Free() {
	d := v.d
	d.mu.Lock()
	defer d.mu.Unlock()
	vk.FreeDescriptorSets(d.dev, d.setPool, 1, &v.set)
	d.freeBuffer(v.buf)
	d.used -= v.bytes
}

// Batch records a submission of operations on an arena. It holds the
// device from Begin to Submit; operations run in order.
type Batch struct {
	a      *Arena
	cb     uintptr
	staged int
	reads  []pendingRead
	err    error
}

type pendingRead struct {
	dst []float32
	at  int // in staging, in words
}

// Begin starts a batch on the arena.
func (a *Arena) Begin() *Batch {
	d := a.d
	d.mu.Lock()
	b := &Batch{a: a, cb: d.cmd}
	begin := vkCommandBufferBeginInfo{sType: stCommandBufferBegin, flags: commandBufferOneTimeOnly}
	b.err = check("begin commands", vk.BeginCommandBuffer(d.cmd, &begin))
	return b
}

func (b *Batch) stage(n int) (int, bool) {
	if b.err != nil {
		return 0, false
	}
	if b.staged+n > b.a.stagingWords {
		b.err = errors.New("gpu: batch moves more than its staging holds")
		return 0, false
	}
	at := b.staged
	b.staged += n
	return at, true
}

// Write copies data into the arena at word off.
func (b *Batch) Write(off int, data []float32) {
	at, ok := b.stage(len(data))
	if !ok || len(data) == 0 {
		return
	}
	copy(unsafe.Slice((*float32)(b.a.staging.mapped), b.a.stagingWords)[at:], data)
	region := vkBufferCopy{srcOffset: uint64(4 * at), dstOffset: uint64(4 * off), size: uint64(4 * len(data))}
	vk.CmdCopyBuffer(b.cb, b.a.staging.buf, b.a.buf.buf, 1, &region)
	barrier(b.cb)
}

// Read copies len(dst) words at off into dst when the batch is submitted.
func (b *Batch) Read(off int, dst []float32) {
	at, ok := b.stage(len(dst))
	if !ok || len(dst) == 0 {
		return
	}
	region := vkBufferCopy{srcOffset: uint64(4 * off), dstOffset: uint64(4 * at), size: uint64(4 * len(dst))}
	vk.CmdCopyBuffer(b.cb, b.a.buf.buf, b.a.staging.buf, 1, &region)
	barrier(b.cb)
	b.reads = append(b.reads, pendingRead{dst, at})
}

// dispatch runs a kernel over gx×gy workgroups, with set0 (0 for none)
// bound as descriptor set 0.
func (b *Batch) dispatch(k kernelKind, set0 uint64, gx, gy int, push ...uint32) {
	if b.err != nil || gx == 0 || gy == 0 {
		return
	}
	d := b.a.d
	vk.CmdBindPipeline(b.cb, bindPointCompute, d.kernels[k].pipe)
	if set0 != 0 {
		vk.CmdBindDescriptorSets(b.cb, bindPointCompute, d.layout, 0, 1, &set0, 0, nil)
	}
	set1 := b.a.set
	vk.CmdBindDescriptorSets(b.cb, bindPointCompute, d.layout, 1, 1, &set1, 0, nil)
	var pc [pushWords]uint32
	copy(pc[:], push)
	vk.CmdPushConstants(b.cb, d.layout, stageCompute, 0, 4*pushWords, unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(b.cb, uint32(gx), uint32(gy), 1)
	barrier(b.cb)
}

func groups(n, per int) int { return (n + per - 1) / per }

// MatVecs computes ms[i]·x into word outOffs[i], x at word inOff; an
// input for quantized matrices is quantized once, into the scratch.
func (b *Batch) MatVecs(ms []*Matrix, outOffs []int, inOff int) {
	quantized := false
	for _, m := range ms {
		if m.d != b.a.d {
			b.err = errors.New("gpu: matrix on another device")
			return
		}
		if (m.kind == kindQ8 || m.kind == kindQ4) && !quantized {
			if b.a.qxOff < 0 {
				b.err = errors.New("gpu: arena has no scratch for quantized inputs")
				return
			}
			b.Quantize(inOff, m.cols)
			quantized = true
		}
	}
	for i, m := range ms {
		for _, pt := range m.parts {
			per := groupRows(m.kind)
			g := groups(pt.rows, per)
			gx := min(g, 65535)
			gy := groups(g, gx)
			out := uint32(outOffs[i] + pt.row0)
			if m.kind == kindQ8 || m.kind == kindQ4 {
				b.dispatch(m.kind, pt.set, gx, gy, uint32(pt.rows), uint32(m.cols), uint32(gx), out, uint32(b.a.qxOff), uint32(b.a.xsOff))
			} else {
				b.dispatch(m.kind, pt.set, gx, gy, uint32(pt.rows), uint32(m.cols), uint32(gx), out, uint32(inOff))
			}
		}
	}
}

// Quantize quantizes n floats at word in into the arena's scratch.
func (b *Batch) Quantize(in, n int) {
	b.dispatch(kindQuantize, 0, groups(n/32, flatGroup), 1, uint32(n), uint32(in), uint32(b.a.qxOff), uint32(b.a.xsOff))
}

// RMSNorm normalizes segs segments of n floats at in (stride apart) into
// out, with weights at word w of vecs: out = w·(rsqrt(dot(x,x)·recipN +
// eps)·x).
func (b *Batch) RMSNorm(vecs *Vectors, in, out, w, n, segs, stride int, recipN, eps float32) {
	b.dispatch(kindRMSNorm, vecs.set, segs, 1, uint32(n), uint32(in), uint32(out), uint32(w), f32bits(recipN), f32bits(eps), uint32(stride))
}

// AddVec adds n floats at word v of vecs to the arena at out.
func (b *Batch) AddVec(vecs *Vectors, out, v, n int) {
	b.dispatch(kindAddVec, vecs.set, groups(n, flatGroup), 1, uint32(n), uint32(out), uint32(v))
}

// Rope rotates adjacent pairs of the query heads at q (qDim floats) and
// key heads at k (kvDim floats), with the position's cos and sin rows.
func (b *Batch) Rope(q, k, qDim, kvDim, headSize, cos, sin int) {
	b.dispatch(kindRope, 0, groups(qDim/2, flatGroup), 1, uint32(qDim/2), uint32(q), uint32(k), uint32(kvDim), uint32(headSize), uint32(cos), uint32(sin))
}

// Copy copies n words from src to dst.
func (b *Batch) Copy(src, dst, n int) {
	b.dispatch(kindCopy, 0, groups(n, flatGroup), 1, uint32(n), uint32(src), uint32(dst))
}

// Attention computes attention for one position: scores of the query at q
// against keys at keys (positions rows, kvDim apart) into att (heads rows
// of attStride), softmax, and the values at vals weighted into out.
func (b *Batch) Attention(q, keys, vals, att, attStride, out, positions, heads, kvHeads, headSize, kvDim int, invSqrt float32) {
	kvMul := uint32(heads / kvHeads)
	b.dispatch(kindScores, 0, groups(positions, rowsPerGroup), heads, uint32(positions), uint32(headSize), uint32(keys),
		uint32(kvDim), kvMul, uint32(q), uint32(att), uint32(attStride), f32bits(invSqrt))
	b.dispatch(kindSoftmax, 0, heads, 1, uint32(positions), uint32(att), uint32(attStride))
	qDim := heads * headSize
	b.dispatch(kindWSum, 0, groups(qDim, flatGroup), 1, uint32(positions), uint32(headSize), uint32(vals), uint32(kvDim), kvMul,
		uint32(att), uint32(attStride), uint32(out), uint32(qDim))
}

// Residual adds n floats at y to x, replacing NaNs by the canonical one if
// canonical is set.
func (b *Batch) Residual(x, y, n int, canonical bool) {
	c := uint32(0)
	if canonical {
		c = 1
	}
	b.dispatch(kindResidual, 0, groups(n, flatGroup), 1, uint32(n), uint32(x), uint32(y), c)
}

// SiluMul sets n floats at a to silu(a)·b.
func (b *Batch) SiluMul(a, bOff, n int) {
	b.dispatch(kindSiluMul, 0, groups(n, flatGroup), 1, uint32(n), uint32(a), uint32(bOff))
}

// Submit runs the batch, waits, delivers the reads and releases the device.
func (b *Batch) Submit() error {
	d := b.a.d
	defer d.mu.Unlock()
	if b.err != nil {
		vk.EndCommandBuffer(b.cb)
		return b.err
	}
	if err := d.submit(); err != nil {
		return err
	}
	staged := unsafe.Slice((*float32)(b.a.staging.mapped), b.a.stagingWords)
	for _, r := range b.reads {
		copy(r.dst, staged[r.at:r.at+len(r.dst)])
	}
	return nil
}

func f32bits(f float32) uint32 { return *(*uint32)(unsafe.Pointer(&f)) }
