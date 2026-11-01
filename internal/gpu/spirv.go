package gpu

import (
	"encoding/binary"
	"math"
)

// A minimal SPIR-V builder for compute shaders, so kernels need no shader
// compiler at build time. It covers what the matrix kernels use: storage
// buffers of 32-bit words, push constants, workgroup arrays, the local and
// workgroup IDs, integer and float arithmetic, structured loops and
// branches, and workgroup barriers. Every float multiply and add is
// decorated NoContraction, so drivers may not fuse them into FMAs: rounding
// happens where package num rounds.

const (
	opExtension          = 10
	opMemoryModel        = 14
	opEntryPoint         = 15
	opExecutionMode      = 16
	opCapability         = 17
	opTypeVoid           = 19
	opTypeBool           = 20
	opTypeInt            = 21
	opTypeFloat          = 22
	opTypeVector         = 23
	opTypeArray          = 28
	opTypeRuntimeArray   = 29
	opTypeStruct         = 30
	opTypePointer        = 32
	opTypeFunction       = 33
	opConstant           = 43
	opFunction           = 54
	opFunctionEnd        = 56
	opVariable           = 59
	opLoad               = 61
	opStore              = 62
	opAccessChain        = 65
	opDecorate           = 71
	opMemberDecorate     = 72
	opCompositeExtract   = 81
	opConvertSToF        = 111
	opBitcast            = 124
	opIAdd               = 128
	opFAdd               = 129
	opISub               = 130
	opIMul               = 132
	opFMul               = 133
	opLogicalAnd         = 167
	opSelect             = 169
	opIEqual             = 170
	opULessThan          = 176
	opShiftRightLogical  = 194
	opShiftLeftLogical   = 196
	opBitwiseAnd         = 199
	opBitFieldSExtract   = 202
	opControlBarrier     = 224
	opLoopMerge          = 246
	opSelectionMerge     = 247
	opLabel              = 248
	opBranch             = 249
	opBranchConditional  = 250
	opReturn             = 253
	decBlock             = 2
	decArrayStride       = 6
	decBuiltIn           = 11
	decNonWritable       = 24
	decBinding           = 33
	decDescriptorSet     = 34
	decOffset            = 35
	decNoContraction     = 42
	builtInWorkgroupID   = 26
	builtInLocalID       = 27
	scInput              = 1
	scWorkgroup          = 4
	scFunction           = 7
	scPushConstant       = 9
	scStorageBuffer      = 12
	capShader            = 1
	capDenormPreserve    = 4464
	capSignedZeroInfNan  = 4466
	capRoundingModeRTE   = 4467
	emLocalSize          = 17
	emDenormPreserve     = 4459
	emSignedZeroInfNan   = 4461
	emRoundingModeRTE    = 4462
	scopeWorkgroup       = 2
	semAcqRelWorkgroupMs = 0x8 | 0x100
)

// floatModes are the IEEE behaviours a device lets shaders request.
type floatModes struct {
	denormPreserve, signedZeroInfNan, roundRTE bool
}

type id = uint32

type builder struct {
	next                         id
	caps, exts, header, annots   []uint32
	types, body, vars            []uint32
	entryIfaces                  []id
	modes                        floatModes
	localX, localY               uint32
	tVoid, tBool, tU32, tI32     id
	tF32, tV3, tFn               id
	pSBu, pPCu, pFnU, pFnF, pFnI id
	pWGf, pInV3                  id
	consts                       map[[2]uint32]id
	main                         id
}

func newBuilder(modes floatModes, localX, localY uint32) *builder {
	b := &builder{next: 1, modes: modes, localX: localX, localY: localY, consts: map[[2]uint32]id{}}
	b.tVoid = b.typ(opTypeVoid)
	b.tBool = b.typ(opTypeBool)
	b.tU32 = b.typ(opTypeInt, 32, 0)
	b.tI32 = b.typ(opTypeInt, 32, 1)
	b.tF32 = b.typ(opTypeFloat, 32)
	b.tV3 = b.typ(opTypeVector, b.tU32, 3)
	b.tFn = b.typ(opTypeFunction, b.tVoid)
	b.pSBu = b.typ(opTypePointer, scStorageBuffer, b.tU32)
	b.pPCu = b.typ(opTypePointer, scPushConstant, b.tU32)
	b.pFnU = b.typ(opTypePointer, scFunction, b.tU32)
	b.pFnF = b.typ(opTypePointer, scFunction, b.tF32)
	b.pFnI = b.typ(opTypePointer, scFunction, b.tI32)
	b.pWGf = b.typ(opTypePointer, scWorkgroup, b.tF32)
	b.pInV3 = b.typ(opTypePointer, scInput, b.tV3)
	b.main = b.id()
	return b
}

func (b *builder) id() id { b.next++; return b.next - 1 }

func inst(dst *[]uint32, op uint32, args ...uint32) {
	*dst = append(*dst, uint32(len(args)+1)<<16|op)
	*dst = append(*dst, args...)
}

func strWords(s string) []uint32 {
	bs := append([]byte(s), 0)
	for len(bs)%4 != 0 {
		bs = append(bs, 0)
	}
	w := make([]uint32, len(bs)/4)
	for i := range w {
		w[i] = binary.LittleEndian.Uint32(bs[4*i:])
	}
	return w
}

func (b *builder) typ(op uint32, args ...uint32) id {
	r := b.id()
	inst(&b.types, op, append([]uint32{r}, args...)...)
	return r
}

func (b *builder) constant(t id, bits uint32) id {
	key := [2]uint32{t, bits}
	if c, ok := b.consts[key]; ok {
		return c
	}
	c := b.id()
	inst(&b.types, opConstant, t, c, bits)
	b.consts[key] = c
	return c
}

func (b *builder) u32(v uint32) id  { return b.constant(b.tU32, v) }
func (b *builder) i32(v int32) id   { return b.constant(b.tI32, uint32(v)) }
func (b *builder) f32(v float32) id { return b.constant(b.tF32, math.Float32bits(v)) }
func (b *builder) decorate(t id, args ...uint32) {
	inst(&b.annots, opDecorate, append([]uint32{t}, args...)...)
}

// storageBuffer declares binding n of set 0: a runtime array of words.
func (b *builder) storageBuffer(binding uint32, readOnly bool) id {
	arr := b.typ(opTypeRuntimeArray, b.tU32)
	b.decorate(arr, decArrayStride, 4)
	st := b.typ(opTypeStruct, arr)
	b.decorate(st, decBlock)
	inst(&b.annots, opMemberDecorate, st, 0, decOffset, 0)
	if readOnly {
		inst(&b.annots, opMemberDecorate, st, 0, decNonWritable)
	}
	ptr := b.typ(opTypePointer, scStorageBuffer, st)
	v := b.id()
	inst(&b.types, opVariable, ptr, v, scStorageBuffer)
	b.decorate(v, decDescriptorSet, 0)
	b.decorate(v, decBinding, binding)
	return v
}

// pushConstants declares n words of push constants.
func (b *builder) pushConstants(n int) id {
	members := make([]uint32, n)
	for i := range members {
		members[i] = b.tU32
	}
	st := b.typ(opTypeStruct, members...)
	b.decorate(st, decBlock)
	for i := range n {
		inst(&b.annots, opMemberDecorate, st, uint32(i), decOffset, uint32(4*i))
	}
	ptr := b.typ(opTypePointer, scPushConstant, st)
	v := b.id()
	inst(&b.types, opVariable, ptr, v, scPushConstant)
	return v
}

// workgroupFloats declares a shared array of n floats.
func (b *builder) workgroupFloats(n uint32) id {
	arr := b.typ(opTypeArray, b.tF32, b.u32(n))
	ptr := b.typ(opTypePointer, scWorkgroup, arr)
	v := b.id()
	inst(&b.types, opVariable, ptr, v, scWorkgroup)
	return v
}

func (b *builder) builtin(which uint32) id {
	v := b.id()
	inst(&b.types, opVariable, b.pInV3, v, scInput)
	b.decorate(v, decBuiltIn, which)
	b.entryIfaces = append(b.entryIfaces, v)
	return v
}

// local declares a function variable of type t (tU32, tI32 or tF32).
func (b *builder) local(t id) id {
	ptr := map[id]id{b.tU32: b.pFnU, b.tF32: b.pFnF, b.tI32: b.pFnI}[t]
	v := b.id()
	inst(&b.vars, opVariable, ptr, v, scFunction)
	return v
}

func (b *builder) op(op uint32, t id, args ...uint32) id {
	r := b.id()
	inst(&b.body, op, append([]uint32{t, r}, args...)...)
	return r
}

func (b *builder) load(t, ptr id) id { return b.op(opLoad, t, ptr) }
func (b *builder) store(ptr, v id)   { inst(&b.body, opStore, ptr, v) }
func (b *builder) iadd(x, y id) id   { return b.op(opIAdd, b.tU32, x, y) }
func (b *builder) imul(x, y id) id   { return b.op(opIMul, b.tU32, x, y) }
func (b *builder) shr(x, n id) id    { return b.op(opShiftRightLogical, b.tU32, x, n) }
func (b *builder) shl(x, n id) id    { return b.op(opShiftLeftLogical, b.tU32, x, n) }
func (b *builder) and(x, y id) id    { return b.op(opBitwiseAnd, b.tU32, x, y) }
func (b *builder) ult(x, y id) id    { return b.op(opULessThan, b.tBool, x, y) }
func (b *builder) ieq(x, y id) id    { return b.op(opIEqual, b.tBool, x, y) }
func (b *builder) land(x, y id) id   { return b.op(opLogicalAnd, b.tBool, x, y) }
func (b *builder) toF(bits id) id    { return b.op(opBitcast, b.tF32, bits) }
func (b *builder) toU(f id) id       { return b.op(opBitcast, b.tU32, f) }
func (b *builder) toI(u id) id       { return b.op(opBitcast, b.tI32, u) }
func (b *builder) sToF(i id) id      { return b.op(opConvertSToF, b.tF32, i) }
func (b *builder) iaddS(x, y id) id  { return b.op(opIAdd, b.tI32, x, y) }
func (b *builder) isubS(x, y id) id  { return b.op(opISub, b.tI32, x, y) }
func (b *builder) imulS(x, y id) id  { return b.op(opIMul, b.tI32, x, y) }

// sext extracts a signed byte: bits [off, off+8) of a signed word.
func (b *builder) sext(word id, off uint32) id {
	return b.op(opBitFieldSExtract, b.tI32, word, b.u32(off), b.u32(8))
}

func (b *builder) fmul(x, y id) id {
	r := b.op(opFMul, b.tF32, x, y)
	b.decorate(r, decNoContraction)
	return r
}

func (b *builder) fadd(x, y id) id {
	r := b.op(opFAdd, b.tF32, x, y)
	b.decorate(r, decNoContraction)
	return r
}

func (b *builder) sel(c, x, y id) id { return b.op(opSelect, b.tU32, c, x, y) }

// word returns element i of a storage buffer.
func (b *builder) word(buf, i id) id {
	return b.load(b.tU32, b.op(opAccessChain, b.pSBu, buf, b.u32(0), i))
}

func (b *builder) setWord(buf, i, v id) {
	b.store(b.op(opAccessChain, b.pSBu, buf, b.u32(0), i), v)
}

func (b *builder) push(pc id, i uint32) id {
	return b.load(b.tU32, b.op(opAccessChain, b.pPCu, pc, b.u32(i)))
}

func (b *builder) shared(arr, i id) id { return b.op(opAccessChain, b.pWGf, arr, i) }

func (b *builder) component(v id, i uint32) id { return b.op(opCompositeExtract, b.tU32, v, i) }

func (b *builder) label(l id) { inst(&b.body, opLabel, l) }

// ifThen runs then when c holds.
func (b *builder) ifThen(c id, then func()) {
	t, m := b.id(), b.id()
	inst(&b.body, opSelectionMerge, m, 0)
	inst(&b.body, opBranchConditional, c, t, m)
	b.label(t)
	then()
	inst(&b.body, opBranch, m)
	b.label(m)
}

// loop runs body while cond (evaluated at the top) holds, then cont.
func (b *builder) loop(cond func() id, body, cont func()) {
	h, bd, c, m := b.id(), b.id(), b.id(), b.id()
	inst(&b.body, opBranch, h)
	b.label(h)
	cv := cond()
	inst(&b.body, opLoopMerge, m, c, 0)
	inst(&b.body, opBranchConditional, cv, bd, m)
	b.label(bd)
	body()
	inst(&b.body, opBranch, c)
	b.label(c)
	cont()
	inst(&b.body, opBranch, h)
	b.label(m)
}

func (b *builder) barrier() {
	inst(&b.body, opControlBarrier, b.u32(scopeWorkgroup), b.u32(scopeWorkgroup), b.u32(semAcqRelWorkgroupMs))
}

// finish assembles the module around the body written so far.
func (b *builder) finish() []uint32 {
	var m []uint32
	inst(&b.caps, opCapability, capShader)
	if b.modes != (floatModes{}) {
		inst(&b.exts, opExtension, strWords("SPV_KHR_float_controls")...)
	}
	modeList := []struct {
		on        bool
		cap, mode uint32
	}{
		{b.modes.denormPreserve, capDenormPreserve, emDenormPreserve},
		{b.modes.signedZeroInfNan, capSignedZeroInfNan, emSignedZeroInfNan},
		{b.modes.roundRTE, capRoundingModeRTE, emRoundingModeRTE},
	}
	var modes []uint32
	for _, md := range modeList {
		if md.on {
			inst(&b.caps, opCapability, md.cap)
			inst(&modes, opExecutionMode, b.main, md.mode, 32)
		}
	}
	inst(&b.header, opMemoryModel, 0, 1)
	inst(&b.header, opEntryPoint, append(append([]uint32{5, b.main}, strWords("main")...), b.entryIfaces...)...)
	inst(&b.header, opExecutionMode, b.main, emLocalSize, b.localX, b.localY, 1)
	b.header = append(b.header, modes...)

	var fn []uint32
	entry := b.id()
	inst(&fn, opFunction, b.tVoid, b.main, 0, b.tFn)
	inst(&fn, opLabel, entry)
	fn = append(fn, b.vars...)
	fn = append(fn, b.body...)
	inst(&fn, opReturn)
	inst(&fn, opFunctionEnd)

	// SPIR-V 1.3: storage buffers without extensions, as Vulkan 1.1 has.
	m = append(m, 0x07230203, 0x00010300, 0, b.next, 0)
	for _, part := range [][]uint32{b.caps, b.exts, b.header, b.annots, b.types, fn} {
		m = append(m, part...)
	}
	return m
}
