package meter

import "fmt"

// Opcodes with special meaning for metering.
const (
	opUnreachable = 0x00
	opBlock       = 0x02
	opLoop        = 0x03
	opIf          = 0x04
	opElse        = 0x05
	opEnd         = 0x0b
	opBr          = 0x0c
	opBrIf        = 0x0d
	opBrTable     = 0x0e
	opReturn      = 0x0f
	opCall        = 0x10
	opCallInd     = 0x11
	opLocalGet    = 0x20
	opLocalSet    = 0x21
	opLocalTee    = 0x22
	opSelect      = 0x1b
	opGlobalGet   = 0x23
	opGlobalSet   = 0x24
	opMemoryGrow  = 0x40
	opI32Const    = 0x41
	opI64Const    = 0x42
	opI64LtS      = 0x53
	opI32Or       = 0x72
	opI64Sub      = 0x7d
	opI64Mul      = 0x7e
	opI64ShrU     = 0x88
	opI64ExtI32U  = 0xad
	prefixFC      = 0xfc
	prefixSIMD    = 0xfd
	blockEmpty    = 0x40
)

// canonKind selects how a result is NaN-canonicalized (see meter.go).
type canonKind uint8

const (
	canonNone canonKind = iota
	canonF32
	canonF64
	canonF32x4
	canonF64x2
)

// scalarCanon lists scalar operators whose NaN results have nondeterministic
// sign and payload: arithmetic, rounding and precision conversions. abs, neg,
// copysign and reinterpretations operate on bits and are deterministic.
var scalarCanon = map[byte]canonKind{
	0x8d: canonF32, 0x8e: canonF32, 0x8f: canonF32, 0x90: canonF32, // ceil floor trunc nearest
	0x91: canonF32, 0x92: canonF32, 0x93: canonF32, 0x94: canonF32, // sqrt add sub mul
	0x95: canonF32, 0x96: canonF32, 0x97: canonF32, // div min max
	0x9b: canonF64, 0x9c: canonF64, 0x9d: canonF64, 0x9e: canonF64,
	0x9f: canonF64, 0xa0: canonF64, 0xa1: canonF64, 0xa2: canonF64,
	0xa3: canonF64, 0xa4: canonF64, 0xa5: canonF64,
	0xb6: canonF32, // f32.demote_f64
	0xbb: canonF64, // f64.promote_f32
}

// simdCanon is the same for SIMD operators (after the 0xfd prefix). pmin
// and pmax return one of their operands unchanged and need nothing.
var simdCanon = map[uint32]canonKind{
	0x67: canonF32x4, 0x68: canonF32x4, 0x69: canonF32x4, 0x6a: canonF32x4, // ceil floor trunc nearest
	0xe3: canonF32x4, 0xe4: canonF32x4, 0xe5: canonF32x4, 0xe6: canonF32x4, // sqrt add sub mul
	0xe7: canonF32x4, 0xe8: canonF32x4, 0xe9: canonF32x4, // div min max
	0x74: canonF64x2, 0x75: canonF64x2, 0x7a: canonF64x2, 0x94: canonF64x2,
	0xef: canonF64x2, 0xf0: canonF64x2, 0xf1: canonF64x2, 0xf2: canonF64x2,
	0xf3: canonF64x2, 0xf4: canonF64x2, 0xf5: canonF64x2,
	0x5e: canonF32x4, // f32x4.demote_f64x2_zero
	0x5f: canonF64x2, // f64x2.promote_low_f32x4
}

// dynCost describes a charge proportional to the i32 operand on top of the
// stack: (n * mul) >> shift.
type dynCost struct {
	mul   int64
	shift int64
}

type instr struct {
	start, end int // byte range within the function body
	op         byte
	cost       int64
	split      bool     // a new metering segment starts after this instruction
	loop       bool     // the segment after this instruction is a loop header
	dyn        *dynCost // charged before the instruction executes
	canon      canonKind

	// Index immediates the rewriter must bounds-check against the original
	// module, because metering adds a global and a local at the next index.
	global, local int64
}

var (
	dynBytes = &dynCost{mul: 1, shift: BytesPerGasShift}
	dynElems = &dynCost{mul: TableElemCost}
	dynPages = &dynCost{mul: MemoryPageCost}
)

// decode reads one instruction. Unsupported proposals (threads, exceptions,
// tail calls, GC) are rejected rather than passed through unmetered.
func decode(r *reader) (instr, error) {
	in := instr{start: r.pos, cost: 1, global: -1, local: -1}
	op, err := r.byte()
	if err != nil {
		return in, err
	}
	in.op = op

	switch {
	case op == opUnreachable || op == opReturn || op == opEnd || op == opElse:
		in.split = true
	case op == 0x01, op == 0x1a, op == 0x1b, op == 0xd1: // nop, drop, select, ref.is_null
	case op == opBlock || op == opLoop || op == opIf:
		err = skipBlockType(r)
		in.split = true
		in.loop = op == opLoop
	case op == opBr || op == opBrIf:
		_, err = r.u32()
		in.split = true
	case op == opBrTable:
		var n uint32
		if n, err = vecLen(r, MaxBrTableTargets, "br_table targets"); err == nil {
			for i := uint64(0); i <= uint64(n) && err == nil; i++ {
				_, err = r.u32()
			}
		}
		in.cost = 2
		in.split = true
	case op == opCall:
		_, err = r.u32()
		in.cost = CallCost
	case op == opCallInd:
		if _, err = r.u32(); err == nil {
			_, err = r.u32()
		}
		in.cost = CallCost + 1
	case op == 0x1c: // select t*
		err = valTypes(r, 1)
	case op >= opLocalGet && op <= opLocalTee:
		var idx uint32
		idx, err = r.u32()
		in.local = int64(idx)
	case op == opGlobalGet || op == opGlobalSet:
		var idx uint32
		idx, err = r.u32()
		in.global = int64(idx)
	case op == 0x25 || op == 0x26: // table.get, table.set
		_, err = r.u32()
	case op >= 0x28 && op <= 0x3e: // loads and stores
		err = skipMemarg(r)
	case op == 0x3f: // memory.size
		_, err = r.u32()
	case op == opMemoryGrow:
		_, err = r.u32()
		in.dyn = dynPages
	case op == 0x41:
		err = r.skipSleb(32)
	case op == opI64Const:
		err = r.skipSleb(64)
	case op == 0x43:
		err = r.skip(4)
	case op == 0x44:
		err = r.skip(8)
	case op >= 0x45 && op <= 0xc4: // numeric, no immediates
		in.canon = scalarCanon[op]
	case op == 0xd0: // ref.null t
		err = valType(r)
	case op == 0xd2: // ref.func
		_, err = r.u32()
	case op == prefixFC:
		err = decodeFC(r, &in)
	case op == prefixSIMD:
		err = decodeSIMD(r, &in)
	default:
		err = fmt.Errorf("unsupported opcode 0x%02x", op)
	}
	in.end = r.pos
	return in, err
}

func decodeFC(r *reader, in *instr) error {
	sub, err := r.u32()
	if err != nil {
		return err
	}
	switch {
	case sub <= 7: // saturating truncation
		return nil
	case sub == 8: // memory.init data mem
		in.dyn = dynBytes
		if _, err := r.u32(); err != nil {
			return err
		}
		_, err = r.u32()
	case sub == 9 || sub == 13 || sub == 16: // data.drop, elem.drop, table.size
		_, err = r.u32()
	case sub == 10: // memory.copy mem mem
		in.dyn = dynBytes
		if _, err := r.u32(); err != nil {
			return err
		}
		_, err = r.u32()
	case sub == 11: // memory.fill mem
		in.dyn = dynBytes
		_, err = r.u32()
	case sub == 12 || sub == 14: // table.init, table.copy
		in.dyn = dynElems
		if _, err := r.u32(); err != nil {
			return err
		}
		_, err = r.u32()
	case sub == 15 || sub == 17: // table.grow, table.fill
		in.dyn = dynElems
		_, err = r.u32()
	default:
		err = fmt.Errorf("unsupported opcode 0xfc %d", sub)
	}
	return err
}

func decodeSIMD(r *reader, in *instr) error {
	sub, err := r.u32()
	if err != nil {
		return err
	}
	in.canon = simdCanon[sub]
	switch {
	case sub <= 11 || sub == 92 || sub == 93: // loads, stores, load_zero
		return skipMemarg(r)
	case sub == 12 || sub == 13: // v128.const, i8x16.shuffle
		return r.skip(16)
	case sub >= 21 && sub <= 34: // extract_lane, replace_lane
		return r.skip(1)
	case sub >= 84 && sub <= 91: // load_lane, store_lane
		if err := skipMemarg(r); err != nil {
			return err
		}
		return r.skip(1)
	case sub <= 255:
		return nil
	}
	return fmt.Errorf("unsupported opcode 0xfd %d", sub)
}

// valType reads a WebAssembly 2.0 value type. Types from later proposals
// (GC, typed function references) are prefixes whose meaning depends on the
// following bytes, which instrumentation may change, so they are rejected.
func valType(r *reader) error {
	b, err := r.byte()
	if err != nil {
		return err
	}
	switch b {
	case 0x7f, 0x7e, 0x7d, 0x7c, 0x7b, 0x70, 0x6f:
		return nil
	}
	return fmt.Errorf("unsupported value type 0x%02x", b)
}

func valTypes(r *reader, limit int) error {
	n, err := vecLen(r, limit, "value types")
	if err != nil {
		return err
	}
	for range n {
		if err := valType(r); err != nil {
			return err
		}
	}
	return nil
}

func skipMemarg(r *reader) error {
	if _, err := r.u32(); err != nil {
		return err
	}
	_, err := r.u32()
	return err
}

func skipBlockType(r *reader) error {
	if r.done() {
		return errEOF
	}
	switch r.buf[r.pos] {
	case blockEmpty, 0x7f, 0x7e, 0x7d, 0x7c, 0x7b, 0x70, 0x6f:
		r.pos++
		return nil
	}
	return r.skipSleb(33) // type index
}
