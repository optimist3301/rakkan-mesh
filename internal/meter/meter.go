// Package meter instruments WebAssembly modules with deterministic gas
// accounting.
//
// Instrumentation adds a mutable i64 global, exported as GasExport, holding the
// remaining gas, and a mutable i32 global, exported as InterruptExport, that
// the host sets to stop the guest. The host sets it before running guest code and reads it
// afterwards. Every function body is split into straight-line segments, each
// ending at a control instruction; on entry to a segment its exact static cost
// is subtracted. Because no instruction inside a segment transfers control
// (calls return to it), the charge equals the cost of what executes.
//
// The remaining gas and the interrupt flag are checked, trapping with
// unreachable when the gas is negative or the flag set, only
// where unbounded execution could otherwise begin: function entry (recursion),
// loop headers, and instructions whose cost depends on an operand (bulk memory
// and table operations, memory.grow), which are charged dynamically. Between
// checks only bounded straight-line code runs, so the host must also treat a
// negative balance after the guest returns as exhaustion. The outcome is a
// pure function of the module and input, so replicas agree on it.
//
// Floating-point results are NaN-canonicalized. WebAssembly leaves the sign
// and payload of a NaN produced by arithmetic nondeterministic, and hardware
// differs (x86 produces 0xffc00000 for 0/0, ARM 0x7fc00000), so the same
// task could produce different bytes on different nodes. After every
// operator that can produce such a NaN, the result x is replaced by
// isnan(x) ? canonical : x, lane-wise for SIMD, with the canonical quiet
// NaNs 0x7fc00000 and 0x7ff8000000000000. Bitwise operators (abs, neg,
// copysign, loads, stores, reinterpretations) are already deterministic.
//
// Custom sections (names, debug info) carry no semantics and are dropped.
//
// The interrupt flag exists because the gas counter cannot carry the signal:
// the guest updates it with a non-atomic read-modify-write in every segment,
// and on real hardware its loads are served from its own store buffer, so a
// store from another thread is overwritten almost every time. The guest
// never writes the flag, so the host's store is seen at the next check.
//
// A start function would run during instantiation, before the host can set
// the gas balance, so the start section is removed and the function exported
// as StartExport for the host to call explicitly.
package meter

import (
	"bytes"
	"errors"
	"fmt"
	"slices"
)

// ScheduleVersion identifies the cost schedule below. Nodes comparing gas
// usage must run the same version.
const ScheduleVersion = 1

// Cost schedule, version 1. Instructions not listed cost 1.
const (
	CallCost         = 4
	MemoryPageCost   = 4096 // per 64 KiB page grown
	BytesPerGasShift = 4    // bulk memory: 1 gas per 16 bytes
	TableElemCost    = 1    // bulk table: per element
	HostCallCost     = 10   // plus bulk memory rate on bytes transferred
)

// Exported names added by instrumentation.
const (
	GasExport       = "__rakkan_gas"
	InterruptExport = "__rakkan_interrupt"
	StartExport     = "__rakkan_start"
)

// Structural limits, protecting the instrumenter, the compiler and the
// runtime from modules crafted to be expensive to process. They are part of
// the module acceptance rules, so nodes agree on which modules are valid.
//
// wazero's compiler is superlinear in nesting depth, br_table size and in
// locals live across many branches, so a small module can take minutes to
// compile. These limits cut the worst ratios without rejecting code real
// toolchains emit; compilation also counts against the task timeout, past
// which the node kills the worker.
const (
	MaxTypes          = 100_000
	MaxParams         = 128 // per function type, also results
	MaxImports        = 1_000
	MaxFunctions      = 100_000
	MaxTables         = 4
	MaxTableElements  = 1 << 18 // declared maxima above this are lowered to it
	MaxGlobals        = 100_000
	MaxExports        = 100_000
	MaxSegments       = 100_000 // data and element segments, each
	MaxLocals         = 50_000  // per function, excluding parameters
	MaxBodyBytes      = 1 << 20
	MaxNesting        = 1_024 // block, loop and if depth within a function
	MaxBrTableTargets = 16_384
)

var ErrUnsupported = errors.New("unsupported module")

const (
	secCustom    = 0
	secType      = 1
	secImport    = 2
	secFunc      = 3
	secTable     = 4
	secMemory    = 5
	secGlobal    = 6
	secExport    = 7
	secStart     = 8
	secElem      = 9
	secCode      = 10
	secData      = 11
	secDataCount = 12
)

// sectionOrder ranks known sections in the order the binary format requires
// (data count, id 12, comes before code).
var sectionOrder = map[byte]int{1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 12: 10, 10: 11, 11: 12}

// Every section is parsed completely, reading each element of every vector,
// before the module reaches wazero. Its decoder preallocates slices from
// declared vector counts, so a few bytes claiming billions of table entries
// would otherwise exhaust host memory. Complete parsing also guarantees the
// rewriter never drops trailing bytes or lets appended data complete a
// truncated entry.

type section struct {
	id   byte
	data []byte
}

// module holds what the rewriter needs to know about the original module.
type module struct {
	sections           []section
	types              []int    // parameter count per type
	funcs              []uint32 // type index per defined function
	numImportedFuncs   int
	numImportedGlobals int
	numGlobals         int // imported + defined
	exportNames        map[string]bool
	importModules      []string // distinct import namespaces
	start              int64    // -1 when absent
}

// ImportModules returns the distinct namespaces a module imports from,
// after validating its structure as Instrument would. Schedulers use it to
// match a task to nodes offering the host APIs it needs, without compiling
// untrusted code.
func ImportModules(wasm []byte) ([]string, error) {
	m, err := parse(wasm)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrUnsupported, err)
	}
	return m.importModules, nil
}

// Instrument returns a metered copy of a WebAssembly binary.
func Instrument(wasm []byte) ([]byte, error) {
	m, err := parse(wasm)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrUnsupported, err)
	}
	out, err := m.rewrite()
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrUnsupported, err)
	}
	return out, nil
}

var header = []byte{0x00, 'a', 's', 'm', 0x01, 0x00, 0x00, 0x00}

func parse(wasm []byte) (*module, error) {
	if !bytes.HasPrefix(wasm, header) {
		return nil, errors.New("not a WebAssembly 1.0 binary")
	}
	m := &module{exportNames: map[string]bool{}, start: -1}
	r := &reader{buf: wasm, pos: len(header)}
	last := 0
	for !r.done() {
		id, err := r.byte()
		if err != nil {
			return nil, err
		}
		size, err := r.u32()
		if err != nil {
			return nil, err
		}
		data, err := r.bytes(uint64(size))
		if err != nil {
			return nil, err
		}
		if id == secCustom {
			if _, err := (&reader{buf: data}).name(); err != nil {
				return nil, fmt.Errorf("custom section name: %w", err)
			}
			continue
		}
		ord, ok := sectionOrder[id]
		if !ok {
			return nil, fmt.Errorf("unknown section %d", id)
		}
		if ord <= last {
			return nil, fmt.Errorf("section %d out of order", id)
		}
		last = ord
		m.sections = append(m.sections, section{id, data})

		sr := &reader{buf: data}
		switch id {
		case secType:
			err = m.parseTypes(sr)
		case secImport:
			err = m.parseImports(sr)
		case secFunc:
			err = m.parseFuncs(sr)
		case secTable:
			var clamped []byte
			if clamped, err = parseTables(sr); err == nil {
				m.sections[len(m.sections)-1].data = clamped
			}
		case secMemory:
			err = eachEntry(sr, 1, "memories", func() error { return skipLimits(sr) })
		case secGlobal:
			err = m.parseGlobals(sr)
		case secExport:
			err = m.parseExports(sr)
		case secStart:
			var idx uint32
			idx, err = sr.u32()
			m.start = int64(idx)
		case secElem:
			err = eachEntry(sr, MaxSegments, "element segments", func() error { return skipElemSegment(sr) })
		case secData:
			err = eachEntry(sr, MaxSegments, "data segments", func() error { return skipDataSegment(sr) })
		case secDataCount:
			_, err = sr.u32()
		case secCode:
			sr.pos = len(data) // checked by rewriteCode
		}
		// The rewriter re-encodes or appends to these sections, so trailing
		// bytes must not be silently dropped or completed by appended data.
		if err == nil && !sr.done() {
			err = errors.New("trailing bytes")
		}
		if err != nil {
			return nil, fmt.Errorf("section %d: %w", id, err)
		}
	}
	if m.numGlobals > MaxGlobals {
		return nil, fmt.Errorf("%d globals, limit %d", m.numGlobals, MaxGlobals)
	}
	for _, name := range []string{GasExport, InterruptExport, StartExport} {
		if m.exportNames[name] {
			return nil, fmt.Errorf("export name %q is reserved", name)
		}
	}
	return m, nil
}

func (m *module) parseTypes(r *reader) error {
	return eachEntry(r, MaxTypes, "types", func() error {
		if b, err := r.byte(); err != nil || b != 0x60 {
			return errors.New("malformed function type")
		}
		start := r.pos
		if err := valTypes(r, MaxParams); err != nil {
			return err
		}
		params, _ := (&reader{buf: r.buf, pos: start}).u32()
		if err := valTypes(r, MaxParams); err != nil {
			return err
		}
		m.types = append(m.types, int(params))
		return nil
	})
}

// parseTables checks table declarations and re-encodes them with every
// maximum at most MaxTableElements, so table.grow fails deterministically
// instead of exhausting memory.
func parseTables(r *reader) ([]byte, error) {
	n, err := vecLen(r, MaxTables, "tables")
	if err != nil {
		return nil, err
	}
	out := appendUleb(nil, uint64(n))
	for range n {
		rt, err := r.byte()
		if err != nil {
			return nil, err
		}
		if rt != 0x70 && rt != 0x6f {
			return nil, fmt.Errorf("unsupported table element type 0x%02x", rt)
		}
		flag, err := r.byte()
		if err != nil {
			return nil, err
		}
		if flag > 1 {
			return nil, fmt.Errorf("unsupported table limits flags %d", flag)
		}
		minSize, err := r.u32()
		if err != nil {
			return nil, err
		}
		maxSize := uint32(MaxTableElements)
		if flag == 1 {
			declared, err := r.u32()
			if err != nil {
				return nil, err
			}
			maxSize = min(declared, maxSize)
		}
		if minSize > MaxTableElements {
			return nil, fmt.Errorf("table of %d elements, limit %d", minSize, MaxTableElements)
		}
		out = append(out, rt, 0x01)
		out = appendUleb(out, uint64(minSize))
		out = appendUleb(out, uint64(maxSize))
	}
	return out, nil
}

func (m *module) parseImports(r *reader) error {
	n, err := vecLen(r, MaxImports, "imports")
	if err != nil {
		return err
	}
	for range n {
		mod, err := r.name()
		if err != nil {
			return err
		}
		if !slices.Contains(m.importModules, mod) {
			m.importModules = append(m.importModules, mod)
		}
		if _, err := r.name(); err != nil {
			return err
		}
		kind, err := r.byte()
		if err != nil {
			return err
		}
		switch kind {
		case 0x00: // func
			_, err = r.u32()
			m.numImportedFuncs++
		case 0x01: // table: reftype + limits
			if err = valType(r); err == nil {
				err = skipLimits(r)
			}
		case 0x02: // memory
			err = skipLimits(r)
		case 0x03: // global: valtype + mut
			if err = valType(r); err == nil {
				err = r.skip(1)
			}
			m.numImportedGlobals++
			m.numGlobals++
		default:
			err = fmt.Errorf("unknown import kind %d", kind)
		}
		if err != nil {
			return err
		}
	}
	return nil
}

func skipLimits(r *reader) error {
	flag, err := r.byte()
	if err != nil {
		return err
	}
	// Shared (threads) and 64-bit (memory64) limits are unsupported.
	if flag > 1 {
		return fmt.Errorf("unsupported limits flags %d", flag)
	}
	if _, err := r.u32(); err != nil {
		return err
	}
	if flag&0x01 != 0 {
		_, err = r.u32()
	}
	return err
}

// vecLen reads a vector length, enforcing a structural limit.
func vecLen(r *reader, limit int, what string) (uint32, error) {
	n, err := r.u32()
	if err != nil {
		return 0, err
	}
	if uint64(n) > uint64(limit) {
		return 0, fmt.Errorf("%d %s, limit %d", n, what, limit)
	}
	return n, nil
}

// eachEntry reads a vector, calling entry once per declared element.
func eachEntry(r *reader, limit int, what string, entry func() error) error {
	n, err := vecLen(r, limit, what)
	if err != nil {
		return err
	}
	for range n {
		if err := entry(); err != nil {
			return err
		}
	}
	return nil
}

func skipElemSegment(r *reader) error {
	flags, err := r.u32()
	if err != nil {
		return err
	}
	if flags > 7 {
		return fmt.Errorf("unknown element segment flags %d", flags)
	}
	active, explicitTable, exprs := flags&1 == 0, flags&2 != 0, flags&4 != 0
	if active && explicitTable {
		if _, err := r.u32(); err != nil {
			return err
		}
	}
	if active {
		if err := skipConstExpr(r); err != nil {
			return err
		}
	}
	// Element kind (0x00) or reference type, omitted for flags 0 and 4.
	if flags != 0 && flags != 4 {
		if err := r.skip(1); err != nil {
			return err
		}
	}
	if exprs {
		return eachEntry(r, MaxTableElements, "segment elements", func() error { return skipConstExpr(r) })
	}
	return eachEntry(r, MaxTableElements, "segment elements", func() error { _, err := r.u32(); return err })
}

func skipDataSegment(r *reader) error {
	flags, err := r.u32()
	if err != nil {
		return err
	}
	switch flags {
	case 0:
		err = skipConstExpr(r)
	case 1:
	case 2:
		if _, err = r.u32(); err == nil {
			err = skipConstExpr(r)
		}
	default:
		err = fmt.Errorf("unknown data segment flags %d", flags)
	}
	if err != nil {
		return err
	}
	n, err := r.u32()
	if err != nil {
		return err
	}
	return r.skip(uint64(n))
}

func (m *module) parseGlobals(r *reader) error {
	n, err := vecLen(r, MaxGlobals, "globals")
	if err != nil {
		return err
	}
	for range n {
		if err := valType(r); err != nil {
			return err
		}
		if err := r.skip(1); err != nil { // mutability
			return err
		}
		if err := skipConstExpr(r); err != nil {
			return err
		}
		m.numGlobals++
	}
	return nil
}

// skipConstExpr skips an initializer expression up to and including its end.
func skipConstExpr(r *reader) error {
	for {
		in, err := decode(r)
		if err != nil {
			return err
		}
		if in.op == opEnd {
			return nil
		}
	}
}

func (m *module) parseFuncs(r *reader) error {
	n, err := vecLen(r, MaxFunctions, "functions")
	if err != nil {
		return err
	}
	for range n {
		idx, err := r.u32()
		if err != nil {
			return err
		}
		if int(idx) >= len(m.types) {
			return fmt.Errorf("function type %d out of range", idx)
		}
		m.funcs = append(m.funcs, idx)
	}
	return nil
}

func (m *module) parseExports(r *reader) error {
	n, err := vecLen(r, MaxExports, "exports")
	if err != nil {
		return err
	}
	for range n {
		name, err := r.name()
		if err != nil {
			return err
		}
		kind, err := r.byte()
		if err != nil {
			return err
		}
		idx, err := r.u32()
		if err != nil {
			return err
		}
		// The global section precedes exports, so numGlobals is final here.
		// An out-of-range export would otherwise alias the injected global.
		if kind == 0x03 && int(idx) >= m.numGlobals {
			return fmt.Errorf("export of global %d out of range", idx)
		}
		m.exportNames[name] = true
	}
	return nil
}

func (m *module) rewrite() ([]byte, error) {
	gas := uint32(m.numGlobals) // index of the injected gas global
	intr := gas + 1             // index of the injected interrupt flag
	secs := slices.Clone(m.sections)

	// Globals: gas (mut i64) and interrupt flag (mut i32), both 0.
	secs, err := upsert(secs, secGlobal, []byte{0x7e, 0x01, opI64Const, 0x00, opEnd})
	if err != nil {
		return nil, err
	}
	if secs, err = upsert(secs, secGlobal, []byte{0x7f, 0x01, opI32Const, 0x00, opEnd}); err != nil {
		return nil, err
	}

	// Export: the gas global, and the start function if there is one.
	exports := [][]byte{
		appendUleb(append(appendName(nil, GasExport), 0x03), uint64(gas)),
		appendUleb(append(appendName(nil, InterruptExport), 0x03), uint64(intr)),
	}
	if m.start >= 0 {
		exports = append(exports, appendUleb(append(appendName(nil, StartExport), 0x00), uint64(m.start)))
	}
	for _, e := range exports {
		if secs, err = upsert(secs, secExport, e); err != nil {
			return nil, err
		}
	}

	// Start: removed, see package doc.
	secs = slices.DeleteFunc(secs, func(s section) bool { return s.id == secStart })

	for i := range secs {
		if secs[i].id == secCode {
			if secs[i].data, err = m.rewriteCode(secs[i].data, gas); err != nil {
				return nil, err
			}
		}
	}

	return encode(secs), nil
}

func encode(secs []section) []byte {
	out := slices.Clone(header)
	for _, s := range secs {
		out = append(out, s.id)
		out = appendUleb(out, uint64(len(s.data)))
		out = append(out, s.data...)
	}
	return out
}

// upsert appends elem to the vector in section id, creating the section at
// its required position when absent.
func upsert(secs []section, id byte, elem []byte) ([]section, error) {
	i := slices.IndexFunc(secs, func(s section) bool { return s.id == id })
	if i < 0 {
		i = slices.IndexFunc(secs, func(s section) bool { return sectionOrder[s.id] > sectionOrder[id] })
		if i < 0 {
			i = len(secs)
		}
		secs = slices.Insert(secs, i, section{id: id, data: []byte{0x00}})
	}
	data, err := appendToVec(secs[i].data, elem)
	if err != nil {
		return nil, err
	}
	secs[i].data = data
	return secs, nil
}

// appendToVec appends one encoded element to a vector, rewriting its count.
func appendToVec(vec, elem []byte) ([]byte, error) {
	r := &reader{buf: vec}
	n, err := r.u32()
	if err != nil {
		return nil, err
	}
	out := appendUleb(nil, uint64(n)+1)
	out = append(out, vec[r.pos:]...)
	return append(out, elem...), nil
}

func (m *module) rewriteCode(data []byte, gas uint32) ([]byte, error) {
	r := &reader{buf: data}
	n, err := r.u32()
	if err != nil {
		return nil, err
	}
	if int(n) != len(m.funcs) {
		return nil, fmt.Errorf("code section has %d bodies, function section %d", n, len(m.funcs))
	}
	out := appendUleb(nil, uint64(n))
	for i := range n {
		size, err := r.u32()
		if err != nil {
			return nil, err
		}
		if size > MaxBodyBytes {
			return nil, fmt.Errorf("function %d body is %d bytes, limit %d", i, size, MaxBodyBytes)
		}
		body, err := r.bytes(uint64(size))
		if err != nil {
			return nil, err
		}
		params := m.types[m.funcs[i]]
		nb, err := m.rewriteBody(body, params, gas)
		if err != nil {
			return nil, fmt.Errorf("function %d: %w", m.numImportedFuncs+int(i), err)
		}
		out = appendUleb(out, uint64(len(nb)))
		out = append(out, nb...)
	}
	if !r.done() {
		return nil, errors.New("code section: trailing bytes")
	}
	return out, nil
}

func (m *module) rewriteBody(body []byte, params int, gas uint32) ([]byte, error) {
	r := &reader{buf: body}
	groups, err := r.u32()
	if err != nil {
		return nil, err
	}
	groupsStart := r.pos
	locals := 0
	for range groups {
		cnt, err := r.u32()
		if err != nil {
			return nil, err
		}
		if err := valType(r); err != nil {
			return nil, err
		}
		if locals += int(cnt); locals > MaxLocals {
			return nil, fmt.Errorf("more than %d locals", MaxLocals)
		}
	}
	localsEnd := r.pos
	totalLocals := int64(params + locals)

	var code []instr
	needScratch := false
	var needCanon [5]bool // by canonKind
	depth := 1            // the function body is a block
	for !r.done() {
		in, err := decode(r)
		if err != nil {
			return nil, err
		}
		switch in.op {
		case opBlock, opLoop, opIf:
			if depth++; depth > MaxNesting {
				return nil, fmt.Errorf("blocks nested deeper than %d", MaxNesting)
			}
		case opEnd:
			depth--
		}
		if in.global >= int64(m.numGlobals) {
			return nil, fmt.Errorf("global %d out of range", in.global)
		}
		if in.local >= totalLocals {
			return nil, fmt.Errorf("local %d out of range", in.local)
		}
		needScratch = needScratch || in.dyn != nil
		needCanon[in.canon] = true
		code = append(code, in)
	}

	// Locals: append scratch locals after the declared ones: i32 for
	// dynamic charges, and one per value type that canonicalization needs.
	var extra []byte // value type per scratch local
	scratch := uint32(totalLocals)
	if needScratch {
		extra = append(extra, 0x7f)
	}
	var canonLocal [5]uint32
	for _, c := range []struct {
		kinds []canonKind
		vt    byte
	}{{[]canonKind{canonF32}, 0x7d}, {[]canonKind{canonF64}, 0x7c}, {[]canonKind{canonF32x4, canonF64x2}, 0x7b}} {
		used := false
		for _, k := range c.kinds {
			used = used || needCanon[k]
		}
		if !used {
			continue
		}
		idx := uint32(totalLocals) + uint32(len(extra))
		for _, k := range c.kinds {
			canonLocal[k] = idx
		}
		extra = append(extra, c.vt)
	}
	var out []byte
	if len(extra) > 0 {
		out = appendUleb(out, uint64(groups)+uint64(len(extra)))
		out = append(out, body[groupsStart:localsEnd]...)
		for _, vt := range extra {
			out = append(out, 0x01, vt)
		}
	} else {
		out = append(out, body[:localsEnd]...)
	}

	// Instructions: charge each segment on entry.
	segStart, check := true, true
	for i, in := range code {
		if segStart {
			cost := int64(0)
			for _, c := range code[i:] {
				cost += c.cost
				if c.split {
					break
				}
			}
			out = emitCharge(out, gas, cost)
			if check {
				out = emitCheck(out, gas)
			}
			segStart = false
		}
		if in.dyn != nil {
			out = emitDynCharge(out, gas, scratch, in.dyn)
		}
		out = append(out, body[in.start:in.end]...)
		if in.canon != canonNone {
			out = emitCanon(out, in.canon, canonLocal[in.canon])
		}
		if in.split {
			segStart, check = true, in.loop
		}
	}
	return out, nil
}

// emitCharge emits: gas -= cost
func emitCharge(out []byte, gas uint32, cost int64) []byte {
	out = appendUleb(append(out, opGlobalGet), uint64(gas))
	out = appendSleb(append(out, opI64Const), cost)
	out = append(out, opI64Sub, opGlobalSet)
	return appendUleb(out, uint64(gas))
}

// emitCheck emits: if (gas < 0) | interrupt { unreachable }
// The interrupt flag is the global after the gas counter.
func emitCheck(out []byte, gas uint32) []byte {
	out = appendUleb(append(out, opGlobalGet), uint64(gas))
	out = append(out, opI64Const, 0x00, opI64LtS)
	out = appendUleb(append(out, opGlobalGet), uint64(gas+1))
	return append(out, opI32Or, opIf, blockEmpty, opUnreachable, opEnd)
}

// Canonical quiet NaNs, little-endian.
var (
	canonNaN32 = []byte{0x00, 0x00, 0xc0, 0x7f}
	canonNaN64 = []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f}
)

// emitCanon replaces the float result on top of the stack with the
// canonical NaN if it is a NaN, using local t as scratch.
func emitCanon(out []byte, kind canonKind, t uint32) []byte {
	set := appendUleb([]byte{opLocalSet}, uint64(t))
	get := appendUleb([]byte{opLocalGet}, uint64(t))
	switch kind {
	case canonF32, canonF64:
		// select(canonical, x, x != x)
		konst, ne, nan := byte(0x43), byte(0x5c), canonNaN32
		if kind == canonF64 {
			konst, ne, nan = 0x44, 0x62, canonNaN64
		}
		out = append(out, set...)
		out = append(append(out, konst), nan...)
		out = append(out, get...)
		out = append(out, get...)
		out = append(out, get...)
		return append(out, ne, opSelect)
	default:
		// bitselect(x, canonical, x == x): lanes equal to themselves are not NaN.
		lane, eq := canonNaN32, byte(0x41)
		if kind == canonF64x2 {
			lane, eq = canonNaN64, 0x47
		}
		var splat []byte
		for len(splat) < 16 {
			splat = append(splat, lane...)
		}
		out = append(out, set...)
		out = append(out, get...)
		out = append(append(out, prefixSIMD, 0x0c), splat...)
		out = append(out, get...)
		out = append(out, get...)
		out = append(out, prefixSIMD, eq)
		return append(out, prefixSIMD, 0x52)
	}
}

// emitDynCharge charges for the i32 operand on top of the stack, leaving it
// in place: gas -= (n * mul) >> shift, then checks.
func emitDynCharge(out []byte, gas, scratch uint32, d *dynCost) []byte {
	out = appendUleb(append(out, opLocalTee), uint64(scratch))
	out = appendUleb(append(out, opGlobalGet), uint64(gas))
	out = appendUleb(append(out, opLocalGet), uint64(scratch))
	out = append(out, opI64ExtI32U)
	if d.mul != 1 {
		out = appendSleb(append(out, opI64Const), d.mul)
		out = append(out, opI64Mul)
	}
	if d.shift != 0 {
		out = appendSleb(append(out, opI64Const), d.shift)
		out = append(out, opI64ShrU)
	}
	out = append(out, opI64Sub, opGlobalSet)
	out = appendUleb(out, uint64(gas))
	return emitCheck(out, gas)
}
