package wasmtest

// Function indices of the host imports declared by TaskModule.
const (
	FnInputLen uint32 = iota
	FnInputRead
	FnOutputWrite
	FnLog
)

// TaskModule builds a module importing the full rakkan host ABI with a
// single exported entry point "run" of type () -> i32.
func TaskModule(memPages uint32, locals []byte, body []byte) Module {
	return Module{
		Imports: []Import{
			{Module: "rakkan", Name: "input_len", Results: []byte{I32}},
			{Module: "rakkan", Name: "input_read", Params: []byte{I32, I32, I32}, Results: []byte{I32}},
			{Module: "rakkan", Name: "output_write", Params: []byte{I32, I32}},
			{Module: "rakkan", Name: "log", Params: []byte{I32, I32}},
		},
		Funcs:        []Func{{Results: []byte{I32}, Locals: locals, Body: body, Export: "run"}},
		Memory:       true,
		MemMin:       memPages,
		ExportMemory: true,
	}
}

// Echo copies its input to its output.
func Echo() []byte { return EchoPages(1) }

// EchoPages is Echo with the given initial memory, for larger inputs.
func EchoPages(pages uint32) []byte {
	return TaskModule(pages, []byte{I32}, Cat(
		Call(FnInputLen), []byte{OpLocalSet, 0},
		I32Const(0), I32Const(0), []byte{OpLocalGet, 0}, Call(FnInputRead), []byte{OpDrop},
		I32Const(0), []byte{OpLocalGet, 0}, Call(FnOutputWrite),
		I32Const(0),
	)).Encode()
}

// Spin loops forever.
func Spin() []byte {
	return TaskModule(1, nil, Cat([]byte{OpLoop, BlockEmpty, OpBr, 0, OpEnd}, I32Const(0))).Encode()
}

// ULEB encodes an unsigned LEB128 value.
func ULEB(v uint32) []byte { return uleb(uint64(v)) }

// Repeat concatenates n copies of b.
func Repeat(b []byte, n int) []byte {
	out := make([]byte, 0, len(b)*n)
	for range n {
		out = append(out, b...)
	}
	return out
}

// CompileBomb is small and within structural limits, but slow to compile
// for wazero: many locals live across many branch merges. With 16k locals
// it takes over ten seconds.
func CompileBomb(numLocals int) []byte {
	locals := Repeat([]byte{I32}, numLocals)
	var body []byte
	for k := range 2000 {
		body = append(body, Cat([]byte{0x02, BlockEmpty, OpLocalGet}, ULEB(uint32(k%numLocals)), []byte{OpBrIf, 0},
			I32Const(1), []byte{OpLocalSet}, ULEB(uint32((k*31)%numLocals)), []byte{OpEnd})...)
	}
	for i := 0; i < numLocals; i += max(numLocals/2000, 1) {
		body = append(body, Cat([]byte{OpLocalGet}, ULEB(uint32(i)), []byte{OpDrop})...)
	}
	return TaskModule(1, locals, Cat(body, I32Const(0))).Encode()
}

// Countdown loops n times and exits 0: work of predictable duration.
func Countdown(n int32) []byte {
	return TaskModule(1, []byte{I32}, Cat(
		I32Const(n), []byte{OpLocalSet, 0},
		[]byte{OpLoop, BlockEmpty},
		[]byte{OpLocalGet, 0}, I32Const(1), []byte{OpI32Sub, OpLocalTee, 0, OpBrIf, 0},
		[]byte{OpEnd},
		I32Const(0),
	)).Encode()
}

// Match reads a little-endian u32 from its input and exits 0 if it equals
// target, 1 otherwise.
func Match(target int32) []byte {
	return TaskModule(1, nil, Cat(
		I32Const(0), I32Const(0), I32Const(4), Call(FnInputRead), []byte{OpDrop},
		I32Const(0), []byte{0x28, 0x02, 0x00}, // i32.load align=4 offset=0
		I32Const(target), []byte{0x47}, // i32.ne
	)).Encode()
}

// Search finds a preimage of a 32-bit hash in a range. Its input is three
// little-endian u32s: start, count, target. It exits 0 and outputs the
// preimage (u32) if one of start..start+count-1 hashes to target, else
// exits 1. The hash is Mix32.
func Search() []byte {
	const (
		i, end, target, h = 0, 1, 2, 3
		load              = 0x28
		store             = 0x36
		geU, eq           = 0x4f, 0x46
		add, mul, xor     = 0x6a, 0x6c, 0x73
		shrU, ret         = 0x76, 0x0f
	)
	get := func(l byte) []byte { return []byte{OpLocalGet, l} }
	set := func(l byte) []byte { return []byte{OpLocalSet, l} }
	tee := func(l byte) []byte { return []byte{OpLocalTee, l} }
	loadAt := func(off byte) []byte { return Cat(I32Const(0), []byte{load, 0x02, off}) }
	return TaskModule(1, []byte{I32, I32, I32, I32}, Cat(
		I32Const(0), I32Const(0), I32Const(12), Call(FnInputRead), []byte{OpDrop},
		loadAt(0), set(i),
		get(i), loadAt(4), []byte{add}, set(end),
		loadAt(8), set(target),
		[]byte{0x02, BlockEmpty, OpLoop, BlockEmpty},
		get(i), get(end), []byte{geU, OpBrIf, 1},
		// h = Mix32(i)
		get(i), I32Const(-1640531535), []byte{mul}, tee(h),
		get(h), I32Const(16), []byte{shrU, xor}, tee(h),
		I32Const(-2048144789), []byte{mul}, tee(h),
		get(h), I32Const(13), []byte{shrU, xor},
		get(target), []byte{eq, OpIf, BlockEmpty},
		I32Const(16), get(i), []byte{store, 0x02, 0x00},
		I32Const(16), I32Const(4), Call(FnOutputWrite),
		I32Const(0), []byte{ret},
		[]byte{OpEnd},
		get(i), I32Const(1), []byte{add}, set(i),
		[]byte{OpBr, 0, OpEnd, OpEnd},
		I32Const(1),
	)).Encode()
}

// Mix32 is the hash Search inverts: MurmurHash3's finalizer, abridged.
func Mix32(x uint32) uint32 {
	x *= 0x9E3779B1
	x ^= x >> 16
	x *= 0x85EBCA6B
	x ^= x >> 13
	return x
}

// Thumbnail decodes its input image, fits it in maxW x maxH and outputs
// it encoded (format 0 PNG, 1 JPEG). Exit codes: 0 ok, 1 decode failed,
// 2 resize failed, 3 encode failed.
func Thumbnail(maxW, maxH, format, quality int32) []byte {
	const img = "rakkan_image_v1"
	const ltS, ret = 0x48, 0x0f // i32.lt_s, return
	fail := func(code int32) []byte {
		return Cat(I32Const(0), []byte{ltS, OpIf, BlockEmpty}, I32Const(code), []byte{ret, OpEnd})
	}
	return Module{
		Imports: []Import{
			{Module: img, Name: "decode_input", Results: []byte{I32}},
			{Module: img, Name: "fit", Params: []byte{I32, I32, I32}, Results: []byte{I32}},
			{Module: img, Name: "encode_output", Params: []byte{I32, I32, I32}, Results: []byte{I32}},
		},
		Funcs: []Func{{Results: []byte{I32}, Locals: []byte{I32}, Export: "run", Body: Cat(
			Call(0), []byte{OpLocalTee, 0}, fail(1),
			[]byte{OpLocalGet, 0}, I32Const(maxW), I32Const(maxH), Call(1), []byte{OpLocalTee, 0}, fail(2),
			[]byte{OpLocalGet, 0}, I32Const(format), I32Const(quality), Call(2), fail(3),
			I32Const(0),
		)}},
		Memory: true, MemMin: 1, ExportMemory: true,
	}.Encode()
}

// BlobEcho copies attached blob 0 to its output, 64 KiB at a time.
func BlobEcho() []byte {
	const (
		geS, add64, ext, eqz = 0x59, 0x7c, 0xad, 0x45
		size, off, n         = 0, 1, 2
	)
	return Module{
		Imports: []Import{
			{Module: "rakkan", Name: "blob_size", Params: []byte{I32}, Results: []byte{I64}},
			{Module: "rakkan", Name: "blob_read", Params: []byte{I32, I32, I64, I32}, Results: []byte{I32}},
			{Module: "rakkan", Name: "output_write", Params: []byte{I32, I32}},
		},
		Funcs: []Func{{Results: []byte{I32}, Locals: []byte{I64, I64, I32}, Export: "run", Body: Cat(
			I32Const(0), Call(0), []byte{OpLocalSet, size},
			I64Const(0), []byte{OpLocalSet, off},
			[]byte{0x02, BlockEmpty, OpLoop, BlockEmpty},
			[]byte{OpLocalGet, off, OpLocalGet, size, geS, OpBrIf, 1},
			I32Const(0), I32Const(0), []byte{OpLocalGet, off}, I32Const(65536), Call(1), []byte{OpLocalTee, n},
			[]byte{eqz, OpBrIf, 1},
			I32Const(0), []byte{OpLocalGet, n}, Call(2),
			[]byte{OpLocalGet, off, OpLocalGet, n, ext, add64, OpLocalSet, off},
			[]byte{OpBr, 0, OpEnd, OpEnd},
			I32Const(0),
		)}},
		Memory: true, MemMin: 2, ExportMemory: true,
	}.Encode()
}
