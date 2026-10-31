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
func Echo() []byte {
	return TaskModule(1, []byte{I32}, Cat(
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
