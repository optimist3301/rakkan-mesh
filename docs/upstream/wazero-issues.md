# Issues for tetratelabs/wazero

Drafts of three reports for https://github.com/tetratelabs/wazero, found while
building the Rakkan sandbox. Each was reproduced against plain wazero v1.12.0,
without any Rakkan code, using the program at the end of this file.

Environment for all three: wazero v1.12.0, Go 1.26.10, linux/amd64,
AMD Ryzen 7 5800U, default (compiler) runtime unless stated.

> **Issue 1 is a denial of service** reachable by anyone who can submit a
> module for compilation. Consider reporting it privately through GitHub's
> "Report a vulnerability" (security advisory) flow instead of a public issue,
> and publishing the issue text after a fix.

---

## Issue 1: Decoder preallocates from declared vector counts; a 14-byte module crashes the host process

### Summary

`CompileModule` allocates slices sized by vector counts declared in the binary
before checking that the section contains that many entries. A module of a
few bytes declaring a huge count makes the Go runtime abort with
`fatal error: out of memory`. That is not a panic and cannot be recovered, so
it terminates the whole host process, not just the compilation.

### Reproduction

Table section declaring 2³²−1 tables (14 bytes in total):

```
00 61 73 6d 01 00 00 00   ; header
04 05 ff ff ff ff 0f      ; table section, count = 4294967295, no entries
```

```go
rt := wazero.NewRuntime(ctx)
_, err := rt.CompileModule(ctx, []byte("\x00asm\x01\x00\x00\x00\x04\x05\xff\xff\xff\xff\x0f"))
```

```
$ (ulimit -v 4000000; ./repro table)
runtime: out of memory: cannot allocate 137438953472-byte block (3833856 in use)
fatal error: out of memory
```

Without the `ulimit`, the process tries to commit 128 GiB and is killed by the
kernel's OOM killer, or the machine swaps heavily.

A data segment declaring a 4 GiB payload (13 bytes) does the same through
`make([]byte, vs)` in `internal/wasm/binary/data.go`:

```
00 61 73 6d 01 00 00 00   ; header
0b 07 01 01 ff ff ff ff 0f ; data section, 1 passive segment of 4294967295 bytes
```

```
$ (ulimit -v 4000000; ./repro data)
runtime: out of memory: cannot allocate 4294967296-byte block (3833856 in use)
fatal error: out of memory
```

### Affected code

All `make` calls sized by a decoded count in `internal/wasm/binary`, for example:

- `section.go`: `decodeImportSection` (`make([]wasm.Import, vs)`), `decodeFunctionSection`,
  `decodeTableSection` (`make([]wasm.Table, vs)`), `decodeGlobalSection`, `decodeExportSection`,
  `decodeElementSection`, `decodeCodeSection`, `decodeDataSection`, `decodeTagSection`
- `element.go`: `make([]wasm.ConstantExpression, vs)` (two places)
- `data.go`: `ret.Init = make([]byte, vs)`
- `value.go`: `make([]wasm.ValueType, 0, num)`
- `code.go`: `make([]wasm.ValueType, 0, sum)` for locals

### Suggested fix

Bound every count by the bytes left in the section before allocating: each
entry takes at least one byte, so `count <= remaining` is always a valid
check (or, more tightly, `count * minEntrySize <= remaining`). Alternatively
grow the slices with `append` while decoding. For `data.go`, compare the
payload length with the remaining bytes before `make`.

---

## Issue 2: Compilation time is superlinear in nesting depth, `br_table` size, and locals × blocks

### Summary

Small modules, tens of kilobytes, take seconds to compile, and the time grows
superlinearly with the module, which makes compile time a cheap DoS vector
for hosts running untrusted modules. Three shapes show it; each module below
is a single function of type `() -> ()`.

### Measurements (`CompileModule`, default compiler)

**Nesting depth**: `n` nested `block`s and `n` `end`s. About quadratic.

| depth | module size | compile time |
|---|---|---|
| 4 000 | 12 KB | 52 ms |
| 8 000 | 24 KB | 192 ms |
| 16 000 | 48 KB | 718 ms |
| 32 000 | 96 KB | 2.90 s |

**`br_table` size**: one `br_table` with `n` targets, all to the same label.
About ×3.6 per doubling.

| targets | module size | compile time |
|---|---|---|
| 25 000 | 25 KB | 162 ms |
| 50 000 | 50 KB | 454 ms |
| 100 000 | 100 KB | 1.62 s |
| 200 000 | 200 KB | 5.91 s |

**Locals × blocks**: `n` i32 locals and 2 000 sequential blocks of the form
`block; local.get a; br_if 0; i32.const 1; local.set b; end`. Each block touches
two locals, yet time grows linearly with the total number of locals, which
suggests phis or per-block state for every local at every merge point.

| locals | module size | compile time |
|---|---|---|
| 250 | 24 KB | 242 ms |
| 500 | 25 KB | 438 ms |
| 1 000 | 26 KB | 1.07 s |
| 2 000 | 26 KB | 2.21 s |

With 16 000 locals and the same 2 000 blocks (64 KB), compilation takes over
13 seconds and allocates more than 300 MB.

### Expected

Compile time roughly linear in module size, as in V8 and Wasmtime/Cranelift.

### Suggestions

- Locals × blocks: pruned or lazy SSA construction, creating block parameters
  only for variables actually defined and used across the merge.
- `br_table`: deduplicate identical targets, or avoid per-target work that
  scales with the number of targets times something else (control stack
  depth, live values).
- Nesting: avoid per-block work proportional to the control stack depth.

Until then, documenting the complexity would help embedders set limits. We
currently cap nesting at 1 024 and `br_table` at 16 384 targets.

---

## Issue 3: `WithCloseOnContextDone(true)` slows tight loops about 30×

### Summary

Enabling `WithCloseOnContextDone` makes a tight loop about 30 times slower,
even when the context never ends.

### Reproduction

A function counting a local down from 50 000 000:

```wat
(func (export "run") (local i32)
  (local.set 0 (i32.const 50000000))
  (loop
    (br_if 0 (local.tee 0 (i32.sub (local.get 0) (i32.const 1))))))
```

```
WithCloseOnContextDone(false): 50M iterations in 34ms
WithCloseOnContextDone(true):  50M iterations in 1.035s
```

### Expected

A small constant overhead per loop iteration, such as one load and a
predictable branch on a flag in the module context.

### Context

We need interruption for untrusted code, so we use a workaround: our module
rewriter inserts a check of a host-set i32 global at loop headers and
function entries. That costs about 1.7× on the same loop, including our gas
metering. A cheaper built-in mechanism would let embedders drop the rewriting.

---

## Reproducer

`go.mod`:

```
module repro

go 1.24

require github.com/tetratelabs/wazero v1.12.0
```

`main.go` (run as `./repro table|data|nesting|brtable|locals|close`):

```go
package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/tetratelabs/wazero"
)

func compile(name string, wasm []byte) {
	ctx := context.Background()
	rt := wazero.NewRuntime(ctx)
	defer rt.Close(ctx)
	start := time.Now()
	_, err := rt.CompileModule(ctx, wasm)
	fmt.Printf("%s (%d bytes): %v, err=%v\n", name, len(wasm), time.Since(start).Round(time.Millisecond), err)
}

func uleb(v uint32) []byte {
	var out []byte
	for {
		b := byte(v & 0x7f)
		v >>= 7
		if v != 0 {
			b |= 0x80
		}
		out = append(out, b)
		if v == 0 {
			return out
		}
	}
}

func cat(parts ...[]byte) []byte {
	var out []byte
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}

func section(id byte, content []byte) []byte { return cat([]byte{id}, uleb(uint32(len(content))), content) }

var header = []byte("\x00asm\x01\x00\x00\x00")

// module returns one function of type () -> () with the given locals and body.
func module(numLocals int, body []byte) []byte {
	locals := []byte{0}
	if numLocals > 0 {
		locals = cat([]byte{1}, uleb(uint32(numLocals)), []byte{0x7f})
	}
	code := cat(locals, body, []byte{0x0b})
	return cat(header,
		section(1, []byte{1, 0x60, 0, 0}),
		section(3, []byte{1, 0}),
		section(10, cat([]byte{1}, uleb(uint32(len(code))), code)))
}

func repeat(b []byte, n int) []byte {
	var out []byte
	for range n {
		out = append(out, b...)
	}
	return out
}

func main() {
	switch os.Args[1] {
	case "table":
		compile("table section claiming 2^32-1 tables", cat(header, section(4, []byte{0xff, 0xff, 0xff, 0xff, 0x0f})))
	case "data":
		compile("data segment claiming 4 GiB", cat(header, section(11, []byte{1, 1, 0xff, 0xff, 0xff, 0xff, 0x0f})))
	case "nesting":
		for _, n := range []int{4000, 8000, 16000, 32000} {
			compile(fmt.Sprintf("nesting depth %d", n), module(0, cat(repeat([]byte{0x02, 0x40}, n), repeat([]byte{0x0b}, n))))
		}
	case "brtable":
		for _, n := range []int{25000, 50000, 100000, 200000} {
			compile(fmt.Sprintf("br_table with %d targets", n), module(0, cat([]byte{0x02, 0x40, 0x41, 0x00, 0x0e}, uleb(uint32(n)), make([]byte, n+1), []byte{0x0b})))
		}
	case "locals":
		for _, n := range []int{250, 500, 1000, 2000} {
			var body []byte
			for k := range 2000 {
				body = append(body, cat([]byte{0x02, 0x40, 0x20}, uleb(uint32(k%n)), []byte{0x0d, 0x00, 0x41, 0x01, 0x21}, uleb(uint32((k*31)%n)), []byte{0x0b})...)
			}
			compile(fmt.Sprintf("%d locals, 2000 blocks", n), module(n, body))
		}
	case "close":
		// local0 = 50_000_000; loop { local0 -= 1; br_if local0 }
		body := cat([]byte{0x41, 0x80, 0xe1, 0xeb, 0x17, 0x21, 0x00, 0x03, 0x40, 0x20, 0x00, 0x41, 0x01, 0x6b, 0x22, 0x00, 0x0d, 0x00, 0x0b})
		code := cat([]byte{1, 1, 0x7f}, body, []byte{0x0b})
		wasm := cat(header, section(1, []byte{1, 0x60, 0, 0}), section(3, []byte{1, 0}),
			section(7, []byte{1, 3, 'r', 'u', 'n', 0, 0}), section(10, cat([]byte{1}, uleb(uint32(len(code))), code)))
		for _, closeOnDone := range []bool{false, true} {
			ctx := context.Background()
			rt := wazero.NewRuntimeWithConfig(ctx, wazero.NewRuntimeConfig().WithCloseOnContextDone(closeOnDone))
			mod, err := rt.Instantiate(ctx, wasm)
			if err != nil {
				panic(err)
			}
			start := time.Now()
			if _, err := mod.ExportedFunction("run").Call(ctx); err != nil {
				panic(err)
			}
			fmt.Printf("WithCloseOnContextDone(%v): 50M iterations in %v\n", closeOnDone, time.Since(start).Round(time.Millisecond))
			rt.Close(ctx)
		}
	}
}
```
