package sandbox

import (
	"fmt"
	"slices"
	"strings"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
)

// HostModule is the only import namespace a task may link against.
const HostModule = "rakkan"

// DefaultEntry is the export invoked when a task names no entry point.
const DefaultEntry = "run"

type signature struct {
	params, results []api.ValueType
}

var (
	i32 = api.ValueTypeI32

	// hostABI lists every function a task may import. Anything else (WASI,
	// other modules, imported memories or tables) is rejected before
	// instantiation, so the guest has no ambient authority: no clock, no
	// filesystem, no network, no randomness.
	hostABI = map[string]signature{
		// input_len() -> total input size in bytes
		"input_len": {nil, []api.ValueType{i32}},
		// input_read(dst, offset, len) -> bytes copied into guest memory
		"input_read": {[]api.ValueType{i32, i32, i32}, []api.ValueType{i32}},
		// output_write(src, len) appends to the task output
		"output_write": {[]api.ValueType{i32, i32}, nil},
		// log(src, len) appends to the diagnostic log, truncated at the limit
		"log": {[]api.ValueType{i32, i32}, nil},
	}

	// Entry points take no arguments and return an exit code; 0 means success.
	entrySignature = signature{nil, []api.ValueType{i32}}
)

// validate checks a compiled module against the host ABI and entry contract.
func validate(cm wazero.CompiledModule, entry string) error {
	if strings.HasPrefix(entry, "__rakkan_") {
		return fmt.Errorf("%w: %q is reserved", ErrMissingEntry, entry)
	}
	for _, fn := range cm.ImportedFunctions() {
		mod, name, _ := fn.Import()
		want, ok := hostABI[name]
		if mod != HostModule || !ok {
			return fmt.Errorf("%w: %s.%s", ErrForbiddenImport, mod, name)
		}
		if !sameSignature(fn, want) {
			return fmt.Errorf("%w: %s.%s has wrong signature", ErrForbiddenImport, mod, name)
		}
	}
	if mems := cm.ImportedMemories(); len(mems) > 0 {
		mod, name, _ := mems[0].Import()
		return fmt.Errorf("%w: memory %s.%s", ErrForbiddenImport, mod, name)
	}

	// An initial size above the memory limit is already rejected at compile
	// time; a declared maximum above it is clamped and memory.grow fails.
	if _, ok := cm.ExportedMemories()["memory"]; !ok {
		return fmt.Errorf("%w: module must export its linear memory as \"memory\"", ErrInvalidModule)
	}

	fn, ok := cm.ExportedFunctions()[entry]
	if !ok {
		return fmt.Errorf("%w: no export %q", ErrMissingEntry, entry)
	}
	if !sameSignature(fn, entrySignature) {
		return fmt.Errorf("%w: %q must have signature () -> i32", ErrMissingEntry, entry)
	}
	return nil
}

func sameSignature(fn api.FunctionDefinition, s signature) bool {
	return slices.Equal(fn.ParamTypes(), s.params) && slices.Equal(fn.ResultTypes(), s.results)
}
