package sandbox

import (
	"fmt"
	"slices"
	"strings"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
)

// HostModule is the base import namespace, available to every task.
const HostModule = "rakkan"

// DefaultEntry is the export invoked when a task names no entry point.
const DefaultEntry = "run"

type signature struct {
	params, results []api.ValueType
}

func sig(params []api.ValueType, results ...api.ValueType) signature {
	return signature{params, results}
}

var (
	i32 = api.ValueTypeI32
	i64 = api.ValueTypeI64

	// hostABI lists every function a task may import, by namespace.
	// Anything else (WASI, other modules, imported memories or tables) is
	// rejected before instantiation, so the guest has no ambient authority:
	// no clock, no filesystem, no network, no randomness.
	hostABI = map[string]map[string]signature{
		HostModule: {
			// input_len() -> total input size in bytes
			"input_len": sig(nil, i32),
			// input_read(dst, offset, len) -> bytes copied into guest memory
			"input_read": sig([]api.ValueType{i32, i32, i32}, i32),
			// output_write(src, len) appends to the task output
			"output_write": sig([]api.ValueType{i32, i32}),
			// log(src, len) appends to the diagnostic log, truncated at the limit
			"log": sig([]api.ValueType{i32, i32}),
			// blob_count() -> number of blobs attached to the task
			"blob_count": sig(nil, i32),
			// blob_size(index) -> size in bytes, or -1 for no such blob
			"blob_size": sig([]api.ValueType{i32}, i64),
			// blob_read(index, dst, offset, len) -> bytes read into guest
			// memory (0 at the end), or -1 for no such blob
			"blob_read": sig([]api.ValueType{i32, i32, i64, i32}, i32),
		},
		ImageModule:       imageABI,
		TensorModule:      tensorABI,
		RandomXModule:     randomxABI,
		RandomXFastModule: randomxFastABI,
	}

	// capabilityOf names the capability each optional namespace provides.
	// Nodes advertise capabilities; schedulers send tasks only to nodes
	// having every capability a task imports.
	capabilityOf = map[string]string{ImageModule: ImageCapability, TensorModule: TensorCapability, RandomXModule: RandomXCapability, RandomXFastModule: RandomXFastCapability}
	f32          = api.ValueTypeF32

	// Entry points take no arguments and return an exit code; 0 means success.
	entrySignature = signature{nil, []api.ValueType{i32}}
)

// Capabilities lists the optional host APIs this build provides.
func Capabilities() []string {
	var out []string
	rxShared.mu.Lock()
	fast := rxShared.enabled
	rxShared.mu.Unlock()
	for _, c := range capabilityOf {
		if c == RandomXFastCapability && !fast {
			continue // only a node with a shared dataset offers it
		}
		out = append(out, c)
	}
	slices.Sort(out)
	return out
}

// CapabilitiesFor returns the capabilities a module needs, from the import
// namespaces it uses. Unknown namespaces are reported as themselves; no
// node provides them, and the module fails validation wherever it runs.
func CapabilitiesFor(importModules []string) []string {
	var out []string
	for _, m := range importModules {
		if m == HostModule {
			continue
		}
		c, ok := capabilityOf[m]
		if !ok {
			c = "import:" + m
		}
		if !slices.Contains(out, c) {
			out = append(out, c)
		}
	}
	slices.Sort(out)
	return out
}

// validate checks a compiled module against the host ABI and entry contract.
func validate(cm wazero.CompiledModule, entry string) error {
	if strings.HasPrefix(entry, "__rakkan_") {
		return fmt.Errorf("%w: %q is reserved", ErrMissingEntry, entry)
	}
	for _, fn := range cm.ImportedFunctions() {
		mod, name, _ := fn.Import()
		want, ok := hostABI[mod][name]
		if !ok {
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
