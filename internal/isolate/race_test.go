//go:build race

package isolate

// ThreadSanitizer reserves far more address space than the worker limits
// allow, so tests relax or skip address space limits under -race.
const raceEnabled = true
