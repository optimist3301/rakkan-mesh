package membership

import "testing"

// BenchmarkSolve20 solves 20-bit proofs on all CPUs; 24 bits takes 16 times
// as long on average.
func BenchmarkSolve20(b *testing.B) {
	_, n := mustRoot(&testing.T{})
	for b.Loop() {
		_, id := newKey(&testing.T{})
		if _, err := n.Solve(b.Context(), id, 20); err != nil {
			b.Fatal(err)
		}
	}
}
