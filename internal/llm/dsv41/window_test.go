package dsv41

import (
	"slices"
	"testing"
)

// The window's index layout decides how attention's online softmax groups
// its entries into blocks of 64, so the rounding. The tiny fixture has one
// block and cannot tell; these are the reference's get_window_topk_idxs
// answers (positions when prefilling, ring slots when decoding; -1 empty).
func TestWindowRowsMatchReference(t *testing.T) {
	for _, c := range []struct {
		prefill bool
		win, n  int // window; prompt length when prefilling
		pos     int
		want    []int
	}{
		{true, 8, 5, 0, []int{0, -1, -1, -1, -1}},
		{true, 8, 5, 2, []int{0, 1, 2, -1, -1}},
		{true, 8, 5, 4, []int{0, 1, 2, 3, 4}},
		{true, 8, 23, 0, []int{0, -1, -1, -1, -1, -1, -1, -1}},
		{true, 8, 23, 2, []int{0, 1, 2, -1, -1, -1, -1, -1}},
		{true, 8, 23, 4, []int{0, 1, 2, 3, 4, -1, -1, -1}},
		{true, 8, 23, 7, []int{0, 1, 2, 3, 4, 5, 6, 7}},
		{true, 8, 23, 12, []int{5, 6, 7, 8, 9, 10, 11, 12}},
		{true, 8, 23, 22, []int{15, 16, 17, 18, 19, 20, 21, 22}},
		{false, 8, 0, 3, []int{-1, -1, -1, -1, 0, 1, 2, 3}},
		{false, 8, 0, 23, []int{0, 1, 2, 3, 4, 5, 6, 7}},
		{false, 8, 0, 15, []int{0, 1, 2, 3, 4, 5, 6, 7}},
	} {
		// Each ring slot holds its own number; prefill rows name positions,
		// held in slot position mod window.
		ring := make([][]float32, c.win)
		for i := range ring {
			ring[i] = []float32{float32(i)}
		}
		s := &State{prefillN: c.n}
		var got []int
		for _, r := range s.windowRows(ring, c.pos) {
			if r == nil {
				got = append(got, -1)
			} else {
				got = append(got, int(r[0]))
			}
		}
		want := slices.Clone(c.want)
		if c.prefill {
			for i, p := range want {
				if p >= 0 {
					want[i] = p % c.win
				}
			}
		}
		if !slices.Equal(got, want) {
			t.Errorf("prefill %v n %d pos %d: slots %v, reference %v", c.prefill, c.n, c.pos, got, want)
		}
	}
}
