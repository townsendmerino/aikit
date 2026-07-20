package encoder

import "testing"

// TestClampTokenID (§2.3): an out-of-range id maps to row 0, never a panicking
// index — the small-vocab crash the old id=100 fallback reintroduced in the
// batched/Q8/token paths. Every embed-gather site routes through this helper.
func TestClampTokenID(t *testing.T) {
	const vocab = 4 // the repo's own tiny-fixture size; id 100 would index [400:404]
	cases := []struct {
		id   int32
		want int
	}{
		{0, 0}, {3, 3}, // in range
		{4, 0}, {100, 0}, {-1, 0}, {1 << 30, 0}, // out of range → row 0
	}
	for _, c := range cases {
		if got := clampTokenID(c.id, vocab); got != c.want {
			t.Errorf("clampTokenID(%d, %d) = %d, want %d", c.id, vocab, got, c.want)
		}
		if got := clampTokenID(c.id, vocab); got < 0 || got >= vocab {
			t.Errorf("clampTokenID(%d, %d) = %d is not a valid [0,%d) row", c.id, vocab, got, vocab)
		}
	}
}
