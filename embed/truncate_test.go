package embed

import (
	"math"
	"testing"
)

func TestTruncate(t *testing.T) {
	v := []float32{3, 4, 12, 0} // norm 13 over first 3
	got := Truncate(v, 3)

	if len(got) != 3 {
		t.Fatalf("len = %d, want 3", len(got))
	}
	// Input unmodified.
	if v[0] != 3 || v[2] != 12 {
		t.Errorf("input mutated: %v", v)
	}
	// Renormalized to unit length: {3,4,12}/13.
	var sq float64
	for _, x := range got {
		sq += float64(x) * float64(x)
	}
	if math.Abs(math.Sqrt(sq)-1) > 1e-6 {
		t.Errorf("not unit norm: |got| = %.6f", math.Sqrt(sq))
	}
	if math.Abs(float64(got[0])-3.0/13) > 1e-6 {
		t.Errorf("got[0] = %.6f, want %.6f", got[0], 3.0/13)
	}
}

func TestTruncate_clampAndDegenerate(t *testing.T) {
	v := []float32{1, 0, 0}
	if got := Truncate(v, 99); len(got) != 3 { // dim > len clamps
		t.Errorf("over-len: len %d, want 3", len(got))
	}
	if got := Truncate(v, 0); len(got) != 0 { // dim 0
		t.Errorf("dim 0: len %d, want 0", len(got))
	}
	if got := Truncate(v, -5); len(got) != 0 { // negative clamps to 0
		t.Errorf("neg dim: len %d, want 0", len(got))
	}
	// All-zero prefix → zeroed (l2Normalize degenerate path), not NaN.
	got := Truncate([]float32{0, 0, 5}, 2)
	if got[0] != 0 || got[1] != 0 {
		t.Errorf("degenerate prefix not zeroed: %v", got)
	}
}
