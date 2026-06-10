package encoder

import "testing"

func TestPoolOne(t *testing.T) {
	// 3 tokens × 2 dims: [1,2], [3,4], [5,6].
	seq := []float32{1, 2, 3, 4, 5, 6}

	if got := poolOne(seq, 3, 2, poolCLS); got[0] != 1 || got[1] != 2 {
		t.Errorf("CLS = %v, want [1 2]", got)
	}
	if got := poolOne(seq, 3, 2, ""); got[0] != 1 || got[1] != 2 {
		t.Errorf("zero-value pooling = %v, want CLS [1 2]", got)
	}
	// Mean over all 3: [(1+3+5)/3, (2+4+6)/3] = [3, 4].
	if got := poolOne(seq, 3, 2, poolMean); got[0] != 3 || got[1] != 4 {
		t.Errorf("Mean = %v, want [3 4]", got)
	}
	// Masking (the batched path): only the first 2 real tokens → [(1+3)/2,(2+4)/2]=[2,3].
	if got := poolOne(seq, 2, 2, poolMean); got[0] != 2 || got[1] != 3 {
		t.Errorf("masked Mean (L=2) = %v, want [2 3]", got)
	}
	// Degenerate (no tokens) → stable zero vector.
	if got := poolOne(seq, 0, 2, poolMean); got[0] != 0 || got[1] != 0 {
		t.Errorf("L=0 = %v, want [0 0]", got)
	}
	// Result is a fresh slice — mutating it must not alias the input.
	r := poolOne(seq, 3, 2, poolCLS)
	r[0] = 99
	if seq[0] != 1 {
		t.Errorf("CLS aliased the input: seq[0] = %v", seq[0])
	}
}
