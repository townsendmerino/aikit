package linalg

import (
	"math/rand/v2"
	"testing"
)

// TestActGroup_row4LayoutAgrees: the arm64 row4 layout (split-half + 4-row interleave) must give the
// canonical layout's grouped reference result bit-for-bit, through MatmulBTW4A8Row4Into and through
// a repacked-only WeightMat.
func TestActGroup_row4LayoutAgrees(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 10))
	const K, N = 128, 16
	a := agRandMat(r, K)
	wm := QuantizeInt4(agRandMat(r, N*K), N, K, 32)
	q4, s4, _, _ := wm.Int4()
	withActGroup(t, 32)
	canon := make([]float32, N)
	MatmulBTW4A8Into(new(Workspace), a, q4, s4, canon, 1, K, N, 32)
	r4, r4s := RepackW4A8Row4(q4, N, K, 32), RepackW4A8Row4Scales(s4, N, K, 32)
	viaFree := make([]float32, N)
	MatmulBTW4A8Row4Into(new(Workspace), a, r4, r4s, viaFree, 1, K, N, 32)
	only, ok := WrapInt4Row4Only(r4, r4s, N, K, 32)
	if !ok {
		t.Fatal("WrapInt4Row4Only declined")
	}
	viaMethod := make([]float32, N)
	only.MatmulBTW4A8Into(new(Workspace), a, viaMethod, 1)
	for i := range canon {
		if viaFree[i] != canon[i] || viaMethod[i] != canon[i] {
			t.Fatalf("[%d]: canonical %v, row4 free %v, row4 method %v — want bit-identical", i, canon[i], viaFree[i], viaMethod[i])
		}
	}
}
