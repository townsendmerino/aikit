package linalg

import (
	"math/rand/v2"
	"testing"
)

// TestActGroup_row4LayoutAgrees: the arm64 row4 layout (split-half + 4-row interleave), through
// MatmulBTW4A8Row4Into and through a repacked-only WeightMat, runs the per-32 kernel path at M=1: the
// two entry points agree bit-for-bit with each other, and with the canonical layout's Go reference to
// accumulation order.
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
		if viaFree[i] != viaMethod[i] {
			t.Fatalf("[%d]: row4 free %v != row4 method %v — want bit-identical", i, viaFree[i], viaMethod[i])
		}
	}
	if e := agRelErr(viaFree, canon); e > 1e-6 {
		t.Errorf("row4 kernel path vs canonical reference: rel err %.3g", e)
	}
}

// TestActGroup_row4KernelMatchesReference: the arm64 row4 decode kernels under per-32 activation
// scales (fed combined scales), with the S-05 fold on and off, agree with the Go reference to
// accumulation order.
func TestActGroup_row4KernelMatchesReference(t *testing.T) {
	r := rand.New(rand.NewPCG(15, 16))
	const K, N = 512, 96
	a := agRandMat(r, K)
	a[77] = 400
	wm := QuantizeInt4(agRandMat(r, N*K), N, K, 32)
	q4, s4, _, _ := wm.Int4()
	withActGroup(t, 32)
	ref := make([]float32, N)
	matmulW4A8GroupedRef(new(Workspace), 32, a, int4Layout{w4: q4, wS: s4, group: 32, K: K}, ref, 1, N)
	r4, r4s := RepackW4A8Row4(q4, N, K, 32), RepackW4A8Row4Scales(s4, N, K, 32)
	for _, fold := range []bool{true, false} {
		prev := W4A8RowFold()
		SetW4A8RowFold(fold)
		got := make([]float32, N)
		MatmulBTW4A8Row4Into(new(Workspace), a, r4, r4s, got, 1, K, N, 32)
		SetW4A8RowFold(prev)
		if e := agRelErr(got, ref); e > 1e-6 {
			t.Errorf("fold=%v: row4 grouped kernel vs reference: rel err %.3g", fold, e)
		}
	}
}

// TestActGroup_row4MultiRow: M>1 through a repacked-only row4 WeightMat (the per-row loop over the
// M=1 row4 kernel) agrees with the reference.
func TestActGroup_row4MultiRow(t *testing.T) {
	r := rand.New(rand.NewPCG(19, 20))
	const M, K, N = 3, 256, 32
	a := agRandMat(r, M*K)
	a[400] = 250
	q4, s4 := QuantizeGroupsInt4(agRandMat(r, N*K), N, K, 32)
	f16RoundScales(s4) // the WeightMat stores binary16 scales
	withActGroup(t, 32)
	ref := make([]float32, M*N)
	matmulW4A8GroupedRef(new(Workspace), 32, a, int4Layout{w4: q4, wS: s4, group: 32, K: K}, ref, M, N)
	only, ok := WrapInt4Row4Only(RepackW4A8Row4(q4, N, K, 32), RepackW4A8Row4Scales(s4, N, K, 32), N, K, 32)
	if !ok {
		t.Fatal("WrapInt4Row4Only declined")
	}
	got := make([]float32, M*N)
	only.MatmulBTW4A8Into(new(Workspace), a, got, M)
	if e := agRelErr(got, ref); e > 1e-6 {
		t.Errorf("row4 M=%d grouped vs reference: rel err %.3g", M, e)
	}
}
