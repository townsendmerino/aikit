package linalg

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestActGroup_splitHalfKernelMatchesReference: the amd64 split-half decode path under per-32
// activation scales (the unchanged AVX2 kernel fed combined scales) agrees with the Go reference to
// float-accumulation order.
func TestActGroup_splitHalfKernelMatchesReference(t *testing.T) {
	if !splitHalfUsable() {
		t.Skip("split-half AVX2 kernel not usable on this CPU")
	}
	r := rand.New(rand.NewPCG(13, 14))
	const K, N = 512, 96
	a := agRandMat(r, K)
	a[77] = 400 // an outlier, so per-group and per-row genuinely differ
	wm := QuantizeInt4(agRandMat(r, N*K), N, K, 32)
	q4, s4, _, _ := wm.Int4()
	withActGroup(t, 32)
	ref := make([]float32, N)
	matmulW4A8GroupedRef(new(Workspace), 32, a, int4Layout{w4: q4, wS: s4, group: 32, K: K}, ref, 1, N)
	sh, ok := RepackInt4SplitHalfInPlace(append([]byte(nil), q4...), s4, N, K, 32)
	if !ok {
		t.Skip("split-half repack declined")
	}
	got := make([]float32, N)
	sh.MatmulBTW4A8Into(new(Workspace), a, got, 1)
	if e := agRelErr(got, ref); e > 1e-6 {
		t.Errorf("split-half grouped kernel vs reference: rel err %.3g", e)
	}
}

// TestActGroup_splitHalfMultiRow: M>1 through a split-half WeightMat (the per-row loop over the M=1
// AVX2 kernel) agrees with the reference.
func TestActGroup_splitHalfMultiRow(t *testing.T) {
	if !splitHalfUsable() {
		t.Skip("split-half AVX2 kernel not usable on this CPU")
	}
	r := rand.New(rand.NewPCG(17, 18))
	const M, K, N = 4, 256, 64
	a := agRandMat(r, M*K)
	a[300] = 250
	q4, s4 := QuantizeGroupsInt4(agRandMat(r, N*K), N, K, 32)
	withActGroup(t, 32)
	ref := make([]float32, M*N)
	matmulW4A8GroupedRef(new(Workspace), 32, a, int4Layout{w4: q4, wS: s4, group: 32, K: K}, ref, M, N)
	sh, ok := RepackInt4SplitHalfInPlace(append([]byte(nil), q4...), s4, N, K, 32)
	if !ok {
		t.Skip("split-half repack declined")
	}
	got := make([]float32, M*N)
	sh.MatmulBTW4A8Into(new(Workspace), a, got, M)
	if e := agRelErr(got, ref); e > 1e-6 {
		t.Errorf("split-half M=%d grouped vs reference: rel err %.3g", M, e)
	}
}

// TestActGroup_splitHalfScaledKernelOddGroups: the two-accumulator scaled kernel against the Go oracle
// sum over group counts 1..9 (the odd tail and the unrolled pairs), with a zero activation group.
func TestActGroup_splitHalfScaledKernelOddGroups(t *testing.T) {
	if !splitHalfUsable() {
		t.Skip("split-half AVX2 kernel not usable on this CPU")
	}
	r := rand.New(rand.NewPCG(29, 30))
	for nG := 1; nG <= 9; nG++ {
		K := 32 * nG
		a := agRandMat(r, K)
		for i := 0; i < 32; i++ {
			a[i] = 0 // group 0 all-zero: its scale is 0
		}
		aq := make([]int8, K)
		aS := make([]float32, nG)
		QuantizeActivationsGroupedInto(aq, aS, a, 1, K, 32)
		q4, s4 := QuantizeGroupsInt4(agRandMat(r, K), 1, K, 32)
		sh := RepackW4A8SplitHalf(q4, 1, K, 32)
		got := dotW4A8SplitHalfScaledAVX2(&aq[0], &sh[0], &s4[0], &aS[0], nG)
		var want float64
		for k := range K {
			nib := int(q4[k/2]>>(4*(k%2))) & 0xF
			want += float64(nib-8) * float64(aq[k]) * float64(s4[k/32]*aS[k/32])
		}
		if d := math.Abs(float64(got) - want); d > 1e-5*math.Abs(want)+1e-6 {
			t.Errorf("nGroups=%d: kernel %v, oracle %v", nG, got, want)
		}
	}
}
