package linalg

import (
	"math/rand/v2"
	"testing"
)

// TestDotI8Scaled32_matchesGo: the arch kernel (AVX2 on amd64) against its portable oracle, over
// group counts including 0 and 1, random and saturated (+-127, the int32 overflow corner) inputs.
func TestDotI8Scaled32_matchesGo(t *testing.T) {
	r := rand.New(rand.NewPCG(21, 22))
	for _, nG := range []int{0, 1, 2, 7, 96, 560} {
		for _, sat := range []bool{false, true} {
			a, b := make([]int8, 32*nG), make([]int8, 32*nG)
			aS := make([]float32, nG)
			for i := range a {
				if sat {
					a[i], b[i] = -127, 127
				} else {
					a[i], b[i] = int8(r.IntN(255)-127), int8(r.IntN(255)-127)
				}
			}
			for g := range aS {
				aS[g] = float32(r.Float64() * 0.05)
			}
			got, want := dotI8Scaled32(a, b, aS), dotI8Scaled32Go(a, b, aS)
			if d := float64(got - want); d*d > 1e-10*float64(want*want)+1e-12 {
				t.Errorf("nG=%d sat=%v: kernel %v, Go %v", nG, sat, got, want)
			}
		}
	}
}

func TestDotI8Scaled32x2_matchesGo(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2")
	}
	r := rand.New(rand.NewPCG(41, 42))
	for _, nG := range []int{0, 1, 2, 3, 7, 48, 96} {
		for _, sat := range []bool{false, true} {
			a := make([]int8, 32*nG)
			b0, b1 := make([]int8, 32*nG), make([]int8, 32*nG)
			aS := make([]float32, nG)
			for i := range a {
				if sat {
					a[i] = 127
					b0[i], b1[i] = -127, 127
				} else {
					a[i] = int8(r.IntN(255) - 127)
					b0[i] = int8(r.IntN(255) - 127)
					b1[i] = int8(r.IntN(255) - 127)
				}
			}
			for g := range aS {
				aS[g] = float32(r.Float64() * 0.05)
			}
			var got0, got1 float32
			if nG > 0 {
				got0, got1 = dotI8Scaled32x2AVX2(&a[0], &b0[0], &b1[0], &aS[0], nG)
			}
			want0 := dotI8Scaled32Go(a, b0, aS)
			want1 := dotI8Scaled32Go(a, b1, aS)
			if d0 := float64(got0 - want0); d0*d0 > 1e-10*float64(want0*want0)+1e-12 {
				t.Errorf("nG=%d sat=%v col0: got %v, want %v", nG, sat, got0, want0)
			}
			if d1 := float64(got1 - want1); d1*d1 > 1e-10*float64(want1*want1)+1e-12 {
				t.Errorf("nG=%d sat=%v col1: got %v, want %v", nG, sat, got1, want1)
			}
		}
	}
}

// TestActGroup_w8a8KernelMatchesReference: MatmulBTW8A8Into and MatmulBTW8A8Batch under per-32
// activations (the dotI8Scaled32 path) agree with the Go reference, at M=2 with an outlier.
func TestActGroup_w8a8KernelMatchesReference(t *testing.T) {
	r := rand.New(rand.NewPCG(23, 24))
	const M, K, N = 2, 512, 80
	a := agRandMat(r, M*K)
	a[100] = 350
	wm := QuantizeInt8(agRandMat(r, N*K), N, K, true)
	q8, s8, _, _ := wm.Int8()
	withActGroup(t, 32)
	ref := make([]float32, M*N)
	matmulW8A8GroupedRef(new(Workspace), 32, a, q8, s8, ref, M, K, N)
	got := make([]float32, M*N)
	MatmulBTW8A8Into(new(Workspace), a, q8, s8, got, M, K, N)
	batch := make([]float32, M*N)
	MatmulBTW8A8Batch(new(Workspace), a, M, K, []W8A8Op{{BQ: q8, Scales: s8, Dst: batch, N: N}})
	for name, v := range map[string][]float32{"Into": got, "Batch": batch} {
		if e := agRelErr(v, ref); e > 1e-6 {
			t.Errorf("W8A8 %s grouped vs reference: rel err %.3g", name, e)
		}
	}
}
