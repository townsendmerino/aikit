//go:build amd64

package linalg

import (
	"math/rand/v2"
	"testing"
)

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
