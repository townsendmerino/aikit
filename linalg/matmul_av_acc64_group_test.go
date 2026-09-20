package linalg

import (
	"math/rand/v2"
	"testing"
)

// TestMatmulAVAcc64Group_matchesUngrouped is R13's gate (1) for the AV-side grouped kernel: raw-
// bit equality against G separate MatmulAVAcc64 calls, same shape sweep as the QK-side test
// (matmul_qk_acc64_group_test.go) for the same reasons — every real group size, nKeys straddling
// every block boundary the ungrouped kernel's own blocking uses, every real hd, a nonzero headOff.
func TestMatmulAVAcc64Group_matchesUngrouped(t *testing.T) {
	groups := []int{1, 2, 4, 6, 7, 8}
	nKeysCases := []int{1, 7, 8, 9, 15, 16, 17, 127, 128, 129, 4097}
	hds := []int{64, 96, 128, 256}

	for _, G := range groups {
		for _, N := range nKeysCases {
			for _, hd := range hds {
				t.Run(caseName(G, N, hd), func(t *testing.T) {
					rng := rand.New(rand.NewPCG(uint64(G*100000+N*100+hd), 13))
					const nKV = 3
					kvh := 1 // nonzero headOff
					rowStride := nKV * hd
					headOff := kvh * hd

					scores := randVec(rng, G*N)
					vals := randVec(rng, N*rowStride)

					want := make([]float32, G*hd)
					wantAcc := make([]float64, hd)
					for g := range G {
						MatmulAVAcc64(scores[g*N:(g+1)*N], vals, want[g*hd:(g+1)*hd], wantAcc, 1, N, hd, headOff, rowStride)
					}

					got := make([]float32, G*hd)
					gotAcc := make([]float64, G*hd)
					MatmulAVAcc64Group(scores, vals, got, gotAcc, G, N, hd, headOff, rowStride)

					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("G=%d N=%d hd=%d: byte mismatch at %d: grouped=%v ungrouped=%v",
								G, N, hd, i, got[i], want[i])
						}
					}
				})
			}
		}
	}
}
