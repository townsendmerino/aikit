package linalg

import (
	"math/rand/v2"
	"strconv"
	"testing"
)

// TestMatmulQKAcc64Group_matchesUngrouped is R13's (docs/tasks/red-october.md, goinfer) gate (1):
// raw-bit equality of MatmulQKAcc64Group against G separate MatmulQKAcc64 calls, at G ∈
// {1,2,4,6,7,8} (every real group size this repo's models use, plus 1 and 8 as bounds), nKeys
// straddling every block boundary the ungrouped kernel's own 8-wide/16-key NEON paths have, hd ∈
// {64,96,128,256} (every real head width in use), and a nonzero bOff to prove the group shares
// the SAME kv-head offset correctly rather than only working at offset 0.
func TestMatmulQKAcc64Group_matchesUngrouped(t *testing.T) {
	groups := []int{1, 2, 4, 6, 7, 8}
	nKeysCases := []int{1, 7, 8, 9, 15, 16, 17, 127, 128, 129, 4097}
	hds := []int{64, 96, 128, 256}

	for _, G := range groups {
		for _, N := range nKeysCases {
			for _, K := range hds {
				t.Run(caseName(G, N, K), func(t *testing.T) {
					rng := rand.New(rand.NewPCG(uint64(G*100000+N*100+K), 7))
					const nKV = 3
					kvh := 1 // nonzero bOff: kv head 1 of 3, not head 0
					kvDim := nKV * K
					bOff := kvh * K

					a := randVec(rng, G*K)
					bMat := randVec(rng, N*kvDim)

					want := make([]float32, G*N)
					for g := range G {
						MatmulQKAcc64(a[g*K:(g+1)*K], bMat, want[g*N:(g+1)*N], 1, K, N, bOff, kvDim)
					}

					got := make([]float32, G*N)
					MatmulQKAcc64Group(a, bMat, got, G, K, N, bOff, kvDim)

					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("G=%d N=%d K=%d: byte mismatch at %d: grouped=%v ungrouped=%v",
								G, N, K, i, got[i], want[i])
						}
					}
				})
			}
		}
	}
}

func caseName(G, N, K int) string {
	return "G=" + strconv.Itoa(G) + "_N=" + strconv.Itoa(N) + "_K=" + strconv.Itoa(K)
}
