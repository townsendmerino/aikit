package attn

import (
	"math/rand/v2"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestMatmulAVAcc64_exactMatchesStrided proves MatmulAVAcc64 (keys-outer/
// dims-inner, goinfer's A1 move (c) — docs/task-attention-decode-cost.md) is
// bit-identical to the existing MatmulBTAcc64Strided at attendBatchedHeads's
// AV call shape, across every nKeys residue mod 4 (relevant once move (b)
// interleaves at that width) and several real hd values (64/80/96/128 cover
// the model families in goinfer's registry). Exact `==`, not rel-err — the
// two loop nestings must produce identical adds to each output accumulator.
func TestMatmulAVAcc64_exactMatchesStrided(t *testing.T) {
	shapes := []struct {
		name              string
		M, nKeys, hd, nKV int
	}{
		{"decode M=1 hd=128 nKeys=130", 1, 130, 128, 2},
		{"decode M=1 hd=128 nKeys=131", 1, 131, 128, 2},
		{"decode M=1 hd=128 nKeys=132", 1, 132, 128, 2},
		{"decode M=1 hd=128 nKeys=133", 1, 133, 128, 2},
		{"decode M=1 hd=64 nKeys=300", 1, 300, 64, 8},
		{"decode M=1 hd=80 nKeys=97", 1, 97, 80, 4},
		{"decode M=1 hd=96 nKeys=1", 1, 1, 96, 2},
		{"prefill M=8 hd=128 nKeys=300", 8, 300, 128, 4},
		{"prefill M=5 odd hd=16 nKeys=40", 5, 40, 16, 3},
		{"long-ctx M=1 hd=128 nKeys=8192", 1, 8192, 128, 2},
	}
	for _, s := range shapes {
		t.Run(s.name, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(11, 12))
			kvDim := s.nKV * s.hd
			for kvh := 0; kvh < s.nKV; kvh++ {
				scores := randVec(rng, s.M*s.nKeys)
				vals := randVec(rng, s.nKeys*kvDim)

				want := make([]float32, s.M*s.hd)
				linalg.MatmulBTAcc64Strided(scores, vals, want, s.M, s.nKeys, s.hd, kvh*s.hd, 1, kvDim)

				got := make([]float32, s.M*s.hd)
				acc := make([]float64, s.hd)
				MatmulAVAcc64(scores, vals, got, acc, s.M, s.nKeys, s.hd, kvh*s.hd, kvDim)

				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("kvh=%d: byte mismatch at %d: MatmulAVAcc64=%v MatmulBTAcc64Strided=%v",
							kvh, i, got[i], want[i])
					}
				}
			}
		})
	}
}
