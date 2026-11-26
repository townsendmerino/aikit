package attn

import (
	"math/rand/v2"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestMatmulQKAcc64_exactMatchesStrided proves MatmulQKAcc64 (8-wide
// interleaved accumulators, goinfer's A1 move (b)) is bit-identical to
// MatmulBTAcc64Strided at attention's QKᵀ call shape, across every N (nKeys)
// residue mod 8 and several real hd values.
func TestMatmulQKAcc64_exactMatchesStrided(t *testing.T) {
	shapes := []struct {
		name              string
		M, nKeys, hd, nKV int
	}{
		{"decode M=1 hd=128 nKeys=128", 1, 128, 128, 2},
		{"decode M=1 hd=128 nKeys=129", 1, 129, 128, 2},
		{"decode M=1 hd=128 nKeys=130", 1, 130, 128, 2},
		{"decode M=1 hd=128 nKeys=131", 1, 131, 128, 2},
		{"decode M=1 hd=128 nKeys=132", 1, 132, 128, 2},
		{"decode M=1 hd=128 nKeys=133", 1, 133, 128, 2},
		{"decode M=1 hd=128 nKeys=134", 1, 134, 128, 2},
		{"decode M=1 hd=128 nKeys=135", 1, 135, 128, 2},
		{"decode M=1 hd=64 nKeys=1", 1, 1, 64, 8},
		{"decode M=1 hd=64 nKeys=2", 1, 2, 64, 8},
		{"decode M=1 hd=64 nKeys=7", 1, 7, 64, 8},
		{"decode M=1 hd=80 nKeys=97", 1, 97, 80, 4},
		{"prefill M=8 hd=128 nKeys=300", 8, 300, 128, 4},
		{"prefill M=5 odd hd=16 nKeys=41", 5, 41, 16, 3},
		{"long-ctx M=1 hd=128 nKeys=8192", 1, 8192, 128, 2},
	}
	for _, s := range shapes {
		t.Run(s.name, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(21, 22))
			kvDim := s.nKV * s.hd
			for kvh := 0; kvh < s.nKV; kvh++ {
				q := randVec(rng, s.M*s.hd)
				keys := randVec(rng, s.nKeys*kvDim)

				want := make([]float32, s.M*s.nKeys)
				linalg.MatmulBTAcc64Strided(q, keys, want, s.M, s.hd, s.nKeys, kvh*s.hd, kvDim, 1)

				got := make([]float32, s.M*s.nKeys)
				MatmulQKAcc64(q, keys, got, s.M, s.hd, s.nKeys, kvh*s.hd, kvDim)

				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("kvh=%d: byte mismatch at %d: MatmulQKAcc64=%v MatmulBTAcc64Strided=%v",
							kvh, i, got[i], want[i])
					}
				}
			}
		})
	}
}
