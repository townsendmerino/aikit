package linalg

import (
	"math/rand/v2"
	"testing"
)

// TestMatmulBT_MConsistent gates the M-invariance contract: MatmulBT's per-output
// f32 result must NOT depend on M — a row computed alone (M=1) must be BIT-IDENTICAL
// to the same row computed inside a batch (M>1). Same weights, same a-row → same
// logits, regardless of how many rows are computed together.
//
// Why it matters (found cross-repo, goinfer): same-model speculative decoding must
// accept 100% — the draft proposes a token one-at-a-time (M=1) and the target
// verifies a batch (M=K); identical weights must give identical argmax. If MatmulBT
// is M-dependent, the M=1 and M=K logits differ by the f32 reassociation (~1e-5),
// which flips the occasional greedy argmax and drops speculative acceptance below 1.0.
//
// The shapes straddle the former ~4M-MAC naive/blocked cutoff: the M=M0 call took
// the blocked+register-tiled kernel, while each M=1 row historically fell to the
// naive dot-per-output span — a different reduction order. That cross-threshold
// switch was the regression this guards (the threshold is now gone — all M route
// through the blocked kernel). Cases also cover the <8 / <4 column tails, a K with
// a 4-remainder, and the large-K packed path, so the gate also pins blockedFill's
// own internal M-invariance (paired Dot2x8 rows vs the odd Dot8x4 row).
func TestMatmulBT_MConsistent(t *testing.T) {
	rng := rand.New(rand.NewPCG(0x5eed, 0x1eaf))
	rv := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = rng.Float32()*2 - 1
		}
		return v
	}
	cases := []struct {
		name    string
		M, K, N int
	}{
		{"square_blocked_vs_naive", 64, 512, 512}, // single 16.8M (blocked); row 262K (naive)
		{"col_tail_4", 48, 300, 300},              // N=300 → 8-groups to 296, 4-col tail
		{"col_tail_scalar", 48, 300, 302},         // N=302 → +2 scalar tail
		{"k_remainder_2", 64, 510, 512},           // K=510 → 4-lane body + 2-col k-tail
		{"odd_M", 7, 896, 896},                    // odd M: paired rows + one odd row
		{"packed_large_K", 4, 2048, 512},          // K≥packKThreshold → packedFill path
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			a := rv(c.M * c.K)
			b := rv(c.N * c.K)

			batched := make([]float32, c.M*c.N)
			MatmulBT(a, b, batched, c.M, c.K, c.N)

			row := make([]float32, c.N)
			for i := range c.M {
				MatmulBT(a[i*c.K:(i+1)*c.K], b, row, 1, c.K, c.N)
				for j := range c.N {
					if got, want := batched[i*c.N+j], row[j]; got != want {
						t.Fatalf("M-dependent: out[%d,%d] batched=%v M=1=%v (diff %v); "+
							"MatmulBT must be bit-identical across M", i, j, got, want, got-want)
					}
				}
			}
		})
	}
}
