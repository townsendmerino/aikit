package linalg

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestDot2x8_equalsDot8x4 is the correctness proof: the 2×8 microkernel computes each
// dot in the SAME accumulation order as the proven 1×8 Dot8x4 (same b-rows, same 4-lane
// FMLA sequence, same final reduction) — it only folds 2 a-rows into one call. So its
// per-row results must equal two Dot8x4 calls bit-for-bit on arm64 (both asm); we assert
// a tight relative bound to stay portable to the generic fallback. This bit-identity is
// what keeps the encoder golden cosine unchanged when the kernel is wired in.
func TestDot2x8_equalsDot8x4(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	rv := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = rng.Float32()*2 - 1
		}
		return v
	}
	reduce := func(s []float32) float32 { return s[0] + s[1] + s[2] + s[3] }
	for _, K := range []int{4, 8, 64, 768, 3072} {
		n4 := K / 4
		a0, a1 := rv(K), rv(K)
		b := make([][]float32, 8)
		for i := range b {
			b[i] = rv(K)
		}
		bp := func(i int) *float32 { return &b[i][0] }

		var s2 [64]float32
		Dot2x8(&a0[0], &a1[0], bp(0), bp(1), bp(2), bp(3), bp(4), bp(5), bp(6), bp(7), n4, &s2)
		var s8a, s8b [32]float32
		Dot8x4(&a0[0], bp(0), bp(1), bp(2), bp(3), bp(4), bp(5), bp(6), bp(7), n4, &s8a)
		Dot8x4(&a1[0], bp(0), bp(1), bp(2), bp(3), bp(4), bp(5), bp(6), bp(7), n4, &s8b)

		for j := range 8 {
			want0 := reduce(s8a[j*4 : j*4+4])
			got0 := reduce(s2[j*4 : j*4+4])
			want1 := reduce(s8b[j*4 : j*4+4])
			got1 := reduce(s2[(8+j)*4 : (8+j)*4+4])
			rel := func(g, w float32) float64 {
				return math.Abs(float64(g-w)) / (math.Abs(float64(w)) + 1e-9)
			}
			if rel(got0, want0) > 1e-6 {
				t.Errorf("K=%d a0·b%d: 2x8=%v 8x4=%v", K, j, got0, want0)
			}
			if rel(got1, want1) > 1e-6 {
				t.Errorf("K=%d a1·b%d: 2x8=%v 8x4=%v", K, j, got1, want1)
			}
		}
	}
}
