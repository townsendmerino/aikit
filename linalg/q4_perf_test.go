package linalg

import (
	"fmt"
	"math/rand"
	"testing"
)

// BenchmarkQ4vsQ8 compares the int4/int8 weight matmuls at decode (M=1) and
// prefill (M=64) on the same shapes. Q4 (f32 activations) is the prefill path —
// the v1.0.1 column-outer fix made it a win at M>1 but it stays f32-dequant-
// bound at M=1. W4A8 (int8 activations, fused SDOT kernel) is the M=1 decode
// path: ~2× of W8A8 and ~20× faster than Q4 at decode. W8A8 is the int8 ceiling.
func BenchmarkQ4vsQ8(b *testing.B) {
	const group = 32
	rng := rand.New(rand.NewSource(7))
	rf := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	for _, s := range []struct{ K, N int }{
		{2048, 2048},
		{1536, 8960}, // Qwen2.5-1.5B MLP up/gate shape
	} {
		w := rf(s.N * s.K)
		q4p, q4s := QuantizeGroupsInt4(w, s.N, s.K, group)
		q8q, q8s := QuantizeRowsInt8(w, s.N, s.K)
		for _, M := range []int{1, 64} {
			a := rf(M * s.K)
			dst := make([]float32, M*s.N)
			b.Run(fmt.Sprintf("Q4/K%d_N%d/M%d", s.K, s.N, M), func(b *testing.B) {
				b.ResetTimer()
				for range b.N {
					MatmulBTQ4(a, q4p, q4s, dst, M, s.K, s.N, group)
				}
			})
			b.Run(fmt.Sprintf("Q8/K%d_N%d/M%d", s.K, s.N, M), func(b *testing.B) {
				b.ResetTimer()
				for range b.N {
					MatmulBTQ8(a, q8q, q8s, dst, M, s.K, s.N)
				}
			})
			b.Run(fmt.Sprintf("W8A8/K%d_N%d/M%d", s.K, s.N, M), func(b *testing.B) {
				b.ResetTimer()
				for range b.N {
					MatmulBTW8A8(a, q8q, q8s, dst, M, s.K, s.N)
				}
			})
			b.Run(fmt.Sprintf("W4A8/K%d_N%d/M%d", s.K, s.N, M), func(b *testing.B) {
				b.ResetTimer()
				for range b.N {
					MatmulBTW4A8(a, q4p, q4s, dst, M, s.K, s.N, group)
				}
			})
		}
	}
}
