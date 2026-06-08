package linalg

import (
	"fmt"
	"math/rand"
	"testing"
)

// BenchmarkQ4vsQ8 compares the int4 and int8 weight matmuls at decode (M=1) and
// prefill (M=64) on the same shapes, to track the int4 perf fix (it was ~28×
// slower than int8 in goinfer's 1.5B decode; the gate is within ~1.5–2×). Q4
// dequants each weight row once and reuses it across M (column-outer); Q8
// re-widens per row, so Q4's advantage grows with M.
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
		}
	}
}
