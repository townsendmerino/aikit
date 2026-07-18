package linalg

import (
	"math/rand"
	"testing"
)

// BenchmarkW4A8_allocVsInto shows MatmulBTW4A8Into is zero-alloc in steady
// state (reused Workspace) vs the allocating MatmulBTW4A8 — the int4 decode
// equivalent of the W8A8 Into win.
func BenchmarkW4A8_allocVsInto(b *testing.B) {
	const M, K, N, group = 1, 2048, 2048, 32
	rng := rand.New(rand.NewSource(9))
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	w := make([]float32, N*K)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	q4, q4s := QuantizeGroupsInt4(w, N, K, group)
	dst := make([]float32, M*N)

	b.Run("Alloc", func(b *testing.B) {
		b.ReportAllocs()
		for range b.N {
			MatmulBTW4A8(a, q4, q4s, dst, M, K, N, group)
		}
	})
	b.Run("Into", func(b *testing.B) {
		var ws Workspace
		b.ReportAllocs()
		for range b.N {
			MatmulBTW4A8Into(&ws, a, q4, q4s, dst, M, K, N, group)
		}
	})
}
