package linalg

import (
	"math/rand"
	"testing"
)

// TestMatmulBTQ8Into_bitIdentical: the Into variant (any Workspace) must match the
// allocating wrapper exactly (audit #14).
func TestMatmulBTQ8Into_bitIdentical(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, sh := range []struct{ M, K, N int }{{1, 768, 512}, {4, 256, 300}, {8, 512, 64}} {
		a := make([]float32, sh.M*sh.K)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		bq := randI8(sh.N * sh.K)
		scales := make([]float32, sh.N)
		for i := range scales {
			scales[i] = rng.Float32() + 0.1
		}
		want := make([]float32, sh.M*sh.N)
		matmulBTQ8(a, bq, scales, want, sh.M, sh.K, sh.N)
		var ws Workspace
		got := make([]float32, sh.M*sh.N)
		matmulBTQ8Into(&ws, a, bq, scales, got, sh.M, sh.K, sh.N)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("M=%d K=%d N=%d: got[%d]=%v want %v", sh.M, sh.K, sh.N, i, got[i], want[i])
			}
		}
	}
}

// BenchmarkMatmulBTQ8_decode: serial decode shape (M=1). Reused Workspace vs the
// allocating wrapper.
func BenchmarkMatmulBTQ8Into_reusedWS(b *testing.B) {
	const M, K, N = 1, 3072, 8
	a := make([]float32, M*K)
	bq := randI8(N * K)
	scales := make([]float32, N)
	dst := make([]float32, M*N)
	var ws Workspace
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		matmulBTQ8Into(&ws, a, bq, scales, dst, M, K, N)
	}
}
func BenchmarkMatmulBTQ8_wrapper(b *testing.B) {
	const M, K, N = 1, 3072, 8
	a := make([]float32, M*K)
	bq := randI8(N * K)
	scales := make([]float32, N)
	dst := make([]float32, M*N)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		matmulBTQ8(a, bq, scales, dst, M, K, N)
	}
}
