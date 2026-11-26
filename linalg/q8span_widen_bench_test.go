package linalg

import (
	"math/rand"
	"testing"
)

// q8HeadShapes are LM-head decode shapes at M=1: K = hidden dim, N = vocab. 152k and 262k are
// the real large-vocab cases the P2 opportunity is about.
func q8HeadShapes() []struct {
	K, N int
	name string
} {
	return []struct {
		K, N int
		name string
	}{
		{2048, 152064, "K2048_N152064"},
		{3584, 152064, "K3584_N152064"},
		{2048, 262144, "K2048_N262144"},
	}
}

// BenchmarkQ8Head measures the whole production path — matmulBTQ8Into at M=1 on the LM-head
// shapes — so perfgate can A/B the widen substitution end to end (parallel path included). The
// sub-benchmark name carries K/N so perfgate keys on the shape.
func BenchmarkQ8Head(b *testing.B) {
	for _, s := range q8HeadShapes() {
		rng := rand.New(rand.NewSource(2))
		K, N := s.K, s.N
		a := make([]float32, K)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		bQ := make([]int8, N*K)
		for i := range bQ {
			bQ[i] = int8(rng.Intn(255) - 127)
		}
		bScales := make([]float32, N)
		for i := range bScales {
			bScales[i] = 0.01
		}
		dst := make([]float32, N)
		var ws Workspace
		b.Run(s.name, func(b *testing.B) {
			for b.Loop() {
				matmulBTQ8Into(&ws, a, bQ, bScales, dst, 1, K, N)
			}
		})
	}
}

// BenchmarkQ8Span_widenShare decomposes q8Span's per-token work at M=1 into its two halves —
// the scalar int8→f32 widen it does today, and the SIMD dotF32 — so the widen's ACTUAL share
// can be read before any change. widen_scalar vs dot_simd gives the share; widen_scalar minus
// widen_simd is the time the substitution would save. Measured serially: the share is
// parallelism-independent (the parallel path splits the same widen+dot across workers).
func BenchmarkQ8Span_widenShare(b *testing.B) {
	for _, s := range q8HeadShapes() {
		rng := rand.New(rand.NewSource(1))
		K, N := s.K, s.N
		a := make([]float32, K)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		bQ := make([]int8, N*K)
		for i := range bQ {
			bQ[i] = int8(rng.Intn(255) - 127)
		}
		bScales := make([]float32, N)
		for i := range bScales {
			bScales[i] = 0.01
		}
		deq := make([]float32, K)
		dst := make([]float32, N)

		// widen_scalar: exactly what q8Span does today, over all N rows.
		b.Run(s.name+"/widen_scalar", func(b *testing.B) {
			for b.Loop() {
				for j := range N {
					bq := bQ[j*K : j*K+K]
					for k := range K {
						deq[k] = float32(bq[k])
					}
				}
			}
		})
		// widen_simd: the proposed substitution (SIMD widen, scale 1.0), over all N rows.
		b.Run(s.name+"/widen_simd", func(b *testing.B) {
			for b.Loop() {
				for j := range N {
					dequantRowInt8(deq, bQ[j*K:j*K+K], 1.0)
				}
			}
		})
		// dot_simd: the SIMD dotF32 q8Span does, over all N rows (deq fixed — dot cost is
		// independent of its contents).
		for k := range deq {
			deq[k] = float32(bQ[k])
		}
		b.Run(s.name+"/dot_simd", func(b *testing.B) {
			for b.Loop() {
				for j := range N {
					dst[j] = dotF32(a, deq) * bScales[j]
				}
			}
		})
	}
}
