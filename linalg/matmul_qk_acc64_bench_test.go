package linalg

import "testing"

// BenchmarkMatmulQKAcc64 is BenchmarkMatmulAVAcc64's missing twin. S-04 quotes QK
// timings (3,463 ns at depth 130, 219,500 at 8192, ≈4.8 GMAC/s) from goinfer's
// campaign records, and unlike AV the kernel never acquired a local row — so the
// NEON port added in S-04 step 2 had nothing in this repo to be measured against.
//
// Deliberately the same shape family, depth set and reporting as the AV bench, so
// the two halves of one attention token can be read side by side rather than
// against different conventions: hd=128, M=1 (decode — the caller runs one of
// these per (head, token)), depths 130 / 2048 / 8192.
//
// Note the axis difference from AV, which is easy to misread. Here the DEPTH is
// N (the key count) and K is the head dim, because QK computes q·Kᵀ; in the AV
// bench the depth is the reduction length. So this bench's byte count grows with
// depth for the same reason but through a different argument, and its rowStride
// is the key stride in the KV cache.
//
// Read it as a ROW, not a verdict: single-core, and it says nothing about the
// 6-way head fan-out that S-02 finds is the real decode limiter.
func BenchmarkMatmulQKAcc64(b *testing.B) {
	const hd = 128
	for _, depth := range []int{130, 2048, 8192} {
		b.Run(depthName(depth), func(b *testing.B) {
			const M = 1
			q := randF(M * hd)
			keys := randF(depth * hd)
			dst := make([]float32, M*depth)
			b.SetBytes(int64(depth * hd * 4))
			b.ResetTimer()
			for b.Loop() {
				MatmulQKAcc64(q, keys, dst, M, hd, depth, 0, hd)
			}
		})
	}
}
