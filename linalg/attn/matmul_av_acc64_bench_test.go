package attn

import "testing"

// BenchmarkMatmulAVAcc64 exists because task-simd-audit.md S-04 quotes AV timings
// (8,929 ns at depth 130, 565,038 at 8192, ≈1.86 GMAC/s) sourced from goinfer's
// campaign records rather than from a bench in this repo — so the kernel whose
// cost the audit calls "~75% of the token at depth 8k" had no local row to
// re-measure after a change.
//
// Shapes are the audit's: the reference model's hd=128 at the three depths its
// decision rule names. M=1 is the decode case; the caller runs one of these per
// (head, token).
//
// Read it as a ROW, not a verdict: this is a single-core kernel bench, so it says
// nothing about the 6-way head fan-out that S-02 finds is the real decode limiter.
func BenchmarkMatmulAVAcc64(b *testing.B) {
	const hd = 128
	for _, depth := range []int{130, 2048, 8192} {
		b.Run(depthName(depth), func(b *testing.B) {
			const M = 1
			scores := randF(M * depth)
			vals := randF(depth * hd)
			dst := make([]float32, M*hd)
			acc := make([]float64, hd)
			b.SetBytes(int64(depth * hd * 4))
			b.ResetTimer()
			for b.Loop() {
				MatmulAVAcc64(scores, vals, dst, acc, M, depth, hd, 0, hd)
			}
		})
	}
}

func depthName(d int) string {
	switch d {
	case 130:
		return "depth130"
	case 2048:
		return "depth2048"
	default:
		return "depth8192"
	}
}
