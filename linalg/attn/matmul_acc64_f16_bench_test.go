package attn

import (
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// BenchmarkMatmulQKAcc64F16 is BenchmarkMatmulQKAcc64 over a binary16 key cache: the same head dim,
// M and depths, so the f32 and f16 rows read side by side. SetBytes is the f16 cache's bytes.
func BenchmarkMatmulQKAcc64F16(b *testing.B) {
	const hd = 128
	for _, depth := range []int{130, 2048, 8192} {
		b.Run(depthName(depth), func(b *testing.B) {
			const M = 1
			q := randF(M * hd)
			keys := make([]uint16, depth*hd)
			linalg.F32ToF16Slice(keys, randF(depth*hd))
			dst := make([]float32, M*depth)
			b.SetBytes(int64(depth * hd * 2))
			b.ResetTimer()
			for b.Loop() {
				MatmulQKAcc64F16(q, keys, dst, M, hd, depth, 0, hd)
			}
		})
	}
}

// BenchmarkMatmulAVAcc64F16 is BenchmarkMatmulAVAcc64 over a binary16 value cache, at the same
// shapes. SetBytes is the f16 cache's bytes.
func BenchmarkMatmulAVAcc64F16(b *testing.B) {
	const hd = 128
	for _, depth := range []int{130, 2048, 8192} {
		b.Run(depthName(depth), func(b *testing.B) {
			const M = 1
			scores := randF(M * depth)
			vals := make([]uint16, depth*hd)
			linalg.F32ToF16Slice(vals, randF(depth*hd))
			dst := make([]float32, M*hd)
			acc := make([]float64, hd)
			b.SetBytes(int64(depth * hd * 2))
			b.ResetTimer()
			for b.Loop() {
				MatmulAVAcc64F16(scores, vals, dst, acc, M, depth, hd, 0, hd)
			}
		})
	}
}
