//go:build amd64

package linalg

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

// BenchmarkQkAcc64Stride times the 4-key and 16-key acc64 QK kernels on one thread over one head of a cache whose key rows are
// rowStride floats apart, at Phi-3-mini's shape (head dim 96, 3,900 keys) with its kvDim stride (3,072 floats, 12 KiB: a multiple of
// 4 KiB, so rows alias in the L1 sets) and that stride padded by 16 floats (no aliasing), to separate an L1 set-conflict cost from the
// kernels' own arithmetic.
func BenchmarkQkAcc64Stride(b *testing.B) {
	if !hasAVX2 {
		b.Skip("no AVX2")
	}
	const K, N = 96, 3900
	for _, stride := range []int{3072, 3072 + 16, 128} {
		rng := rand.New(rand.NewPCG(3, 5))
		rows := make([]float32, (N-1)*stride+K)
		for i := range rows {
			rows[i] = float32(rng.NormFloat64())
		}
		q := make([]float32, K)
		for i := range q {
			q[i] = float32(rng.NormFloat64())
		}
		dst := make([]float32, N)
		b.Run(fmt.Sprintf("stride%d/k4", stride), func(b *testing.B) {
			for range b.N {
				qkAcc64AVX4(&q[0], &rows[0], stride*4, K/4, N/4, &dst[0])
			}
		})
		b.Run(fmt.Sprintf("stride%d/k16", stride), func(b *testing.B) {
			for range b.N {
				qkAcc64AVX16(&q[0], &rows[0], stride*4, K/4, N/16, &dst[0])
			}
		})
	}
}
