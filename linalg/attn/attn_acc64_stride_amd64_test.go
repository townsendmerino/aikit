//go:build amd64

package attn

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/townsendmerino/aikit/internal/cpufeat"

	"github.com/townsendmerino/aikit/linalg"
)

// BenchmarkQkAcc64Stride times the 4-key and 16-key acc64 QK kernels on one thread over one head of a cache whose key rows are
// rowStride floats apart, at Phi-3-mini's shape (head dim 96, 3,900 keys) with its kvDim stride (3,072 floats, 12 KiB: a multiple of
// 4 KiB, so rows alias in the L1 sets) and that stride padded by 16 floats (no aliasing), to separate an L1 set-conflict cost from the
// kernels' own arithmetic.
func BenchmarkQkAcc64Stride(b *testing.B) {
	if !cpufeat.HasAVX2 {
		b.Skip("no AVX2")
	}
	const K, N = 96, 3904 // a multiple of 16, so both kernels cover every key
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

// BenchmarkQkAcc64StrideF16 is BenchmarkQkAcc64Stride for the f16 QK kernels: the 16-key and 4-key f16 kernels, equal work (3,904 keys,
// a multiple of 16), one thread, at Phi-3-mini's f16 key stride (3,072 halves, 6 KiB: rows alternate between two L1 sets, eight of
// the sixteen live rows in each) and padded by 32 halves. The sub-benchmarks run k16, k4, k4, k16 so -count repeats an A B B A block.
func BenchmarkQkAcc64StrideF16(b *testing.B) {
	if !cpufeat.HasAVX2 || !cpufeat.HasF16C {
		b.Skip("no AVX2+F16C")
	}
	const K, N = 96, 3904
	for _, stride := range []int{3072, 3072 + 32} {
		rng := rand.New(rand.NewPCG(3, 5))
		f := make([]float32, (N-1)*stride+K)
		for i := range f {
			f[i] = float32(rng.NormFloat64())
		}
		rows := make([]uint16, len(f))
		linalg.F32ToF16Slice(rows, f)
		q := make([]float32, K)
		for i := range q {
			q[i] = float32(rng.NormFloat64())
		}
		dst := make([]float32, N)
		k16 := func(b *testing.B) {
			for range b.N {
				qkAcc64AVX16F16(&q[0], &rows[0], stride*2, K/4, N/16, &dst[0])
			}
		}
		k4 := func(b *testing.B) {
			for range b.N {
				qkAcc64AVX4F16(&q[0], &rows[0], stride*2, K/4, N/4, &dst[0])
			}
		}
		b.Run(fmt.Sprintf("stride%d/1-k16", stride), k16)
		b.Run(fmt.Sprintf("stride%d/2-k4", stride), k4)
		b.Run(fmt.Sprintf("stride%d/3-k4", stride), k4)
		b.Run(fmt.Sprintf("stride%d/4-k16", stride), k16)
	}
}
