package linalg

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// BenchmarkQ4KvsSplitHalf: the M=1 Q4_K matmul against the split-half int4 W4A8 matmul (per-32
// activations, the path q4k is gated against) at the qwen2.5-7b FFN shape, single-threaded (serial)
// and at goinfer's decode fan-out. SetBytes is each kind's own weight bytes, so MB/s compares the
// kernels' effective bandwidth.
func BenchmarkQ4KvsSplitHalf(b *testing.B) {
	const K, N = 3584, 18944
	r := rand.New(rand.NewPCG(71, 72))
	a := agRandMat(r, K)
	raw := q4kTestRows(r, N, K, false)
	q4k, err := WrapQ4K(raw, N, K)
	if err != nil {
		b.Fatal(err)
	}
	q4, s4 := QuantizeGroupsInt4(agRandMat(r, N*K), N, K, 32)
	sh, ok := RepackInt4SplitHalfInPlace(q4, s4, N, K, 32)
	if !ok {
		b.Skip("split-half repack declined")
	}
	dst := make([]float32, N)
	for _, thr := range []int{math.MaxInt, 1 << 20} {
		mode := "serial"
		if thr != math.MaxInt {
			mode = "fanout"
		}
		b.Run(fmt.Sprintf("%s/q4k", mode), func(b *testing.B) {
			var ws Workspace
			ws.SetThreshold(thr)
			b.SetBytes(int64(len(raw)))
			for range b.N {
				q4k.MatmulBTInto(&ws, a, dst, 1)
			}
		})
		b.Run(fmt.Sprintf("%s/int4-splithalf-g32", mode), func(b *testing.B) {
			var ws Workspace
			ws.SetThreshold(thr)
			ws.SetActQuantGroup(32)
			b.SetBytes(int64(N*K/2 + 4*N*K/32))
			for range b.N {
				sh.MatmulBTW4A8Into(&ws, a, dst, 1)
			}
		})
	}
}

// BenchmarkQ4KDotRow: the bare dotQ4KAVX2 call on one row (K = 3584, 14 super-blocks), against the
// Go oracle on the same row: ns per super-block, isolated from the matmul around it.
func BenchmarkQ4KDotRow(b *testing.B) {
	const K = 3584
	r := rand.New(rand.NewPCG(73, 74))
	raw := q4kTestRows(r, 1, K, false)
	a := agRandMat(r, K)
	nG := K / q4kGroup
	aq, aS := make([]int8, K), make([]float32, nG)
	QuantizeActivationsGroupedInto(aq, aS, a, 1, K, q4kGroup)
	sums := make([]int32, nG)
	q4kActSums(aq, sums, 1, K)
	asumf := make([]float32, nG)
	for i := range asumf {
		asumf[i] = aS[i] * float32(sums[i])
	}
	var sink float32
	b.Run("avx2", func(b *testing.B) {
		for range b.N {
			sink += dotQ4KAVX2(&raw[0], &aq[0], &aS[0], &asumf[0], K/qkK)
		}
	})
	b.Run("go", func(b *testing.B) {
		for range b.N {
			sink += dotQ4KGo(raw, aq, aS, sums, K)
		}
	})
	_ = sink
}
