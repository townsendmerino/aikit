//go:build arm64

package linalg

import (
	"math"
	"math/rand/v2"
	"testing"
)

// BenchmarkExpF32Kernels is the S-06 step-2 speed row: the NEON kernel against
// the contract oracle it is bit-identical to, and against the f64 math.Exp that
// goinfer's SiLU and softmax actually call today — which is the comparison that
// says what the swap is worth.
//
// 8960 is the 1.5B's intermediate width, i.e. one SiLU row.
func BenchmarkExpF32Kernels(b *testing.B) {
	const n = 8960
	rng := rand.New(rand.NewPCG(4, 4))
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(rng.Float64()*20 - 10)
	}
	dst := make([]float32, n)

	b.Run("neon", func(b *testing.B) {
		b.SetBytes(int64(n) * 4)
		for b.Loop() {
			expF32ContractNEON(&dst[0], &src[0], n)
		}
	})
	b.Run("contract_scalar", func(b *testing.B) {
		b.SetBytes(int64(n) * 4)
		for b.Loop() {
			for i, v := range src {
				dst[i] = expF32Contract(v)
			}
		}
	})
	b.Run("shipped_expF32Core", func(b *testing.B) {
		b.SetBytes(int64(n) * 4)
		for b.Loop() {
			for i, v := range src {
				dst[i] = expF32Core(v)
			}
		}
	})
	b.Run("f64_mathExp", func(b *testing.B) {
		b.SetBytes(int64(n) * 4)
		for b.Loop() {
			for i, v := range src {
				dst[i] = float32(math.Exp(float64(v)))
			}
		}
	})
}

// BenchmarkSoftmaxRowKernels is the softmax half of the S-06 step-2 speed row.
// nKeys=2048 is a mid-depth attention row; the softmax is called once per
// (head, token) so it is O(L²) work over a decode.
func BenchmarkSoftmaxRowKernels(b *testing.B) {
	const n = 2048
	rng := rand.New(rand.NewPCG(9, 9))
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(rng.NormFloat64() * 8)
	}
	dst := make([]float32, n)

	b.Run("neon_contract", func(b *testing.B) {
		b.SetBytes(int64(n) * 4)
		for b.Loop() {
			softmaxRowContractNEON(dst, src)
		}
	})
	b.Run("scalar_contract", func(b *testing.B) {
		b.SetBytes(int64(n) * 4)
		for b.Loop() {
			softmaxRowContract(dst, src)
		}
	})
	b.Run("shipped_SoftmaxRowInto", func(b *testing.B) {
		b.SetBytes(int64(n) * 4)
		for b.Loop() {
			SoftmaxRowInto(dst, src)
		}
	})
}
