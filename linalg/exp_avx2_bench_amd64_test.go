//go:build amd64

package linalg

import (
	"math"
	"math/rand/v2"
	"testing"
)

// BenchmarkExpF32KernelsAMD64 is the amd64 half of the S-06 step-2 speed row.
// The f64_mathExp arm is what goinfer calls today, so it is the comparison that
// decides what the swap is worth on this architecture.
func BenchmarkExpF32KernelsAMD64(b *testing.B) {
	if !hasAVX2 {
		b.Skip("no AVX2+FMA3")
	}
	const n = 8960
	rng := rand.New(rand.NewPCG(4, 4))
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(rng.Float64()*20 - 10)
	}
	dst := make([]float32, n)
	b.Run("avx2", func(b *testing.B) {
		b.SetBytes(int64(n) * 4)
		for b.Loop() {
			expF32ContractAVX2(&dst[0], &src[0], n)
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

// BenchmarkSiLUAMD64 measures the dispatched SiLU, which routes through the AVX2
// exp with pinned Go arithmetic around it.
func BenchmarkSiLUAMD64(b *testing.B) {
	const n = 8960
	rng := rand.New(rand.NewPCG(6, 6))
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(rng.NormFloat64() * 4)
	}
	dst := make([]float32, n)
	b.Run("dispatched", func(b *testing.B) {
		b.SetBytes(int64(n) * 4)
		for b.Loop() {
			SiLUContractInto(dst, src)
		}
	})
	b.Run("goinfer_f64", func(b *testing.B) {
		b.SetBytes(int64(n) * 4)
		for b.Loop() {
			for i, v := range src {
				x64 := float64(v)
				dst[i] = float32(x64 / (1 + math.Exp(-x64)))
			}
		}
	})
}
