package linalg

import (
	"math/rand/v2"
	"testing"
)

// These two benchmarks call only PORTABLE entry points, so they live in a
// portable file. They used to sit in the arm64-only bench file, which meant
// amd64 never ran them and the tanh/erf side of the contract had no measured
// number on that arch at all.

// BenchmarkGELUTanhKernels compares the dispatched GELU-tanh against the scalar
// contract and the shipped GELUTanhF32. n=8960 is one gemma-family gate row.
func BenchmarkGELUTanhKernels(b *testing.B) {
	const n = 8960
	rng := rand.New(rand.NewPCG(8, 8))
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(rng.NormFloat64() * 4)
	}
	dst := make([]float32, n)
	b.Run("dispatched", func(b *testing.B) {
		b.SetBytes(int64(n) * 4)
		for b.Loop() {
			GELUTanhContractInto(dst, src)
		}
	})
	b.Run("scalar_contract", func(b *testing.B) {
		b.SetBytes(int64(n) * 4)
		for b.Loop() {
			geluTanhScalarInto(dst, src)
		}
	})
	b.Run("shipped_GELUTanhF32", func(b *testing.B) {
		b.SetBytes(int64(n) * 4)
		for b.Loop() {
			for i, v := range src {
				dst[i] = GELUTanhF32(v)
			}
		}
	})
}

// BenchmarkGELUErfKernels compares the dispatched exact GELU against the scalar
// contract and the shipped GELUF32.
func BenchmarkGELUErfKernels(b *testing.B) {
	const n = 8960
	rng := rand.New(rand.NewPCG(12, 12))
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(rng.NormFloat64() * 3)
	}
	dst := make([]float32, n)
	b.Run("dispatched", func(b *testing.B) {
		b.SetBytes(int64(n) * 4)
		for b.Loop() {
			GELUContractInto(dst, src)
		}
	})
	b.Run("scalar_contract", func(b *testing.B) {
		b.SetBytes(int64(n) * 4)
		for b.Loop() {
			geluScalarInto(dst, src)
		}
	})
	b.Run("shipped_GELUF32", func(b *testing.B) {
		b.SetBytes(int64(n) * 4)
		for b.Loop() {
			for i, v := range src {
				dst[i] = GELUF32(v)
			}
		}
	})
}
