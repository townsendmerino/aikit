//go:build amd64

package linalg

import (
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/internal/cpufeat"
)

// BenchmarkW4A8_AVX512VNNIvsAVX2 compares the two SIMD tiers directly (not
// through dotW4A8's dispatch, which would just pick the faster one) at a
// transformer-typical K, packed/scales/act all L1-resident — same K and
// GMAC/s reporting convention as BenchmarkDotI8_AVX512VNNIvsAVX2, so the
// numbers are directly comparable across both kernel families. Skips the
// VNNI side (rather than running it and SIGILLing) when the host lacks
// AVX-512 VNNI+VL.
func BenchmarkW4A8_AVX512VNNIvsAVX2(b *testing.B) {
	const (
		K     = 768 // 24 groups of 32 — exact for both kernels' group=32 contract
		group = 32
	)
	nGroups := K / group

	rng := rand.New(rand.NewSource(1))
	act := make([]int8, K)
	for i := range act {
		act[i] = int8(rng.Intn(256) - 128)
	}
	packed := make([]byte, K/2)
	for i := range packed {
		packed[i] = byte(rng.Intn(256))
	}
	scales := make([]float32, nGroups)
	for i := range scales {
		scales[i] = float32(rng.Float64()*0.05 + 0.0001)
	}

	b.Run("AVX2/dotW4A8FoldAVX2", func(b *testing.B) {
		if !cpufeat.HasAVX2 {
			b.Skip("no AVX2")
		}
		var acc float32
		b.ResetTimer()
		for range b.N {
			acc += dotW4A8FoldAVX2(&act[0], &packed[0], &scales[0], nGroups)
		}
		_ = acc
		b.ReportMetric(float64(K)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
	})

	b.Run("AVX512VNNI/dotW4A8FoldAVX512VNNI", func(b *testing.B) {
		if !cpufeat.HasAVX512VNNIVL {
			b.Skip("no AVX-512 VNNI+VL")
		}
		var acc float32
		b.ResetTimer()
		for range b.N {
			acc += dotW4A8FoldAVX512VNNI(&act[0], &packed[0], &scales[0], nGroups)
		}
		_ = acc
		b.ReportMetric(float64(K)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
	})
}

// BenchmarkW4A8_AVX512VNNIvsAVX2_K3072 is the same comparison at a larger K
// (a big encoder fc2-shaped width, matching BenchmarkDotI8_AVX512VNNIvsAVX2_K3072),
// where the per-group loop overhead amortizes further.
func BenchmarkW4A8_AVX512VNNIvsAVX2_K3072(b *testing.B) {
	const (
		K     = 3072
		group = 32
	)
	nGroups := K / group

	rng := rand.New(rand.NewSource(2))
	act := make([]int8, K)
	for i := range act {
		act[i] = int8(rng.Intn(256) - 128)
	}
	packed := make([]byte, K/2)
	for i := range packed {
		packed[i] = byte(rng.Intn(256))
	}
	scales := make([]float32, nGroups)
	for i := range scales {
		scales[i] = float32(rng.Float64()*0.05 + 0.0001)
	}

	b.Run("AVX2/dotW4A8FoldAVX2", func(b *testing.B) {
		if !cpufeat.HasAVX2 {
			b.Skip("no AVX2")
		}
		var acc float32
		b.ResetTimer()
		for range b.N {
			acc += dotW4A8FoldAVX2(&act[0], &packed[0], &scales[0], nGroups)
		}
		_ = acc
		b.ReportMetric(float64(K)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
	})

	b.Run("AVX512VNNI/dotW4A8FoldAVX512VNNI", func(b *testing.B) {
		if !cpufeat.HasAVX512VNNIVL {
			b.Skip("no AVX-512 VNNI+VL")
		}
		var acc float32
		b.ResetTimer()
		for range b.N {
			acc += dotW4A8FoldAVX512VNNI(&act[0], &packed[0], &scales[0], nGroups)
		}
		_ = acc
		b.ReportMetric(float64(K)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
	})
}
