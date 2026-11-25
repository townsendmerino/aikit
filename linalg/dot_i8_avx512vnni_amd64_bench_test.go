//go:build amd64

package linalg

import (
	"testing"

	"github.com/townsendmerino/aikit/internal/cpufeat"
)

// BenchmarkDotI8_AVX512VNNIvsAVX2 compares the two SIMD tiers directly (not
// through dotI8's dispatch, which would just pick the faster one) at a
// transformer-typical K, both operands L1-resident — same K and reporting
// convention as BenchmarkDotI8VsF32_K768, so the numbers are directly
// comparable. Skips the VNNI side (rather than running it and SIGILLing) when
// the host lacks it.
func BenchmarkDotI8_AVX512VNNIvsAVX2(b *testing.B) {
	const K = 768 // 12×64: exact for VNNI, exact for AVX2's 16-multiple too

	ai := make([]int8, K)
	bi := make([]int8, K)
	for i := range ai {
		ai[i], bi[i] = int8(i%127-63), int8((i*7)%127-63)
	}

	b.Run("AVX2/dotI8AVX2", func(b *testing.B) {
		if !cpufeat.HasAVX2 {
			b.Skip("no AVX2")
		}
		var acc int32
		b.ResetTimer()
		for range b.N {
			acc += dotI8AVX2(&ai[0], &bi[0], K)
		}
		_ = acc
		b.ReportMetric(float64(K)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
	})

	b.Run("AVX512VNNI/dotI8AVX512VNNI", func(b *testing.B) {
		if !cpufeat.HasAVX512VNNI {
			b.Skip("no AVX-512 VNNI")
		}
		var acc int32
		b.ResetTimer()
		for range b.N {
			acc += dotI8AVX512VNNI(&ai[0], &bi[0], K)
		}
		_ = acc
		b.ReportMetric(float64(K)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
	})
}

// BenchmarkDotI8_AVX512VNNIvsAVX2_K3072 is the same comparison at a larger K
// (a big encoder fc2-shaped width), where the wider 128-byte VNNI main loop
// has more room to amortize its prologue against the 64-byte AVX2 body.
func BenchmarkDotI8_AVX512VNNIvsAVX2_K3072(b *testing.B) {
	const K = 3072

	ai := make([]int8, K)
	bi := make([]int8, K)
	for i := range ai {
		ai[i], bi[i] = int8(i%127-63), int8((i*7)%127-63)
	}

	b.Run("AVX2/dotI8AVX2", func(b *testing.B) {
		if !cpufeat.HasAVX2 {
			b.Skip("no AVX2")
		}
		var acc int32
		b.ResetTimer()
		for range b.N {
			acc += dotI8AVX2(&ai[0], &bi[0], K)
		}
		_ = acc
		b.ReportMetric(float64(K)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
	})

	b.Run("AVX512VNNI/dotI8AVX512VNNI", func(b *testing.B) {
		if !cpufeat.HasAVX512VNNI {
			b.Skip("no AVX-512 VNNI")
		}
		var acc int32
		b.ResetTimer()
		for range b.N {
			acc += dotI8AVX512VNNI(&ai[0], &bi[0], K)
		}
		_ = acc
		b.ReportMetric(float64(K)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
	})
}
