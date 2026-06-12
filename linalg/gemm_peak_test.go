package linalg

import (
	"math/rand/v2"
	"testing"
)

// BenchmarkGEMMPeakFraction measures the blocked f32 GEMM (MatmulBTInto) as a fraction of
// a MEASURED single-core ceiling (see fmaPeakARM64 / TestFMAPeak_empirical: 95.4 GFLOPS on
// this M1 Pro). GFLOPS is the hard number; the fraction follows the measured peak, not a
// spec sheet (the "8 FMA/cyc" back-of-envelope is ~2× low — Firestorm is 4 pipes × 4 lanes
// = 16). The 1×8 Dot8x4 kernel sat at ~40%; the 2×8 dual-row kernel lifts it to ~68–73%.
func BenchmarkGEMMPeakFraction(b *testing.B) {
	const clockGHz = 3.2
	peakReal := 2.0 * 16 * clockGHz // 16 f32-FMA/cyc (4 pipes × 4 lanes)
	b.Logf("M1 Pro single-core f32 peak @ %.1f GHz ≈ %.1f GFLOPS (measured ceiling ~95)", clockGHz, peakReal)

	shapes := []struct {
		name    string
		M, K, N int
	}{
		{"M8_K768_N768_tile", 8, 768, 768},
		{"M32_K768_N768_tile", 32, 768, 768},
		{"M128_K768_N768_tile", 128, 768, 768},
		{"M512_K384_N1536_minilm_fc1", 512, 384, 1536},
	}
	rng := rand.New(rand.NewPCG(1, 2))
	rv := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = rng.Float32()*2 - 1
		}
		return v
	}
	for _, s := range shapes {
		a, w := rv(s.M*s.K), rv(s.N*s.K)
		dst := make([]float32, s.M*s.N)
		b.Run(s.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				MatmulBTInto(dst, a, w, s.M, s.K, s.N) // serial blocked, Dot2x8
			}
			flops := 2.0 * float64(s.M) * float64(s.N) * float64(s.K)
			secs := b.Elapsed().Seconds() / float64(b.N)
			g := flops / secs / 1e9
			b.ReportMetric(g, "GFLOPS")
			b.ReportMetric(100*g/peakReal, "%peak")
		})
	}
}
