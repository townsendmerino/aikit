package encoder

import (
	"math/rand/v2"
	"testing"
)

// BenchmarkGEMMPeakFraction is the Phase-1 gate for an MR×NR register-blocked f32
// microkernel: it measures the encoder's serial blocked GEMM (matmulBTBlockedInto,
// which routes through linalg.Dot8x4 — the 1×8 microkernel the review flags as
// load-bound) as a fraction of single-core f32 FMA peak, at in-cache M>1 shapes.
//
// Peak (Apple M1 Pro, Firestorm P-core): 4×128-bit NEON FMA pipes × 4 f32 lanes = 16
// f32-FMA/cycle × 2 flops × ~3.2 GHz ≈ 103 GFLOPS single-core. (The task brief assumed
// 8 FMA/cyc ⇒ 51.2 GFLOPS — that's half the pipe count; both fractions are printed so
// the decision is auditable either way.) GFLOPS is the hard number; fractions follow.
func BenchmarkGEMMPeakFraction(b *testing.B) {
	const clockGHz = 3.2
	peakReal := 2.0 * 16 * clockGHz // 16 f32-FMA/cyc (4 pipes × 4 lanes)
	peakBrief := 2.0 * 8 * clockGHz // the brief's conservative 8 FMA/cyc
	b.Logf("M1 Pro single-core f32 peak @ %.1f GHz: %.1f GFLOPS (16 FMA/cyc) / %.1f GFLOPS (brief's 8 FMA/cyc)",
		clockGHz, peakReal, peakBrief)

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
				matmulBTBlockedInto(a, w, dst, s.M, s.K, s.N) // serial, Dot8x4
			}
			flops := 2.0 * float64(s.M) * float64(s.N) * float64(s.K)
			secs := b.Elapsed().Seconds() / float64(b.N)
			g := flops / secs / 1e9
			b.ReportMetric(g, "GFLOPS")
			b.ReportMetric(100*g/peakReal, "%peak16")
			b.ReportMetric(100*g/peakBrief, "%peak8")
		})
	}
}
