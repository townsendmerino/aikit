package linalg

import (
	"math/rand/v2"
	"testing"
)

// BenchmarkGEMMPeakFraction measures the blocked f32 GEMM (MatmulBTInto) as a fraction of
// a ceiling measured ON THE MACHINE RUNNING IT (MeasuredFMAPeakGFLOPS → the per-arch
// register-saturating FMA probe). GFLOPS is the hard number; the fraction follows the
// measured peak, never a spec sheet — the "8 FMA/cyc" back-of-envelope for Firestorm is
// ~2× low (4 pipes × 4 lanes = 16).
//
// THE DENOMINATOR USED TO BE A CONSTANT, AND THIS BENCHMARK HAS NO BUILD TAG. It divided
// by 3.2 GHz × 16 f32-FMA/cyc = 102.4 GFLOPS on every architecture, so on a Zen 2 box
// (measured ceiling ~135) it reported the GEMM at "~50 %peak" where the true figure is
// ~38%. Both numbers are believable; only one is real. Measuring the denominator removes
// the class.
//
// Reference points, each against its OWN measured ceiling:
//   - apple-m1pro (~95 GFLOPS): 1×8 Dot8x4 sat at ~40%; the 2×8 dual-row kernel lifted
//     it to ~68–73%.
//   - nvidia-rtx2070s (~135 GFLOPS): ~38%, i.e. the arm64 PRE-2×8 ratio — has2x8Kernel
//     is false off arm64 (kernel_other.go), so amd64 never packs and runs dot-per-output
//     with a 0.5 FMA-per-load inner loop. Two load ports feeding 2-flop FMAs cap that
//     shape at ~50% of FMA peak before any other effect; it reaches 77% of THAT cap.
func BenchmarkGEMMPeakFraction(b *testing.B) {
	peakReal, ok := measuredFMAPeakGFLOPS()
	if !ok {
		b.Logf("no FMA-peak probe for this architecture — reporting GFLOPS only, no fraction")
	} else {
		b.Logf("measured single-core f32 FMA ceiling on THIS machine: %.1f GFLOPS", peakReal)
	}

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
			// Only report a fraction when the denominator was measured here. Emitting
			// one from a guessed peak is what produced the "~50 %peak" fiction on amd64.
			if ok && peakReal > 0 {
				b.ReportMetric(100*g/peakReal, "%peak")
			}
		})
	}
}
