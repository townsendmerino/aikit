//go:build arm64

package linalg

import (
	"testing"
	"time"
)

func TestFMAPeak_empirical(t *testing.T) {
	const (
		iters       = int64(400_000_000)
		fmaPerIter  = 20
		flopsPerFMA = 8 // 4 f32 lanes × (mul+add)
	)
	fmaPeakARM64(5_000_000) // warm to max clock
	t0 := time.Now()
	fmaPeakARM64(iters)
	el := time.Since(t0)
	g := float64(iters) * fmaPerIter * flopsPerFMA / el.Seconds() / 1e9
	t.Logf("MEASURED single-core f32 NEON FMA ceiling: %.1f GFLOPS (%d iters × %d FMLA in %v)", g, iters, fmaPerIter, el)
	t.Logf("→ implies %.1f f32-FMA/cycle at 3.2 GHz (brief assumed 8; 16 = 4 pipes × 4 lanes)", g/2/3.2)
	t.Logf("→ GEMM at ~42 GFLOPS (M8 tile) = %.0f%% of this measured ceiling", 100*42.0/g)
}
