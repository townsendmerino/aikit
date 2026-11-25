//go:build amd64

package linalg

import (
	"time"

	"github.com/townsendmerino/aikit/internal/cpufeat"
)

// fmaPeakAMD64 runs a register-saturating f32 AVX2 FMA loop (no memory traffic) to
// measure the core's achievable FMA throughput — the amd64 twin of fmaPeakARM64, and
// the denominator BenchmarkGEMMPeakFraction needs so that "fraction of peak" is
// grounded in a measured ceiling rather than a spec sheet.
//
// Each iteration retires fmaPeakFlopsPerIter flops. Requires AVX2+FMA; callers must
// check cpufeat.HasAVX2 first (TestFMAPeakAMD64_empirical skips without it).
//
//go:noescape
func fmaPeakAMD64(iters int64)

// fmaPeakFlopsPerIter is the flop count of one fmaPeakAMD64 loop iteration:
// 14 independent FMAs × 8 f32 lanes × 2 flops (a multiply and an add).
const fmaPeakFlopsPerIter = 14 * 8 * 2

// measureFMAPeak times the AVX2 probe and converts to GFLOP/s.
//
// IT TAKES THE MAX OF SEVERAL SHORT RUNS, not a single long one, because a throughput
// CEILING is a maximum: CPU contention, a cold boost clock or a scheduler migration can
// only ever make an observed rate LOWER than the true ceiling, never higher. Averaging
// would fold those in; the max rejects them.
//
// This is not hypothetical. A single-run version of this measured 136 GFLOPS when run
// alone and 63 inside `go test ./...`, where Go runs packages concurrently and the core
// is neither idle nor at boost — a 2.2× swing in a number whose whole purpose is to be
// a stable denominator.
func measureFMAPeak() (float64, bool) {
	if !cpufeat.HasAVX2 {
		return 0, false
	}
	const (
		warm  = int64(2_000_000)
		iters = int64(10_000_000)
		runs  = 5
	)
	fmaPeakAMD64(warm)
	best := 0.0
	for range runs {
		t0 := time.Now()
		fmaPeakAMD64(iters)
		el := time.Since(t0)
		if el <= 0 {
			continue
		}
		if g := float64(iters) * fmaPeakFlopsPerIter / el.Seconds() / 1e9; g > best {
			best = g
		}
	}
	return best, best > 0
}
