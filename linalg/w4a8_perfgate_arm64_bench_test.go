//go:build arm64

package linalg

import (
	"math/rand"
	"testing"
)

// BenchmarkW4A8_CanonicalVsSplitHalf is the arm64 twin of the amd64 benchmark of the same
// name — the one tools/perfgate's instrument set names as "W4A8 M=1 — the int4 decode
// kernel". Until 2026-09-22 that name existed only under //go:build amd64, so on an arm64
// box perfgate's 45-shape PASS carried no row at all for the production arm64 decode kernel
// (dotW4A8SplitHalf4Row → the S-05 fold), the exact kernel a release had just changed: a
// gate that vouches for less than it reads as (docs/task-simd-audit.md S-09). This file
// closes that: same benchmark name, so the regex picks it up unchanged; perfgate reports it
// "new — no baseline" on the release that adds it and judges it from the next tag on.
//
// Arms: hot (one L1-resident row / quad, the kernel's issue rate) and cold (a 12-matrix
// bank DRAM-streamed through the real dispatch, serial, the regime decode runs in —
// docs/internal/measuring-performance.md §1.3). Both layouts see identical logical weights.
func BenchmarkW4A8_CanonicalVsSplitHalf(b *testing.B) {
	if !hasDotProd {
		b.Skip("DotProd required")
	}
	const (
		group = 32
		K     = 1536
		N     = 8960
		bank  = 12
	)
	nGroups := K / group
	rng := rand.New(rand.NewSource(1))
	act := make([]int8, K)
	for i := range act {
		act[i] = int8(rng.Intn(256) - 128)
	}
	w := make([]float32, 4*K)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	packed, scales := QuantizeGroupsInt4(w, 4, K, group)
	row4 := RepackW4A8Row4(packed, 4, K, group)
	row4s := RepackW4A8Row4Scales(scales, 4, K, group)
	corr := make([]int32, 4*nGroups)
	w4a8LaneCorrNeg8(&act[0], &corr[0], nGroups)

	b.Run("canonical/hot", func(b *testing.B) {
		var acc float32
		b.ResetTimer()
		for range b.N {
			acc += dotW4A8FoldSDOT(&act[0], &packed[0], &scales[0], nGroups)
		}
		sinkW4A8F32ARM64 = acc
		b.ReportMetric(float64(K)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
	})
	b.Run("splithalf/hot", func(b *testing.B) {
		var dst [4]float32
		b.ResetTimer()
		for range b.N {
			dotW4A8SplitHalf4RowFold(&act[0], &corr[0], &row4[0], &row4s[0], &dst[0], nGroups)
		}
		sinkW4A8F32ARM64 = dst[0]
		b.ReportMetric(float64(4*K)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
	})

	// COLD: whole matmuls over a bank sized well past the LLC, through the real entry points.
	type mat struct {
		canon  []byte
		canonS []float32
		row4   []byte
		row4S  []float32
	}
	mats := make([]mat, bank)
	for i := range mats {
		wm := make([]float32, N*K)
		for j := range wm {
			wm[j] = float32(rng.NormFloat64())
		}
		p, s := QuantizeGroupsInt4(wm, N, K, group)
		mats[i] = mat{p, s, RepackW4A8Row4(p, N, K, group), RepackW4A8Row4Scales(s, N, K, group)}
	}
	a := make([]float32, K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	dst := make([]float32, N)
	var ws Workspace
	ws.SetThreshold(1 << 60) // serial: the single-core number, not the fan-out's
	b.Run("canonical/cold", func(b *testing.B) {
		b.ResetTimer()
		for i := range b.N {
			m := &mats[i%bank]
			MatmulBTW4A8Into(&ws, a, m.canon, m.canonS, dst, 1, K, N, group)
		}
		sinkW4A8F32ARM64 = dst[0]
		b.ReportMetric(float64(K)*float64(N)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
	})
	b.Run("splithalf/cold", func(b *testing.B) {
		b.ResetTimer()
		for i := range b.N {
			m := &mats[i%bank]
			MatmulBTW4A8Row4Into(&ws, a, m.row4, m.row4S, dst, 1, K, N, group)
		}
		sinkW4A8F32ARM64 = dst[0]
		b.ReportMetric(float64(K)*float64(N)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
	})
}
