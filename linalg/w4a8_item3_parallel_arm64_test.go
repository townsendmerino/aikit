//go:build arm64

package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// TestW4A8Item3ParallelAggregate is the item-3 harness's remaining GO
// criterion (docs/prompts/w4a8-item3-harness.md): the 6-worker aggregate at
// the real shape, for the combined dotW4A8SplitHalf2Acc kernel, against the
// SAME production fork-join primitive TestW4A8_parallelScaling measures
// (Workspace.parallel — the identical mechanism MatmulBTW4A8Into's parallel
// branch uses), not a from-scratch goroutine loop. No dispatch wiring: this
// calls the harness kernel directly inside the span closure, mirroring
// w4a8Span's per-row shape but repacked-layout, mirroring only the
// parallelization mechanism, not touching the real dispatch or WeightMat.
//
// Reordered ahead of the row-interleave/uncentered grid cells per feedback:
// this is the one measurement that could reinterpret everything else
// (latency-hiding raises each worker's own memory pressure, so parallel
// efficiency could move either direction from the single-call win) — no
// point polishing single-call variants that don't survive fan-out.
func TestW4A8Item3ParallelAggregate(t *testing.T) {
	harnessOnly(t)
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	const (
		K       = 1536
		group   = 32
		N       = 8960
		M       = 1
		NLAYERS = 8 // matches TestW4A8_parallelScaling: past shared-cache residency
	)
	nGroups := K / group
	bpr := (K + 1) / 2

	rng := rand.New(rand.NewSource(31))
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	aq := make([]int8, K)
	aScale := quantizeRowInt8(a, aq)

	type layer struct {
		orig      []byte
		splitHalf []byte
		packed4   []byte
		scales4   []float32
		scales    []float32
	}
	layers := make([]layer, NLAYERS)
	for l := range layers {
		w := make([]float32, N*K)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		packed, scales := QuantizeGroupsInt4(w, N, K, group)
		sh := make([]byte, len(packed))
		for r := range N {
			row := packed[r*bpr : r*bpr+bpr]
			copy(sh[r*bpr:r*bpr+bpr], repackSplitHalfRow(row, K))
		}
		p4 := make([]byte, N*bpr)
		s4 := make([]float32, N*nGroups)
		for q := range N / 4 {
			r0, r1, r2, r3 := q*4, q*4+1, q*4+2, q*4+3
			blk := repackSplitHalf4RowBlock(
				packed[r0*bpr:r0*bpr+bpr], packed[r1*bpr:r1*bpr+bpr],
				packed[r2*bpr:r2*bpr+bpr], packed[r3*bpr:r3*bpr+bpr], K)
			copy(p4[q*4*bpr:(q*4+4)*bpr], blk)
			sblk := interleaveScales4Row(
				scales[r0*nGroups:r0*nGroups+nGroups], scales[r1*nGroups:r1*nGroups+nGroups],
				scales[r2*nGroups:r2*nGroups+nGroups], scales[r3*nGroups:r3*nGroups+nGroups], nGroups)
			copy(s4[q*4*nGroups:(q*4+4)*nGroups], sblk)
		}
		layers[l] = layer{orig: packed, splitHalf: sh, packed4: p4, scales4: s4, scales: scales}
	}
	dst := make([]float32, N)

	spanOrig := func(w4 []byte, wScales []float32, j0, j1 int) {
		for j := j0; j < j1; j++ {
			prow := w4[j*bpr : j*bpr+bpr]
			srow := wScales[j*nGroups : j*nGroups+nGroups]
			dst[j] = dotW4A8FoldSDOT(&aq[0], &prow[0], &srow[0], nGroups) * aScale
		}
	}
	spanCombo := func(w4 []byte, wScales []float32, j0, j1 int) {
		for j := j0; j < j1; j++ {
			prow := w4[j*bpr : j*bpr+bpr]
			srow := wScales[j*nGroups : j*nGroups+nGroups]
			dst[j] = dotW4A8SplitHalf2Acc(&aq[0], &prow[0], &srow[0], nGroups) * aScale
		}
	}
	// Row4's span partitions QUADS, not individual rows: j0/j1 here are quad
	// indices (ws.parallel(N/4, ...) below), each covering 4 real output rows.
	spanRow4 := func(p4 []byte, s4 []float32, q0, q1 int) {
		var out [4]float32
		for q := q0; q < q1; q++ {
			blk := p4[q*4*bpr : q*4*bpr+4*bpr]
			sblk := s4[q*4*nGroups : q*4*nGroups+4*nGroups]
			dotW4A8SplitHalf4Row(&aq[0], &blk[0], &sblk[0], &out[0], nGroups)
			dst[q*4] = out[0] * aScale
			dst[q*4+1] = out[1] * aScale
			dst[q*4+2] = out[2] * aScale
			dst[q*4+3] = out[3] * aScale
		}
	}

	runOrig := func(workers int) float64 {
		var ws Workspace
		ws.SetWorkers(workers)
		ws.SetThreshold(0)
		best := math.Inf(1)
		for range 3 {
			i := 0
			r := testing.Benchmark(func(b *testing.B) {
				for b.Loop() {
					l := layers[i]
					ws.parallel(N, func(j0, j1 int) { spanOrig(l.orig, l.scales, j0, j1) })
					i++
					if i == NLAYERS {
						i = 0
					}
				}
			})
			best = min(best, float64(r.NsPerOp()))
		}
		return best
	}
	runCombo := func(workers int) float64 {
		var ws Workspace
		ws.SetWorkers(workers)
		ws.SetThreshold(0)
		best := math.Inf(1)
		for range 3 {
			i := 0
			r := testing.Benchmark(func(b *testing.B) {
				for b.Loop() {
					l := layers[i]
					ws.parallel(N, func(j0, j1 int) { spanCombo(l.splitHalf, l.scales, j0, j1) })
					i++
					if i == NLAYERS {
						i = 0
					}
				}
			})
			best = min(best, float64(r.NsPerOp()))
		}
		return best
	}
	runRow4 := func(workers int) float64 {
		var ws Workspace
		ws.SetWorkers(workers)
		ws.SetThreshold(0)
		best := math.Inf(1)
		for range 3 {
			i := 0
			r := testing.Benchmark(func(b *testing.B) {
				for b.Loop() {
					l := layers[i]
					ws.parallel(N/4, func(q0, q1 int) { spanRow4(l.packed4, l.scales4, q0, q1) })
					i++
					if i == NLAYERS {
						i = 0
					}
				}
			})
			best = min(best, float64(r.NsPerOp()))
		}
		return best
	}

	bytesPerRow := float64(bpr + nGroups*4)
	totalBytes := float64(N) * bytesPerRow

	var base6Orig, base6Combo, base6Row4 float64
	for _, workers := range []int{1, 2, 4, 6, 8} {
		nsOrig := runOrig(workers)
		nsCombo := runCombo(workers)
		nsRow4 := runRow4(workers)
		gbsOrig := totalBytes / nsOrig
		gbsCombo := totalBytes / nsCombo
		gbsRow4 := totalBytes / nsRow4
		if workers == 6 {
			base6Orig, base6Combo, base6Row4 = gbsOrig, gbsCombo, gbsRow4
		}
		t.Logf("workers=%d: orig %.0f ns %.2f GB/s | combo %.0f ns %.2f GB/s | row4 %.0f ns %.2f GB/s | row4/orig %.3fx | row4/combo %.3fx",
			workers, nsOrig, gbsOrig, nsCombo, gbsCombo, nsRow4, gbsRow4, gbsRow4/gbsOrig, gbsRow4/gbsCombo)
	}
	t.Logf("FINAL GO-bar check: 6-worker row4 %.2f GB/s vs original %.2f GB/s = %.3fx | vs combo(2acc,no row4) %.2f GB/s = %.3fx (bar: >=1.4x over the original baseline)",
		base6Row4, base6Orig, base6Row4/base6Orig, base6Combo, base6Row4/base6Combo)
}
