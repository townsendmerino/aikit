//go:build arm64

package linalg

import (
	"math/rand"
	"testing"
)

// TestW4A8Row4ColdFix_warmIntact is the cold-fix harness pass's warm-side
// check (docs/task-w4a8-neon-bandwidth.md): the grid's own acceptance rule
// is "recovers >=half the cold penalty with warm intact" — a cold win that
// costs the warm 1.6-1.75x is a loss. This measures hot/L1-resident speed
// only (repeated calls against the SAME already-resident quad, matching
// this file family's own "hot" convention in w4a8_item3_bench_arm64_test.go)
// for every variant against the production row4 baseline and against
// canonical×4, at the same real FFN shape (K=1536, N=8960) the rest of this
// probe family uses. The real cold number (through the actual pager, on
// genuinely never-touched real-model experts) is measured separately in
// goinfer, matching "the same cold-touch benchmark that produced the 69%
// number" per the harness brief — this test's job is warm-side acceptance
// only.
func TestW4A8Row4ColdFix_warmIntact(t *testing.T) {
	harnessOnly(t)
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	const (
		K     = 1536
		group = 32
		N     = 8960
	)
	nGroups := K / group
	bpr := (K + 1) / 2

	rng := rand.New(rand.NewSource(101))
	act := make([]int8, K)
	for i := range act {
		act[i] = int8(rng.Intn(255) - 128)
	}
	w := make([]float32, N*K)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	packed, scales := QuantizeGroupsInt4(w, N, K, group)

	row0 := packed[0:bpr]
	s0 := scales[0:nGroups]

	packed4 := RepackW4A8Row4(packed, N, K, group)
	scales4 := RepackW4A8Row4Scales(scales, N, K, group)
	blk0 := packed4[0 : 4*bpr]
	sblk0 := scales4[0 : 4*nGroups]

	w0, w1, w2, w3 := RepackW4A8Row4Deshared(packed[0:4*bpr], 4, K, group)
	ds0, ds1, ds2, ds3 := RepackW4A8Row4DesharedScales(scales[0:4*nGroups], 4, K, group)

	var dst [4]float32

	runCanonicalX4 := func() float64 {
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				for range 4 {
					sinkW4A8F32ARM64 = dotW4A8FoldSDOT(&act[0], &row0[0], &s0[0], nGroups)
				}
			}
		})
		return float64(r.NsPerOp())
	}
	runRow4 := func() float64 {
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				dotW4A8SplitHalf4Row(&act[0], &blk0[0], &sblk0[0], &dst[0], nGroups)
			}
		})
		sinkW4A8F32ARM64 = dst[0]
		return float64(r.NsPerOp())
	}
	runPrefetch := func(dist int) func() float64 {
		return func() float64 {
			r := testing.Benchmark(func(b *testing.B) {
				for b.Loop() {
					dotW4A8SplitHalf4RowPrefetch(&act[0], &blk0[0], &sblk0[0], &dst[0], nGroups, dist)
				}
			})
			sinkW4A8F32ARM64 = dst[0]
			return float64(r.NsPerOp())
		}
	}
	runDeshared := func() float64 {
		r := testing.Benchmark(func(b *testing.B) {
			for b.Loop() {
				dotW4A8SplitHalf4RowDeshared(&act[0], &w0[0], &w1[0], &w2[0], &w3[0], &ds0[0], &ds1[0], &ds2[0], &ds3[0], &dst[0], nGroups)
			}
		})
		sinkW4A8F32ARM64 = dst[0]
		return float64(r.NsPerOp())
	}

	distances := []int{64, 128, 256, 512, 4096}
	variants := []struct {
		name string
		fn   func() float64
	}{
		{"row4 (baseline)", runRow4},
		{"prefetch dist=64B (1 line)", runPrefetch(distances[0])},
		{"prefetch dist=128B (2 lines)", runPrefetch(distances[1])},
		{"prefetch dist=256B (4 lines)", runPrefetch(distances[2])},
		{"prefetch dist=512B (8 lines)", runPrefetch(distances[3])},
		{"prefetch dist=4096B (page-crossing)", runPrefetch(distances[4])},
		{"deshared", runDeshared},
	}

	canonNs := minOf3(runCanonicalX4)
	row4Ns := minOf3(runRow4)
	t.Logf("canonical x4 (hot, L1-resident): %.2f ns", canonNs)
	t.Logf("row4 baseline (hot, L1-resident): %.2f ns (%.3fx vs canonical x4)", row4Ns, canonNs/row4Ns)
	for _, v := range variants {
		ns := minOf3(v.fn)
		t.Logf("%-38s hot: %.2f ns  vs canonical x4: %.3fx  vs row4 baseline: %.3fx (>=1.0 = warm intact)", v.name, ns, canonNs/ns, row4Ns/ns)
	}
}

func minOf3(f func() float64) float64 {
	best := f()
	for range 2 {
		if v := f(); v < best {
			best = v
		}
	}
	return best
}
