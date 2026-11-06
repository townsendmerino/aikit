//go:build amd64

package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// repackSplitHalfRowAMD64 mirrors arm64's repackSplitHalfRow (which lives behind //go:build
// arm64) so this harness does not have to move shipped code to run an experiment. Canonical
// layout: byte i holds weights 2i (low) and 2i+1 (high). Split-half: byte i holds weight i (low)
// and weight i+16 (high), so one 16-byte load yields two contiguous halves.
func repackSplitHalfRowAMD64(packed []byte, K int) []byte {
	if K%32 != 0 {
		panic("repackSplitHalfRowAMD64: K must be a multiple of 32")
	}
	nib := func(k int) byte {
		b := packed[k/2]
		if k%2 == 0 {
			return b & 0x0F
		}
		return b >> 4
	}
	out := make([]byte, len(packed))
	for g := 0; g < K/32; g++ {
		gk, ob := g*32, g*16
		for i := range 16 {
			out[ob+i] = nib(gk+i) | (nib(gk+i+16) << 4)
		}
	}
	return out
}

// TestDotW4A8SplitHalfAVX2_matchesOracle gates the split-half kernel against the SCALAR oracle
// reading the CANONICAL layout — i.e. the repack and the kernel must together reproduce the same
// dot the production path computes. Checking the kernel against a split-half-aware oracle would
// let a repack bug and a kernel bug cancel.
func TestDotW4A8SplitHalfAVX2_matchesOracle(t *testing.T) {
	if !hasAVX2 {
		t.Skip("AVX2 required")
	}
	const group = 32
	rng := rand.New(rand.NewSource(11))
	for _, nGroups := range []int{1, 2, 5, 32, 160} {
		K := nGroups * group
		act := make([]int8, K)
		for i := range act {
			act[i] = int8(rng.Intn(256) - 128)
		}
		canon := make([]byte, K/2)
		for i := range canon {
			canon[i] = byte(rng.Intn(256))
		}
		scales := make([]float32, nGroups)
		for i := range scales {
			scales[i] = float32(rng.Float64()*0.05 + 0.0001)
		}
		sh := repackSplitHalfRowAMD64(canon, K)

		want := dotW4A8Scalar(act, canon, scales, group, K) // canonical oracle
		got := dotW4A8SplitHalfAVX2(&act[0], &sh[0], &scales[0], nGroups)

		den := math.Abs(float64(want))
		if den < 1e-6 {
			den = 1e-6
		}
		if rel := math.Abs(float64(got-want)) / den; rel > 1e-4 {
			t.Errorf("nGroups=%d: splithalf %v vs canonical oracle %v (rel %.3g)", nGroups, got, want, rel)
		} else {
			t.Logf("nGroups=%-4d oracle %.6f  splithalf %.6f  (rel %.2g)", nGroups, want, got, rel)
		}
	}
}

// BenchmarkW4A8_CanonicalVsSplitHalf is the direct A/B. Both arms do identical arithmetic on
// identical weights; only the layout and the two VPUNPCK shuffles differ. The repack is done once
// outside the timed loop, which is what a load-time repack would do in production.
func BenchmarkW4A8_CanonicalVsSplitHalf(b *testing.B) {
	if !hasAVX2 {
		b.Skip("AVX2 required")
	}
	const (
		group = 32
		K     = 5120
	)
	nGroups := K / group
	rng := rand.New(rand.NewSource(1))
	act := make([]int8, K)
	for i := range act {
		act[i] = int8(rng.Intn(256) - 128)
	}
	canon := make([]byte, K/2)
	for i := range canon {
		canon[i] = byte(rng.Intn(256))
	}
	scales := make([]float32, nGroups)
	for i := range scales {
		scales[i] = float32(rng.Float64()*0.05 + 0.0001)
	}
	sh := repackSplitHalfRowAMD64(canon, K)

	b.Run("canonical/dotW4A8FoldAVX2", func(b *testing.B) {
		var acc float32
		b.ResetTimer()
		for range b.N {
			acc += dotW4A8FoldAVX2(&act[0], &canon[0], &scales[0], nGroups)
		}
		_ = acc
		b.ReportMetric(float64(K)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
	})
	b.Run("splithalf/dotW4A8SplitHalfAVX2", func(b *testing.B) {
		var acc float32
		b.ResetTimer()
		for range b.N {
			acc += dotW4A8SplitHalfAVX2(&act[0], &sh[0], &scales[0], nGroups)
		}
		_ = acc
		b.ReportMetric(float64(K)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
	})

	// COLD: N distinct weight rows streamed from DRAM, which is what decode actually does — one
	// row per output, never reused within a token. A hot-only win is not a decode win, and this
	// kernel's own ops/byte harness reports cold at 99.4% of hot for the canonical layout, so the
	// prediction is that the ratio survives. Predictions are not results.
	const N = 17408 // the FFN gate/up row count: ~45 MB of weights, well past L3
	bpr := K / 2
	canonAll := make([]byte, N*bpr)
	for i := range canonAll {
		canonAll[i] = byte(rng.Intn(256))
	}
	shAll := make([]byte, N*bpr)
	for r := range N {
		copy(shAll[r*bpr:(r+1)*bpr], repackSplitHalfRowAMD64(canonAll[r*bpr:(r+1)*bpr], K))
	}
	scAll := make([]float32, N*nGroups)
	for i := range scAll {
		scAll[i] = float32(rng.Float64()*0.05 + 0.0001)
	}

	b.Run("cold-canonical/dotW4A8FoldAVX2", func(b *testing.B) {
		var acc float32
		b.ResetTimer()
		for i := range b.N {
			r := i % N
			acc += dotW4A8FoldAVX2(&act[0], &canonAll[r*bpr], &scAll[r*nGroups], nGroups)
		}
		_ = acc
		b.ReportMetric(float64(K)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
	})
	b.Run("cold-splithalf/dotW4A8SplitHalfAVX2", func(b *testing.B) {
		var acc float32
		b.ResetTimer()
		for i := range b.N {
			r := i % N
			acc += dotW4A8SplitHalfAVX2(&act[0], &shAll[r*bpr], &scAll[r*nGroups], nGroups)
		}
		_ = acc
		b.ReportMetric(float64(K)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
	})
}
