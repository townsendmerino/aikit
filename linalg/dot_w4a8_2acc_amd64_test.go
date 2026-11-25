//go:build amd64

package linalg

import (
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/internal/cpufeat"
)

// TestDotW4A8Fold2AccAVX2_matchesOracle gates the two-accumulator kernel on correctness before
// any timing is believed. It is checked against the SCALAR oracle rather than against
// dotW4A8FoldAVX2, because the two SIMD kernels can agree with each other and both be wrong.
//
// Exact bit-identity with dotW4A8FoldAVX2 is NOT expected and not asserted: two accumulator
// chains sum the groups in a different order, and f32 addition is not associative. The tolerance
// is a relative one, and the delta against the single-chain kernel is reported so the size of
// that reassociation is on record rather than assumed small.
func TestDotW4A8Fold2AccAVX2_matchesOracle(t *testing.T) {
	if !cpufeat.HasAVX2 {
		t.Skip("AVX2 required")
	}
	const group = 32
	rng := rand.New(rand.NewSource(7))
	// Even nGroups only — the kernel's stated contract. 160 is the real FFN shape (K=5120).
	for _, nGroups := range []int{2, 4, 10, 32, 160} {
		K := nGroups * group
		act := make([]int8, K)
		for i := range act {
			act[i] = int8(rng.Intn(256) - 128)
		}
		packed := make([]byte, K/2)
		for i := range packed {
			packed[i] = byte(rng.Intn(256))
		}
		scales := make([]float32, nGroups)
		for i := range scales {
			scales[i] = float32(rng.Float64()*0.05 + 0.0001)
		}

		want := dotW4A8Scalar(act, packed, scales, group, K)
		got := dotW4A8Fold2AccAVX2(&act[0], &packed[0], &scales[0], nGroups)
		one := dotW4A8FoldAVX2(&act[0], &packed[0], &scales[0], nGroups)

		den := math.Abs(float64(want))
		if den < 1e-6 {
			den = 1e-6
		}
		rel := math.Abs(float64(got-want)) / den
		if rel > 1e-4 {
			t.Errorf("nGroups=%d: 2Acc %v vs scalar oracle %v (rel %.3g)", nGroups, got, want, rel)
		}
		t.Logf("nGroups=%-4d oracle %.6f  2Acc %.6f (rel %.2g)  1Acc %.6f (rel %.2g)  2Acc-vs-1Acc %.2g",
			nGroups, want, got, rel, one, math.Abs(float64(one-want))/den,
			math.Abs(float64(got-one))/den)
	}
}

// BenchmarkW4A8_1AccVs2Acc is the direct A/B the issue-width probe is explicitly not allowed to
// substitute for: aikit's own priors note records that probe as "a hint, never load-bearing",
// citing this exact mechanism as the fix a mistrusted reading would have argued against building.
// Order-alternated within one process so a session-level drift cannot land on one arm.
func BenchmarkW4A8_1AccVs2Acc(b *testing.B) {
	if !cpufeat.HasAVX2 {
		b.Skip("AVX2 required")
	}
	const (
		group = 32
		K     = 5120 // the real FFN/hidden projection width
	)
	nGroups := K / group
	rng := rand.New(rand.NewSource(1))
	act := make([]int8, K)
	for i := range act {
		act[i] = int8(rng.Intn(256) - 128)
	}
	packed := make([]byte, K/2)
	for i := range packed {
		packed[i] = byte(rng.Intn(256))
	}
	scales := make([]float32, nGroups)
	for i := range scales {
		scales[i] = float32(rng.Float64()*0.05 + 0.0001)
	}

	b.Run("1Acc/dotW4A8FoldAVX2", func(b *testing.B) {
		var acc float32
		b.ResetTimer()
		for range b.N {
			acc += dotW4A8FoldAVX2(&act[0], &packed[0], &scales[0], nGroups)
		}
		_ = acc
		b.ReportMetric(float64(K)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
	})
	b.Run("2Acc/dotW4A8Fold2AccAVX2", func(b *testing.B) {
		var acc float32
		b.ResetTimer()
		for range b.N {
			acc += dotW4A8Fold2AccAVX2(&act[0], &packed[0], &scales[0], nGroups)
		}
		_ = acc
		b.ReportMetric(float64(K)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
	})
}
