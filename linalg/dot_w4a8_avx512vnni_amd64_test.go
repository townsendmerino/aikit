//go:build amd64

package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// TestAVX512VNNIVL_detection sanity-checks the AVX512VL probe runs without
// faulting — mirrors TestAVX512VNNI_detection. Doesn't assert VL presence
// (host-dependent), only that CPUID leaf 7 itself is being read correctly.
func TestAVX512VNNIVL_detection(t *testing.T) {
	maxLeaf, _, _, _ := cpuid(0, 0)
	if maxLeaf < 1 {
		t.Fatalf("CPUID leaf 0 reported maxLeaf=%d; probe is broken", maxLeaf)
	}
	t.Logf("hasAVX512VNNIVL=%v hasAVX512VNNI=%v hasAVX2=%v", hasAVX512VNNIVL, hasAVX512VNNI, hasAVX2)
}

// TestAVX512VNNI_dotW4A8FoldAVX512VNNI_matchesScalar directly exercises the
// raw VPDPBUSD W4A8 kernel (bypassing dotW4A8's dispatch) against
// dotW4A8Scalar, at nGroups values covering small, boundary, and
// benchmark-typical sizes. Like dotW4A8FoldAVX2, this kernel folds the
// per-group scale in f32, so the bar is TestW4A8_dotMatchesScalar's relative
// tolerance (1e-5), not bit-exact equality. Weighted toward the nibble/sign
// extremes specifically — nibble 0 and 15 (weight -8 and +7) crossed with
// activation -128 and 127 — since that's where an off-by-one in the
// uncentered-nibble correction (Σnib·act - 8·Σact) would first show up.
// Skips when the host lacks AVX-512 VNNI+VL (the asm would SIGILL).
func TestAVX512VNNI_dotW4A8FoldAVX512VNNI_matchesScalar(t *testing.T) {
	if !hasAVX512VNNIVL {
		t.Skip("CPU lacks AVX-512 VNNI+VL; dotW4A8FoldAVX512VNNI asm path not exercised")
	}
	const group = 32
	rng := rand.New(rand.NewSource(97))

	packNibbles := func(nib []byte) []byte {
		packed := make([]byte, len(nib)/2)
		for i := 0; i < len(nib); i += 2 {
			packed[i/2] = (nib[i] & 0x0F) | ((nib[i+1] & 0x0F) << 4)
		}
		return packed
	}
	relErr := func(got, want float32) float64 {
		return math.Abs(float64(got-want)) / (math.Abs(float64(want)) + 1e-9)
	}

	// Fully random nibbles/activations/scales across a spread of group counts.
	for _, nGroups := range []int{1, 2, 3, 7, 24, 64, 96} {
		K := nGroups * group
		nib := make([]byte, K)
		act := make([]int8, K)
		scales := make([]float32, nGroups)
		for i := range nib {
			nib[i] = byte(rng.Intn(16))
			act[i] = int8(rng.Intn(256) - 128)
		}
		for g := range scales {
			scales[g] = float32(rng.Float64()*0.05 + 0.0001)
		}
		packed := packNibbles(nib)
		got := dotW4A8FoldAVX512VNNI(&act[0], &packed[0], &scales[0], nGroups)
		want := dotW4A8Scalar(act, packed, scales, group, K)
		if re := relErr(got, want); re > 1e-5 {
			t.Errorf("random nGroups=%d: got=%v want=%v relErr=%.2e, want ≤ 1e-5", nGroups, got, want, re)
		}
	}

	// Nibble/activation sign-magnitude extremes.
	for _, nGroups := range []int{1, 4, 24} {
		K := nGroups * group
		for _, combo := range [][2]int{{0, -128}, {0, 127}, {15, -128}, {15, 127}} {
			nib := make([]byte, K)
			act := make([]int8, K)
			for i := range nib {
				nib[i] = byte(combo[0])
				act[i] = int8(combo[1])
			}
			scales := make([]float32, nGroups)
			for g := range scales {
				scales[g] = 1.0
			}
			packed := packNibbles(nib)
			got := dotW4A8FoldAVX512VNNI(&act[0], &packed[0], &scales[0], nGroups)
			want := dotW4A8Scalar(act, packed, scales, group, K)
			if re := relErr(got, want); re > 1e-5 {
				t.Errorf("extreme nib=%d act=%d nGroups=%d: got=%v want=%v relErr=%.2e", combo[0], combo[1], nGroups, got, want, re)
			}
		}
	}

	// nGroups=0.
	if got := dotW4A8FoldAVX512VNNI(nil, nil, nil, 0); got != 0 {
		t.Errorf("nGroups=0: got=%v, want 0", got)
	}
}

// TestAVX512VNNI_dotW4A8_dispatchesThroughEveryTier exercises the full
// dotW4A8 dispatcher (not the raw kernel) at K values straddling the VNNI
// and AVX2 group-count boundary and the ragged-K%32 tail, so a mismatch
// between tiers (not just within one kernel) would show up. Runs regardless
// of hasAVX512VNNIVL/hasAVX2: on lesser hardware it exercises whichever
// tiers are present.
func TestAVX512VNNI_dotW4A8_dispatchesThroughEveryTier(t *testing.T) {
	const group = 32
	rng := rand.New(rand.NewSource(103))
	for _, K := range []int{32, 64, 96, 300, 768, 2048, 3072, 3100} {
		nGroups := (K + group - 1) / group
		act := make([]int8, K)
		for i := range act {
			act[i] = int8(rng.Intn(256) - 128)
		}
		packed := make([]byte, (K+1)/2)
		for i := range packed {
			packed[i] = byte(rng.Intn(256))
		}
		scales := make([]float32, nGroups)
		for i := range scales {
			scales[i] = float32(rng.NormFloat64())
		}
		got := dotW4A8(act, packed, scales, group, K)
		want := dotW4A8Scalar(act, packed, scales, group, K)
		rel := math.Abs(float64(got-want)) / (math.Abs(float64(want)) + 1e-9)
		if rel > 1e-5 {
			t.Errorf("K=%d: dotW4A8=%v scalar=%v relErr=%.2e, want ≤ 1e-5", K, got, want, rel)
		}
	}
}

// TestAVX512VNNI_dotW4A8FoldAVX512VNNI_centersInInt32 pins the 2026-09-24 fix: the Σact correction is
// applied to the exact int32 partials BEFORE the f32 fold. The pre-fix kernel folded Σnib·act and
// −8·Σact into two f32 accumulators and combined them at the end, so each carried the UNCENTERED
// magnitude and their rounding did not cancel (measured: up to 3.2e-3 per logit through goinfer's int4
// forward, failing its nemotron-tiny golden on AVX-512 runners).
//
// The check is built so the two kernels' errors are far apart: weights almost all 0 (nibble 8, with a
// sparse ±1) against large same-sign activations, so the uncentered terms are ~100x the centered result.
// The reference is exact (float64 of the exact int partials); the bar is the textbook recursive-summation
// bound on the CENTERED terms, (nGroups+8)·u·Σ|terms| with u = 2^-24. A kernel whose error scales with the
// centered magnitude (this one) meets it; one whose error scales with the uncentered magnitude (the
// pre-fix kernel) misses it by an order of magnitude — verified by running this body against exact Go
// models of both kernels on an AVX2 host (checkDotW4A8FoldCentering; the model lives outside the repo).
// (An all-zero-weight case is deliberately NOT used: there the old kernel's two accumulators are exact
// negatives of each other at every step and cancel perfectly, so it would pass either kernel.)
//
// Skips when the host lacks AVX-512 VNNI+VL (the asm would SIGILL) — i.e. on both project boxes; it is
// meant to run on CI's AVX-512 runners, and its Logf says when it did.
func TestAVX512VNNI_dotW4A8FoldAVX512VNNI_centersInInt32(t *testing.T) {
	if !hasAVX512VNNIVL {
		t.Skip("CPU lacks AVX-512 VNNI+VL; dotW4A8FoldAVX512VNNI asm path not exercised")
	}
	t.Logf("running on an AVX-512 VNNI+VL host (hasAVX2=%v)", hasAVX2)
	checkDotW4A8FoldCentering(t, func(act []int8, packed []byte, scales []float32, nGroups int) float32 {
		return dotW4A8FoldAVX512VNNI(&act[0], &packed[0], &scales[0], nGroups)
	})
}

// checkDotW4A8FoldCentering is the body of TestAVX512VNNI_dotW4A8FoldAVX512VNNI_centersInInt32, split out so
// the checks themselves can be exercised against a portable model of the kernel on hosts without VNNI.
func checkDotW4A8FoldCentering(t *testing.T, kernel func(act []int8, packed []byte, scales []float32, nGroups int) float32) {
	t.Helper()
	rng := rand.New(rand.NewSource(211))
	pack := func(nib []byte) []byte {
		p := make([]byte, len(nib)/2)
		for i := 0; i < len(nib); i += 2 {
			p[i/2] = (nib[i] & 0x0F) | ((nib[i+1] & 0x0F) << 4)
		}
		return p
	}
	for _, nGroups := range []int{1, 2, 7, 24, 96, 256} {
		for trial := range 4 {
			K := nGroups * 32
			act := make([]int8, K)
			nib := make([]byte, K)
			for i := range act {
				act[i] = int8(100 + rng.Intn(28)) // large, same sign: Σact does not cancel
				nib[i] = 8                        // weight 0 …
				if rng.Intn(16) == 0 {
					nib[i] = byte(7 + 2*rng.Intn(2)) // … except a sparse ±1
				}
			}
			scales := make([]float32, nGroups)
			for g := range scales {
				scales[g] = float32(rng.Float64()*0.05 + 0.0001) // non-dyadic: products do not round exactly
			}
			var exact, mag float64
			for g, s := range scales {
				var c, m int64
				for k := g * 32; k < g*32+32; k++ {
					v := int64(int(nib[k])-8) * int64(act[k])
					c += v
					if v < 0 {
						v = -v
					}
					m += v
				}
				exact += float64(s) * float64(c)
				mag += math.Abs(float64(s)) * float64(m)
			}
			tol := float64(nGroups+8)*mag/(1<<24) + 1e-30
			got := kernel(act, pack(nib), scales, nGroups)
			if d := math.Abs(float64(got) - exact); d > tol {
				t.Errorf("nGroups=%d trial=%d: got %.9g, exact %.9g, |Δ| %.3g > %.3g — the error scales with the "+
					"UNCENTERED magnitude: the Σact correction is not being applied in int32 before the f32 fold",
					nGroups, trial, got, exact, d, tol)
			}
		}
	}
}
