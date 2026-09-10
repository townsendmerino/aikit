package linalg

import (
	"math/rand/v2"
	"slices"
	"testing"
)

// TestWorkspace_quantizeActivationsSurvivesWeightOnlyMatmul is audit C-05.
//
// QuantizeActivations hands the caller views into the Workspace's own scratch
// and documents how long they stay valid. The weight-only Q8/Q4 matmuls used to
// take their dequantized-weight-row scratch from the SAME f32 buffer the
// activation scales live in, via f32Buf. Because that buffer was already at
// least K long, taking a K-wide (or 8K-wide) slice of it did not GROW it — so
// the documented "valid until the next call that grows the scratch" rule was
// satisfied while the scales were being overwritten with dequantized weight
// values underneath the caller. Finite, plausible, and silently wrong.
//
// The interleaving here is the one the audit names: quantize once, then run a
// weight-only matmul on the same Workspace before consuming the scales.
func TestWorkspace_quantizeActivationsSurvivesWeightOnlyMatmul(t *testing.T) {
	const M, K, N = 2, 64, 8 // K >= M is what makes the old aliasing bite
	rng := rand.New(rand.NewPCG(3, 5))
	rv := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}

	var ws Workspace
	a := rv(M * K)
	w := rv(N * K)
	bQ, bScales := QuantizeRowsInt8(w, N, K)
	dst := make([]float32, M*N)

	// WARM FIRST. This is what makes the test bite, and it is the part that is
	// easy to get wrong: if the Workspace is cold, the span's f32Buf(8*K) call
	// GROWS the buffer, which allocates a fresh array and leaves the caller's
	// older scales view pointing at the abandoned one — intact, so a naive
	// ordering here passes even against the defect. The hazard is the in-place
	// case: once the buffer is already big enough, no growth occurs and the
	// dequantized weight rows are written straight over the scales.
	MatmulBTQ8Into(&ws, a, bQ, bScales, dst, M, K, N)

	_, scales := ws.QuantizeActivations(a, M, K)
	want := slices.Clone(scales)
	MatmulBTQ8Into(&ws, a, bQ, bScales, dst, M, K, N)
	if !slices.Equal(scales, want) {
		t.Errorf("MatmulBTQ8Into overwrote the activation scales:\n got %v\nwant %v", scales, want)
	}

	// Same for the group-int4 weight-only path, warmed the same way.
	const group = 32
	packed, gScales := QuantizeGroupsInt4(w, N, K, group)
	var ws4 Workspace
	MatmulBTQ4Into(&ws4, a, packed, gScales, dst, M, K, N, group)
	_, scales4 := ws4.QuantizeActivations(a, M, K)
	want4 := slices.Clone(scales4)
	MatmulBTQ4Into(&ws4, a, packed, gScales, dst, M, K, N, group)
	if !slices.Equal(scales4, want4) {
		t.Errorf("MatmulBTQ4Into overwrote the activation scales:\n got %v\nwant %v", scales4, want4)
	}
}
