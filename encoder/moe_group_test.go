package encoder

import (
	"math/rand"
	"testing"
)

// moeReference is the pre-item-33 moeMLP: both expert projections at M=1, once
// per (token, rank). Kept as the oracle — item 33 claims to change only WHICH
// tokens share a GEMM call, never the arithmetic.
func moeReference(h, router, w1, w2t, bias []float32, numExperts, topK, D, intermediate, L int, s *scratch) {
	// The oracle's own router-logit row and combination accumulator. Local, not
	// pooled: the production moeMLP no longer has per-token buffers to share.
	scores := make([]float32, numExperts)
	out := make([]float32, D)
	x1 := s.val[:intermediate]
	contrib := s.mid[:D]

	for t := range L {
		row := h[t*D : (t+1)*D]
		s.mm(row, router, scores, 1, D, numExperts)
		softmaxRow(scores)
		clear(out)
		for range topK {
			best, bestIdx := float32(-1), -1
			for e, sc := range scores {
				if sc > best {
					best, bestIdx = sc, e
				}
			}
			if bestIdx < 0 {
				break
			}
			scores[bestIdx] = -1
			off := bestIdx * intermediate
			matmulBTBlockedInto(row, w1[off*D:(off+intermediate)*D], x1, 1, D, intermediate)
			gelu(x1)
			matmulBTBlockedInto(x1, w2t[off*D:(off+intermediate)*D], contrib, 1, intermediate, D)
			for j := range D {
				out[j] += best * contrib[j]
			}
		}
		for j := range D {
			row[j] += out[j] + bias[j]
		}
	}
}

// TestMoEMLP_groupedMatchesPerToken is the gate for item 33. Grouping tokens by
// expert must be BIT-IDENTICAL, and it rests on two separate properties:
//
//   - linalg's M-invariance: dst[i,j] does not depend on M, so a token batched
//     with others yields the same bits as computed alone.
//   - the rank loop staying OUTERMOST, so each token accumulates rank 0 before
//     rank 1 exactly as before. Grouping across ranks would reorder that sum,
//     and float addition is not associative.
//
// Exact equality, not a tolerance — a tolerance would accept a reordered sum,
// which is precisely what must not happen.
func TestMoEMLP_groupedMatchesPerToken(t *testing.T) {
	rng := rand.New(rand.NewSource(33))
	rnd := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64()) * 0.05
		}
		return v
	}
	for _, tc := range []struct {
		L, D, inter, experts, topK int
	}{
		{1, 64, 128, 8, 2},   // single token: every expert bucket has 0 or 1
		{7, 64, 128, 8, 2},   // fewer tokens than experts
		{64, 96, 192, 8, 2},  // the real shape's ratio
		{64, 96, 192, 4, 1},  // top-1: no rank ordering to preserve
		{33, 96, 192, 3, 3},  // topK == numExperts: every expert serves every token
		{128, 64, 128, 8, 2}, // bigger M per expert
	} {
		router := rnd(tc.experts * tc.D)
		w1 := rnd(tc.experts * tc.inter * tc.D)
		w2t := rnd(tc.experts * tc.inter * tc.D)
		bias := rnd(tc.D)
		base := rnd(tc.L * tc.D)

		want := append([]float32(nil), base...)
		sRef := getScratch()
		sRef.ensureLayer(scratchLayout{}, tc.L, tc.D, tc.inter, 1, tc.D, tc.L)
		moeReference(want, router, w1, w2t, bias, tc.experts, tc.topK, tc.D, tc.inter, tc.L, sRef)
		putScratch(sRef)

		got := append([]float32(nil), base...)
		sGot := getScratch()
		sGot.ensureLayer(scratchLayout{}, tc.L, tc.D, tc.inter, 1, tc.D, tc.L)
		moeMLP(got, router, w1, w2t, bias, tc.experts, tc.topK, tc.D, tc.inter, tc.L, false, sGot)
		putScratch(sGot)

		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("L=%d D=%d experts=%d topK=%d: element %d (token %d, dim %d) = %v, per-token reference %v",
					tc.L, tc.D, tc.experts, tc.topK, i, i/tc.D, i%tc.D, got[i], want[i])
			}
		}
	}
}

// TestMoEMLP_endToEndMatchesReference runs the real checkpoint through both
// paths. The synthetic test above uses random weights, which route roughly
// uniformly; a trained router is skewed, so some experts take many tokens and
// others none — the bucket-size distribution the grouped path actually meets.
func TestMoEMLP_endToEndMatchesReference(t *testing.T) {
	m := loadMoE(t)
	defer func() { _ = m.Close() }()

	text := moeText(120)
	want, err := m.Encode(text, false)
	if err != nil {
		t.Fatal(err)
	}
	// Re-encode: the grouped path must be deterministic across calls too, since
	// it now reuses routing scratch between layers and forwards.
	got, err := m.Encode(text, false)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("component %d differs between two encodes of the same text: %v vs %v",
				i, got[i], want[i])
		}
	}
}
