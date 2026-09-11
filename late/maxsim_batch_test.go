package late

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// maxSimPerPair is the pre-M-20 reference: one linalg.Dot per (query, doc)
// pair, with the first candidate taken unconditionally.
func maxSimPerPair(query, doc [][]float32) float64 {
	var sum float64
	for _, q := range query {
		var best float32
		for j, d := range doc {
			if s := linalg.Dot(q, d); j == 0 || s > best {
				best = s
			}
		}
		sum += float64(best)
	}
	return sum
}

// TestMaxSim_batchedMatchesPerPair gates audit M-20. The 8-row kernel
// reassociates, so this is a tight tolerance rather than equality — the same
// posture Flat and HNSW take for the same kernel. Sizes straddle the group
// boundary (7, 8, 9, 16, 17) and dims straddle the 4-lane tail (63, 64, 65) so
// the ragged paths are covered, and negative scores are exercised because the
// first-candidate seeding is the subtle part.
func TestMaxSim_batchedMatchesPerPair(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 7))
	mk := func(n, d int) [][]float32 {
		out := make([][]float32, n)
		for i := range out {
			v := make([]float32, d)
			for j := range v {
				v[j] = float32(rng.NormFloat64())
			}
			out[i] = v
		}
		return out
	}
	for _, d := range []int{1, 4, 63, 64, 65, 128} {
		for _, nDoc := range []int{0, 1, 7, 8, 9, 16, 17, 33} {
			for _, nQ := range []int{0, 1, 3} {
				q, doc := mk(nQ, d), mk(nDoc, d)
				got, want := MaxSim(q, doc), maxSimPerPair(q, doc)
				if math.Abs(got-want) > 1e-4*math.Max(1, math.Abs(want)) {
					t.Fatalf("d=%d nDoc=%d nQ=%d: MaxSim=%v, per-pair=%v", d, nDoc, nQ, got, want)
				}
			}
		}
	}
}

// TestMaxSim_allNegativeScoresNotFlooredAtZero pins the seeding contract the
// batching had to preserve: cosine can be negative, so a doc whose every token
// is a poor match must score its true (negative) best, not 0.
func TestMaxSim_allNegativeScoresNotFlooredAtZero(t *testing.T) {
	q := [][]float32{{1, 0, 0, 0}}
	// Nine opposing tokens, so the batched group of 8 runs AND a remainder.
	doc := make([][]float32, 9)
	for i := range doc {
		doc[i] = []float32{float32(-1 - i), 0, 0, 0}
	}
	if got, want := MaxSim(q, doc), -1.0; math.Abs(got-want) > 1e-6 {
		t.Errorf("MaxSim = %v, want %v (the least-negative token, not 0)", got, want)
	}
}
