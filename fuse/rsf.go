package fuse

import "sort"

// Scored is one ranked item with its raw score — the input to RSF, which fuses by
// score magnitude rather than rank alone. The score-aware counterpart of the bare
// key that RRF consumes.
type Scored[K comparable] struct {
	Key   K
	Score float64
}

// RSF fuses rankings by Relative Score Fusion: each ranking's raw scores are
// min-max normalized to [0,1] independently, then summed per item. Equivalent to
// RSFWeighted with every weight 1.
//
// Where RRF uses only rank order, RSF preserves how MUCH better one hit is than
// the next within a list — so a dense ranking with one dominant cosine match and
// a long flat tail fuses differently than under RRF. Use RSF when the per-ranking
// scores are meaningfully calibrated (e.g. cosine similarities, BM25 within one
// corpus); use RRF when the score scales are incomparable or noisy and only order
// is trustworthy. Normalization is per-ranking, so the two lists' raw scales need
// not match — only their within-list spreads.
//
// Returns items by descending fused score, ties broken by first-appearance order
// across the rankings (deterministic). An item absent from a ranking contributes
// 0 from it. A ranking whose scores are all equal (or has one item) normalizes
// every member to 1.0 — it carries order but no within-list signal.
func RSF[K comparable](rankings ...[]Scored[K]) []Result[K] {
	return RSFWeighted(nil, rankings...)
}

// RSFWeighted is RSF with a per-ranking weight — the score-based analogue of
// RRFWeighted, for tilting the blend toward the dense or the lexical side. weights
// may be nil (all 1) or must have one entry per ranking; a length mismatch panics.
func RSFWeighted[K comparable](weights []float64, rankings ...[]Scored[K]) []Result[K] {
	if weights != nil && len(weights) != len(rankings) {
		panic("fuse: len(weights) must equal number of rankings")
	}

	scores := make(map[K]float64)
	firstSeen := make(map[K]int)
	order := 0

	for r, ranking := range rankings {
		if len(ranking) == 0 {
			continue
		}
		w := 1.0
		if weights != nil {
			w = weights[r]
		}
		lo, hi := ranking[0].Score, ranking[0].Score
		for _, s := range ranking {
			if s.Score < lo {
				lo = s.Score
			}
			if s.Score > hi {
				hi = s.Score
			}
		}
		span := hi - lo
		for _, s := range ranking {
			if _, ok := firstSeen[s.Key]; !ok {
				firstSeen[s.Key] = order
				order++
			}
			norm := 1.0 // all-equal (or single-item) list: no within-list signal
			if span > 0 {
				norm = (s.Score - lo) / span
			}
			scores[s.Key] += w * norm
		}
	}

	out := make([]Result[K], 0, len(scores))
	for key, s := range scores {
		out = append(out, Result[K]{Key: key, Score: s})
	}
	sort.SliceStable(out, func(a, b int) bool {
		if out[a].Score != out[b].Score {
			return out[a].Score > out[b].Score
		}
		return firstSeen[out[a].Key] < firstSeen[out[b].Key]
	})
	return out
}

// Scores projects a typed result slice to []Scored[K] for RSF — the score-aware
// counterpart of Keys:
//
//	dense := fuse.Scores(annHits, func(h ann.Hit) int { return h.Index },
//	                              func(h ann.Hit) float64 { return h.Score })
func Scores[T any, K comparable](items []T, key func(T) K, score func(T) float64) []Scored[K] {
	out := make([]Scored[K], len(items))
	for i, it := range items {
		out[i] = Scored[K]{Key: key(it), Score: score(it)}
	}
	return out
}
