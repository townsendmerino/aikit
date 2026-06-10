// Package ann is the dense (semantic) retriever. v1 is a flat brute-force
// cosine scan — the "vicinity" equivalent in docs/DESIGN.md §1. HNSW lands later
// behind this same Hit/Query shape; flat is exact and fine at repo scale.
//
// Invariants the rest of the codebase depends on:
//
//   - **Input vectors are L2-normalized.** embed.StaticModel.Encode
//     normalizes its output before returning, so cosine similarity is
//     just the dot product — Query computes that, not a full
//     ‖a‖‖b‖-divided cosine. Passing non-normalized vectors silently
//     produces incorrect rankings; the precision contract lives at the
//     embed boundary, not here.
//   - **Scores are float32-precision.** Query scores each vector with the
//     SIMD dot kernel (linalg.Dot, float32 accumulation), not a float64
//     scalar sum. For unit-norm float32 inputs the per-element error is
//     bounded and recall is unaffected; only sub-ULP near-ties may order
//     differently than an exact float64 scan would. The ascending-Index
//     tie-break still makes the result deterministic.
//   - **Similarity, not distance.** semble's dense backend (vicinity)
//     returns cosine *distance* (1 − sim) and search.py flips it back to
//     similarity; ken skips the round-trip and scores similarity
//     directly, with "higher = better." Anything reading the Score
//     field must treat it that way.
//   - **Goroutine-safety.** A built *Flat is read-only — Query takes no
//     locks and is safe to call concurrently across goroutines. New is
//     not thread-safe (single builder); Query is.
//   - **No mutation.** *Flat is immutable after New, by design. v0.3's
//     incremental indexing (internal/search/watch.go, ADR-012) does not
//     mutate an existing *Flat — instead the writer builds a brand-new
//     *Flat alongside a new *bm25.Index + chunks slice, wraps them in a
//     new *search.Index snapshot, and publishes the pointer atomically.
//     Readers load the snapshot pointer once at query entry. So Flat
//     stays goroutine-safe-by-immutability; what changes is that the
//     containing search.Index can be swapped wholesale between queries.
package ann

import (
	"sort"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/aikit/topk"
)

// Hit is one scored item, highest Score (cosine similarity) first.
type Hit struct {
	Index int
	Score float64
}

// Flat is an exhaustive cosine index over a fixed set of unit vectors.
type Flat struct {
	vecs [][]float32 // each assumed L2-normalized (embed.Encode guarantees this)
	dim  int
}

// New builds a flat index. Vectors are used by reference, not copied.
func New(vecs [][]float32) *Flat {
	d := 0
	if len(vecs) > 0 {
		d = len(vecs[0])
	}
	return &Flat{vecs: vecs, dim: d}
}

// Len is the number of indexed vectors.
func (f *Flat) Len() int { return len(f.vecs) }

// Query returns the k highest cosine-similarity vectors to q, descending,
// ties broken by ascending index for determinism. k<=0 or k>=Len returns
// all, sorted.
//
// Two paths by design:
//
//   - k<=0 || k>=Len: full sort over every dot-product result. Preserves
//     the documented "return all, sorted" semantic for callers that want
//     the complete ranked list.
//   - 0 < k < Len: min-heap of size K via internal/topk. O(N log K) vs
//     the full-sort path's O(N log N) — at medium scale (~378k chunks,
//     k=10) this was 30.88% of hybrid search CPU per ADR-025. Final
//     sort.SliceStable imposes the ascending-Index tie-break the doc
//     comment promises, which the heap on its own doesn't guarantee
//     (heap only sorts by score; ties within result come out in
//     heap-internal order). The K-sized stable sort is O(K log K) —
//     cheap at K=10.
func (f *Flat) Query(q []float32, k int) []Hit {
	// Full-sort path: k<=0 returns everything; k>=Len would have nothing
	// to discard anyway. Either way the heap buys us nothing.
	if k <= 0 || k >= len(f.vecs) {
		hits := make([]Hit, 0, len(f.vecs))
		for i, v := range f.vecs {
			if len(v) != len(q) {
				continue // dimension mismatch ⇒ skip rather than panic
			}
			hits = append(hits, Hit{Index: i, Score: float64(linalg.Dot(q, v))})
		}
		sort.Slice(hits, func(a, b int) bool {
			if hits[a].Score != hits[b].Score {
				return hits[a].Score > hits[b].Score
			}
			return hits[a].Index < hits[b].Index
		})
		return hits
	}

	// Heap path: 0 < k < Len. Score every vector, push into the K-sized
	// min-heap; the heap only retains the K highest seen so far.
	sel := topk.New[int](k)
	for i, v := range f.vecs {
		if len(v) != len(q) {
			continue // dimension mismatch ⇒ skip rather than panic
		}
		sel.Push(i, float64(linalg.Dot(q, v)))
	}
	items := sel.Result() // descending by score; tie order is heap-internal
	// Stable secondary sort by ascending Index to honor the doc-comment
	// tie-break contract. K is small (typically 10), so this is cheap.
	sort.SliceStable(items, func(a, b int) bool {
		if items[a].Score != items[b].Score {
			return items[a].Score > items[b].Score
		}
		return items[a].Item < items[b].Item
	})
	hits := make([]Hit, len(items))
	for j, s := range items {
		hits[j] = Hit{Index: s.Item, Score: s.Score}
	}
	return hits
}
