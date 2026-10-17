package bm25

import (
	"math"
	"slices"

	"github.com/townsendmerino/aikit/internal/accum"
	"github.com/townsendmerino/aikit/topk"
)

// Result is one scored document, highest Score first.
type Result struct {
	Doc   int
	Score float64
}

// idf is the Lucene/bm25s BM25 IDF: ln(1 + (N - df + 0.5)/(df + 0.5)).
// The +1 inside the log keeps it non-negative even for terms in most docs,
// which is the variant bm25s uses by default.
func (ix *Index) idf(term string) float64 {
	e := ix.entry(term)
	if e == nil {
		return 0
	}
	return ix.idfOf(e)
}

// idfOf is idf for a term whose entry the caller already holds — the scoring
// paths look the term up once and would otherwise hash it a second time.
func (ix *Index) idfOf(e *termEntry) float64 {
	if e.df == 0 {
		return 0
	}
	n := float64(ix.N())
	return math.Log(1 + (n-float64(e.df)+0.5)/(float64(e.df)+0.5))
}

// IDF returns the Lucene/bm25s BM25 IDF for term. Public wrapper around
// the unexported idf used by query scoring; identical formula. Returns 0
// for unknown terms. Exposed for downstream tooling that ranks tokens
// by corpus distinctiveness — e.g. a pseudo-relevance-feedback (PRF)
// term harvester, or an oracle identifier picker for retrieval-eval
// experiments.
func (ix *Index) IDF(term string) float64 { return ix.idf(term) }

// DF returns the raw document-frequency of term — the number of indexed
// documents that contain it at least once. Returns 0 for unknown terms.
// Paired with IDF for tooling that needs to *filter* by DF (e.g. drop
// near-hapax tokens that are too rare to predict) before ranking by IDF.
func (ix *Index) DF(term string) int {
	if e := ix.entry(term); e != nil {
		return int(e.df)
	}
	return 0
}

// Scores returns the BM25 score of every document for query (already
// tokenized). The result is indexed by document id; length == ix.N().
func (ix *Index) Scores(query []string) []float64 {
	// Public API: must return a dense slice indexed by doc id, so this materializes
	// one from the touched set. TopK does NOT go through here — it selects over the
	// touched ids directly, which is the whole point of item 11.
	a := ix.scoreQuery(query)
	defer accum.Put(a)
	out := make([]float64, ix.N())
	for _, d := range a.Touched {
		out[d] = a.Scores[d]
	}
	return out
}

// TopK returns the k highest-scoring documents with Score > 0, ties broken
// by ascending document id so results are deterministic.
//
// Two paths by design:
//
//   - k<0: full sort over every scoring doc. Preserves the "no truncation"
//     escape hatch the original `if k >= 0 && k < len(res)` gate exposed
//     to callers that want every positive-scored document.
//   - k>=0: min-heap of size K via internal/topk. O(N log K) vs the
//     full-sort path's O(N log N). At medium scale (~378k chunks, k=10)
//     this was 36% of bm25 search CPU per ADR-025. Final sort.SliceStable
//     imposes the ascending-Doc tie-break the doc comment promises,
//     which the heap on its own doesn't guarantee. K-sized stable sort
//     is O(K log K) — cheap at K=10. k=0 returns an empty slice (topk
//     with cap 0 always discards), matching the prior `k=0 → empty`
//     behavior from the original truncation gate.
func (ix *Index) TopK(query []string, k int) []Result {
	// Dynamic pruning (item 39) answers the k>=0 case without scoring documents
	// that cannot reach the top k. It is EXACT — same documents, same scores,
	// bit for bit — so this is a pure implementation swap, not a mode. It
	// declines for parameters that would invalidate its bound, and for k<0,
	// which asks for every scoring document and so has nothing to prune.
	if k >= 0 {
		if out, ok := ix.topKWAND(query, k); ok {
			return out
		}
	}

	return ix.topKExhaustive(query, k)
}

// topKExhaustive is the pre-item-39 TopK: score every posting of every query
// term, then select. It remains the reference the pruning path is gated
// against, the answer for k<0, and the fallback whenever the bound is unusable.
func (ix *Index) topKExhaustive(query []string, k int) []Result {
	a := ix.scoreQuery(query)
	defer accum.Put(a)

	// Full-sort path: k<0 means "no truncation, return everything".
	if k < 0 {
		res := make([]Result, 0, len(a.Touched))
		for _, d := range a.Touched {
			if s := a.Scores[d]; s > 0 {
				res = append(res, Result{Doc: int(d), Score: s})
			}
		}
		slices.SortFunc(res, resultCmp)
		return res
	}

	// Heap path: k>=0. Push every positive-scored doc; the heap retains
	// the K highest. k=0 selector discards everything → empty result,
	// matching the prior gate's k=0 behavior.
	// Selection over the TOUCHED SET, in ascending doc order — the same order the
	// old full-corpus range produced, so topk's first-seen-wins tie-break retains
	// exactly the same items.
	sel := topk.New[int](k)
	th := sel.Threshold()
	for _, d := range a.Touched {
		if s := a.Scores[d]; s > 0 && s > th {
			sel.Push(int(d), s)
			th = sel.Threshold()
		}
	}
	items := sel.Result()
	// Stable secondary sort by ascending Doc id to honor the doc-comment
	// tie-break contract.
	slices.SortFunc(items, topk.ItemCmp[int])
	out := make([]Result, len(items))
	for j, s := range items {
		out[j] = Result{Doc: s.Item, Score: s.Score}
	}
	return out
}

// resultCmp orders by score descending, then by ascending document id — a
// strict total order (Doc is unique within the slice), so slices.SortFunc
// reproduces the previous sort.Slice/SliceStable output exactly while
// avoiding their reflect-based Swapper (A5; the same change audit #24 made
// in ann/hnsw.go). Same algorithm ann and sparse each wrote out over their
// own result types; now the one shared topk.Cmp (items sort via the
// directly-usable topk.ItemCmp instead of a local wrapper).
func resultCmp(a, b Result) int { return topk.Cmp(a.Doc, b.Doc, a.Score, b.Score) }

// scoreQuery accumulates every (deduped) query term's postings and returns the
// accumulator. The caller must accum.Put it.
func (ix *Index) scoreQuery(query []string) *accum.Accum {
	a := accum.Get(ix.N())
	for i, term := range query {
		if dupTerm(query, i) {
			continue // term contributes once; tf is per-document, not per-query
		}
		e := ix.entry(term)
		if e == nil {
			continue
		}
		idf := ix.idfOf(e)
		if idf == 0 {
			continue
		}
		a.BeginRun()
		// K1/B are exported and may be tuned by the caller before querying, so
		// they cannot be folded into the postings at Build (item 10's
		// "precompute the impact" would bake them in). Hoisting them out of the
		// posting loop is free and keeps them live.
		k1, bb := ix.K1, ix.B
		k1p1 := k1 + 1
		for _, p := range ix.list(e) {
			tf := float64(p.tf)
			denom := tf + k1*(1-bb+bb*ix.norm[p.doc])
			a.Add(p.doc, idf*(tf*k1p1)/denom)
		}
	}
	a.OrderTouched()
	return a
}

// dupTerm reports whether query[i] appeared earlier in query. A linear scan over the
// preceding terms replaces the per-query `seen` map: queries are tens of terms, where
// the map's allocation and hashing cost more than the scan (item 11, second-order).
func dupTerm(query []string, i int) bool {
	for j := range i {
		if query[j] == query[i] {
			return true
		}
	}
	return false
}
