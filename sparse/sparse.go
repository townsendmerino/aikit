// Package sparse is learned-sparse (SPLADE-style) retrieval: an inverted index
// over sparse document vectors, scored by sparse dot product. It is the third
// retrieval signal alongside dense (ann) and lexical (bm25), and feeds the same
// fuse.RRF flow.
//
// A learned-sparse vector is a SPLADE-family model's expansion of a document (or
// query) over a vocabulary: most entries are zero, a few hundred terms carry a
// learned positive weight. The relevance score of a query against a document is
// the sparse dot product of their vectors — score(q, d) = Σ_t q_t·d_t — which an
// inverted index computes by walking only the query's non-zero terms.
//
// # Inference-optional
//
// This package is the index + scorer HALF: New and Query operate on PRE-COMPUTED
// SparseVec values, produced by any SPLADE-family model out of band (e.g. a
// Python export) or in-process via encoder.SPLADE, which reuses encoder's BERT
// machinery to produce the vectors directly — see encoder.LoadSPLADE /
// SPLADE.Expand. Both paths produce the same SparseVec shape, so sparse retrieval
// is equally usable with vectors computed elsewhere or in-process.
//
// # Fusing with dense and lexical
//
// Hit.Index matches ann.Hit, so a sparse ranking joins a hybrid search the same
// way the dense and lexical ones do:
//
//	dense  := fuse.Keys(annHits,    func(h ann.Hit) int  { return h.Index })
//	lexical := fuse.Keys(bm25Hits,  func(r bm25.Result) int { return r.Doc })
//	learned := fuse.Keys(sparseHits, func(h sparse.Hit) int { return h.Index })
//	fused  := fuse.RRF(60, dense, lexical, learned)
package sparse

import (
	"slices"

	"github.com/townsendmerino/aikit/internal/accum"
	"github.com/townsendmerino/aikit/topk"
)

// SparseVec is a sparse vector over a term space (e.g. a SPLADE expansion over a
// BERT vocabulary): Terms holds term ids and Weights the parallel weights, so
// Weights[i] is the weight of term Terms[i]. The two slices are walked to the
// shorter length. Terms need not be sorted or unique — duplicate terms have their
// weights summed. The zero value is the empty (all-zero) vector.
type SparseVec struct {
	Terms   []uint32
	Weights []float32
}

// Hit is one scored document, highest Score (sparse dot product) first. The field
// name Index matches ann.Hit, so sparse and dense rankings feed fuse.Keys
// identically.
type Hit struct {
	Index int
	Score float64
}

// posting is one (document, weight) entry in a term's posting list.
type posting struct {
	doc int32
	w   float32
}

// Index is an inverted index over a corpus of sparse document vectors, scoring a
// query vector by sparse dot product. Build once with New; a built *Index is
// read-only, so Query is safe to call concurrently across goroutines (New is
// not). Immutable after New, like ann.Flat and bm25.Index.
type Index struct {
	postings map[uint32][]posting
	ndocs    int
}

// New builds an index from per-document sparse vectors; document ids are the
// slice indices (docs[i] is document i). The vectors are read, not retained.
// Entries with weight ≤ 0 are skipped: SPLADE weights are non-negative, and a
// non-positive weight cannot raise a dot product of non-negative weights — it
// would only bloat the postings.
func New(docs []SparseVec) *Index {
	ix := &Index{postings: make(map[uint32][]posting), ndocs: len(docs)}
	for d, v := range docs {
		n := min(len(v.Terms), len(v.Weights))
		for i := range n {
			w := v.Weights[i]
			if w <= 0 {
				continue
			}
			t := v.Terms[i]
			ix.postings[t] = append(ix.postings[t], posting{doc: int32(d), w: w})
		}
	}
	return ix
}

// Len is the number of indexed documents.
func (ix *Index) Len() int { return ix.ndocs }

// Scores returns the sparse-dot score of every document for the query vector,
// indexed by document id (length == Len()). Duplicate query terms have their
// weights summed, so each term's posting list is walked at most once.
func (ix *Index) Scores(q SparseVec) []float64 {
	// Public API: dense slice indexed by doc id, materialized from the touched set.
	// Query does NOT go through here — it selects over the touched ids directly.
	a := ix.scoreQuery(q)
	defer accum.Put(a)
	out := make([]float64, ix.ndocs)
	for _, d := range a.Touched {
		out[d] = a.Scores[d]
	}
	return out
}

// Query returns the k highest-scoring documents with Score > 0, ordered by
// descending score, ties broken by ascending document id for determinism. k < 0
// returns every positive-scoring document, sorted; k == 0 returns none. A
// document shares no weighted term with the query iff its score is 0, so the
// Score > 0 filter is "retrieve only documents the query actually touches".
func (ix *Index) Query(q SparseVec, k int) []Hit {
	// WAND dynamic pruning (item 39) was built for this package and MEASURED
	// OUT — see bm25/wand.go for the surviving half. The pivot loop costs
	// O(query terms) per iteration, so it stops paying between 6 and 12 terms,
	// and a SPLADE expansion emits 20-40: at 30 terms it ran 2.4x slower than
	// this scan. What it did win (4.93x) was a hand-built 3-term query, which is
	// not a shape anyone sends to a sparse index. MaxScore is the algorithm for
	// the long-query case and remains open.
	a := ix.scoreQuery(q)
	defer accum.Put(a)

	if k < 0 {
		out := make([]Hit, 0, len(a.Touched))
		for _, d := range a.Touched {
			if s := a.Scores[d]; s > 0 {
				out = append(out, Hit{Index: int(d), Score: s})
			}
		}
		slices.SortFunc(out, hitCmp)
		return out
	}

	// Selection over the TOUCHED SET in ascending doc order — the order the old
	// full-corpus range produced, so topk's first-seen-wins tie-break retains the
	// same items.
	sel := topk.New[int](k)
	th := sel.Threshold()
	for _, d := range a.Touched {
		if s := a.Scores[d]; s > 0 && s > th {
			sel.Push(int(d), s)
			th = sel.Threshold()
		}
	}
	items := sel.Result()
	// Stable secondary sort by ascending doc id to honor the tie-break contract
	// (the heap only orders by score).
	slices.SortFunc(items, topk.ItemCmp[int])
	out := make([]Hit, len(items))
	for j, s := range items {
		out[j] = Hit{Index: s.Item, Score: s.Score}
	}
	return out
}

// hitCmp orders by score descending, then by ascending document id — a strict
// total order (the ids are unique), so slices.SortFunc reproduces the
// previous sort.Slice/SliceStable output exactly without their reflect-based
// Swapper (A5). Same algorithm ann and bm25 each wrote out over their own
// result types; now the one shared topk.Cmp (items sort via the
// directly-usable topk.ItemCmp instead of a local wrapper).
func hitCmp(a, b Hit) int { return topk.Cmp(a.Index, b.Index, a.Score, b.Score) }

// termWeight is one distinct query term and its folded weight.
type termWeight struct {
	term uint32
	w    float64
}

// foldQuery collapses duplicate query terms into dst, summing their weights and
// preserving FIRST-APPEARANCE order.
//
// That order is not cosmetic and this is the single implementation of it, used
// by both the exhaustive scan and the pruning path (item 39). This package
// promises deterministic scores; ranging a map here once made the float64
// accumulation order random per call, so identical queries produced 0.6 vs
// 0.6000000000000001 and ties flipped (audit #16). Two implementations of the
// fold would reintroduce exactly that divergence between the two paths, which
// is why the pruning path calls this rather than repeating it.
//
// The dedupe is a linear scan rather than a map: a SPLADE query is tens of
// terms, where the map's allocation and hashing cost more than the scan (item
// 11, second-order), and a scan preserves first-appearance order by
// construction.
func foldQuery(q SparseVec, dst []termWeight) []termWeight {
	n := min(len(q.Terms), len(q.Weights))
	for i := range n {
		t := q.Terms[i]
		found := false
		for j := range dst {
			if dst[j].term == t {
				dst[j].w += float64(q.Weights[i])
				found = true
				break
			}
		}
		if !found {
			dst = append(dst, termWeight{t, float64(q.Weights[i])})
		}
	}
	return dst
}

// scoreQuery accumulates the query's postings. The caller must accum.Put the result.
//
// Duplicate query terms are folded in FIRST-APPEARANCE order, preserved exactly: this
// package promises deterministic scores, and ranging a map here once made the float64
// accumulation order random per call, so identical queries produced 0.6 vs
// 0.6000000000000001 and ties flipped (audit #16). The dedupe is a linear scan over
// the accumulated terms rather than a map — a SPLADE query is tens of terms, where the
// map's allocation and hashing cost more than the scan (item 11, second-order) — and a
// scan preserves first-appearance order by construction.
func (ix *Index) scoreQuery(q SparseVec) *accum.Accum {
	var stackBuf [64]termWeight
	order := foldQuery(q, stackBuf[:0])
	a := accum.Get(ix.ndocs)
	for _, tw := range order {
		if tw.w == 0 {
			continue
		}
		a.BeginRun()
		for _, p := range ix.postings[tw.term] {
			a.Add(p.doc, tw.w*float64(p.w))
		}
	}
	a.OrderTouched()
	return a
}
