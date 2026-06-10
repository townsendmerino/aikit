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
// Python export, or a future in-process expansion head). A masked-LM expansion
// head that produces the vectors in-process — reusing encoder's NomicBert
// machinery — is a planned follow-up; until then sparse retrieval is fully usable
// by indexing vectors you compute elsewhere.
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
	"sort"

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
		for i := 0; i < n; i++ {
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
	scores := make([]float64, ix.ndocs)
	n := min(len(q.Terms), len(q.Weights))
	qw := make(map[uint32]float64, n)
	for i := 0; i < n; i++ {
		qw[q.Terms[i]] += float64(q.Weights[i])
	}
	for t, w := range qw {
		if w == 0 {
			continue
		}
		for _, p := range ix.postings[t] {
			scores[p.doc] += w * float64(p.w)
		}
	}
	return scores
}

// Query returns the k highest-scoring documents with Score > 0, ordered by
// descending score, ties broken by ascending document id for determinism. k < 0
// returns every positive-scoring document, sorted; k == 0 returns none. A
// document shares no weighted term with the query iff its score is 0, so the
// Score > 0 filter is "retrieve only documents the query actually touches".
func (ix *Index) Query(q SparseVec, k int) []Hit {
	scores := ix.Scores(q)

	if k < 0 {
		out := make([]Hit, 0, len(scores))
		for d, s := range scores {
			if s > 0 {
				out = append(out, Hit{Index: d, Score: s})
			}
		}
		sort.Slice(out, func(i, j int) bool {
			if out[i].Score != out[j].Score {
				return out[i].Score > out[j].Score
			}
			return out[i].Index < out[j].Index
		})
		return out
	}

	sel := topk.New[int](k)
	for d, s := range scores {
		if s > 0 {
			sel.Push(d, s)
		}
	}
	items := sel.Result()
	// Stable secondary sort by ascending doc id to honor the tie-break contract
	// (the heap only orders by score).
	sort.SliceStable(items, func(a, b int) bool {
		if items[a].Score != items[b].Score {
			return items[a].Score > items[b].Score
		}
		return items[a].Item < items[b].Item
	})
	out := make([]Hit, len(items))
	for j, s := range items {
		out[j] = Hit{Index: s.Item, Score: s.Score}
	}
	return out
}
