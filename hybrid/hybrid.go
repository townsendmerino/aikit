// Package hybrid is a thin, opt-in convenience wrapper around the
// retrieve-both-signals-then-fuse step every hybrid-search example in this
// repo (examples/rag, examples/colbert) hand-writes identically:
//
//	den := dense.Query(queryVec, shortlist)
//	lex := lexical.TopK(queryTokens, shortlist)
//	fused := fuse.RRF(fuse.DefaultK,
//	    fuse.Keys(den, func(h ann.Hit) int { return h.Index }),
//	    fuse.Keys(lex, func(r bm25.Result) int { return r.Doc }))
//
// Retriever.Query is exactly that, in one call. Nothing here requires it —
// docs/architecture.md's own pipeline diagram is captioned "how a consumer
// composes it", not a shape aikit enforces — so hand-wiring the four lines
// above stays equally valid; this package exists only to cut the repetition
// for the common case.
//
// # What it deliberately does NOT own
//
// A Retriever composes two ALREADY-BUILT indices; it does not build them
// (different callers want different dense index types — ann.Flat, FlatI8,
// HNSW, ...built however they like), embed or tokenize the query (different
// models, different tokenizers), chunk documents, or rerank the fused
// shortlist (encoder.CrossEncoder and late.Index need different inputs — pair
// text vs. per-token vectors — so composing one of them in here would either
// pick a winner or need to accommodate both, and the fused Result the caller
// gets back is exactly what those examples' own rerank stage already consumes
// as input). Each of those stays the caller's job, same as fuse itself
// "knows nothing about bm25, ann, or embeddings" (fuse's own doc comment) —
// this package is one level up that ladder, not a framework replacing it.
package hybrid

import (
	"slices"

	"github.com/townsendmerino/aikit/ann"
	"github.com/townsendmerino/aikit/bm25"
	"github.com/townsendmerino/aikit/fuse"
)

// DenseIndex is the subset of ann.Flat / ann.FlatI8 / ann.HNSW / ann.FlatBinary*
// a Retriever needs — every one of them already satisfies it as-is.
type DenseIndex interface {
	Query(q []float32, k int) []ann.Hit
}

// LexicalIndex is the subset of *bm25.Index a Retriever needs.
type LexicalIndex interface {
	TopK(query []string, k int) []bm25.Result
}

// Retriever composes an already-built dense index and lexical index, both
// over the SAME corpus (shared row-index id space, exactly as ann.Hit.Index
// and bm25.Result.Doc already assume of each other in every example that
// fuses them).
type Retriever struct {
	Dense   DenseIndex
	Lexical LexicalIndex
}

// New returns a Retriever over dense and lexical. Neither is copied; build
// them however you like (ann.New, ann.LoadFlatI8Mmap, bm25.Build, a loaded
// index, ...) before passing them in.
func New(dense DenseIndex, lexical LexicalIndex) *Retriever {
	return &Retriever{Dense: dense, Lexical: lexical}
}

// Query retrieves up to `shortlist` candidates from each signal and
// rank-fuses them with fuse.RRF(fuse.DefaultK, ...) — descending fused score,
// ties broken by first-appearance (dense list scanned first, then lexical).
// Result.Key is the corpus row index, ready to rerank or display directly, or
// hand to a fuse.RRFWeighted call of your own if fuse.DefaultK's unweighted
// blend isn't what you want (Retriever intentionally has no weighting knob —
// use fuse directly for that).
//
// shortlist <= 0 returns nil. This is a deliberate normalization, not a
// passthrough: ann.Flat.Query treats k<=0 as "return the whole corpus" while
// bm25.Index.TopK treats k=0 as "return nothing" — composing them raw would
// make Query's k=0 behavior depend on which DenseIndex/LexicalIndex
// implementation was passed to New, silently. "No results requested" is the
// one answer that can't surprise either way.
func (r *Retriever) Query(queryVec []float32, queryTokens []string, shortlist int) []fuse.Result[int] {
	if shortlist <= 0 {
		return nil
	}
	den := r.Dense.Query(queryVec, shortlist)
	lex := r.Lexical.TopK(queryTokens, shortlist)
	total := len(den) + len(lex)
	out := make([]fuse.Result[int], 0, total)
	k := fuse.DefaultK

	if total <= 64 {
		// Small shortlist fast-path: linear scan avoids map allocation entirely.
		//
		// Both loops below must dedup against out, not just against each other:
		// DenseIndex/LexicalIndex are caller-supplied interfaces (see the doc
		// comment above), not guaranteed duplicate-free the way this repo's own
		// ann/bm25 implementations happen to be — a single ranking repeating a
		// key would otherwise get two entries here, splitting its score instead
		// of summing it (fuse.RRFWeighted, which this inlines, dedups every
		// ranking the same way).
		for rank0, h := range den {
			key := h.Index
			found := false
			for i := range out {
				if out[i].Key == key {
					out[i].Score += 1.0 / (k + float64(rank0+1))
					found = true
					break
				}
			}
			if !found {
				out = append(out, fuse.Result[int]{Key: key, Score: 1.0 / (k + float64(rank0+1))})
			}
		}
		for rank0, res := range lex {
			key := res.Doc
			found := false
			for i := range out {
				if out[i].Key == key {
					out[i].Score += 1.0 / (k + float64(rank0+1))
					found = true
					break
				}
			}
			if !found {
				out = append(out, fuse.Result[int]{Key: key, Score: 1.0 / (k + float64(rank0+1))})
			}
		}
	} else {
		// Large shortlist: hashmap lookup. Same dedup requirement as the small
		// path above, for both rankings.
		pos := make(map[int]int, total)
		for rank0, h := range den {
			key := h.Index
			if i, ok := pos[key]; ok {
				out[i].Score += 1.0 / (k + float64(rank0+1))
			} else {
				pos[key] = len(out)
				out = append(out, fuse.Result[int]{Key: key, Score: 1.0 / (k + float64(rank0+1))})
			}
		}
		for rank0, res := range lex {
			key := res.Doc
			if i, ok := pos[key]; ok {
				out[i].Score += 1.0 / (k + float64(rank0+1))
			} else {
				pos[key] = len(out)
				out = append(out, fuse.Result[int]{Key: key, Score: 1.0 / (k + float64(rank0+1))})
			}
		}
	}

	slices.SortStableFunc(out, func(a, b fuse.Result[int]) int {
		switch {
		case a.Score > b.Score:
			return -1
		case a.Score < b.Score:
			return 1
		}
		return 0
	})
	return out
}
