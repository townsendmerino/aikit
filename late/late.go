// Package late is ColBERT-style late-interaction (MaxSim) reranking: instead of
// pooling a query or document down to one vector (a bi-encoder, encoder's usual
// path) or scoring a (query, document) pair jointly in one forward
// (encoder.CrossEncoder), late interaction keeps every token's own
// contextualized vector and lets each query token independently find its
// best-matching document token:
//
//	MaxSim(q, d) = Σ_i max_j cos(q_i, d_j)
//
// # Inference-optional
//
// Like sparse, this package is the SCORING half: MaxSim / ScoreBatch / Index
// operate on PRE-COMPUTED per-token vectors, produced out of band or in-process
// via encoder.Model.EncodeTokens / EncodeTokensWithIDs — built for exactly this,
// see their doc comments. Every row of every matrix must be L2-normalized
// (embed.L2Normalize); MaxSim treats a dot product as cosine similarity, the
// same unit-vector contract ann.Flat and embed.Encode use.
//
// # A reranker, not a first-stage retriever
//
// A document's token matrix is L tokens × D floats — roughly L× the footprint
// of its single pooled vector. Cheap to compute (EncodeTokens is one forward,
// no more expensive than Encode) but too large to hold for a whole corpus the
// way ann/bm25/sparse do. Index is sized for a SHORTLIST — the same role
// encoder.CrossEncoder plays in examples/rag — not a corpus-scale index. Get the
// shortlist from a dense+lexical fuse.RRF first, then rerank it here; see
// examples/colbert.
package late

import (
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/aikit/topk"
)

// Hit is one scored document, highest Score (MaxSim) first. The field name
// Index matches ann.Hit/sparse.Hit, so a MaxSim ranking composes with fuse.Keys
// the same way — though as a shortlist reranker it typically stands as the
// FINAL stage rather than one more leg to fuse (see encoder.CrossEncoder's
// identical role in examples/rag).
type Hit struct {
	Index int
	Score float64
}

// MaxSim scores query against doc: for every query token vector, its highest
// cosine similarity (a plain dot product — every row must be L2-normalized)
// against any document token vector, summed over query tokens.
//
// An empty query scores 0 (no terms to sum). An empty doc gives every query
// token a max of 0 (no candidate to match it against), not an error — the
// natural result of an empty inner-max loop, the same "degenerate input,
// defined output" posture as embed.L2Normalize's zero-vector case.
func MaxSim(query, doc [][]float32) float64 {
	// EIGHT doc tokens per kernel call (audit M-20). This is the one-vs-many
	// shape linalg.Dot8x4 exists for: the query strip is held in registers and
	// amortized across 8 document rows instead of re-streamed per row, which is
	// what a Dot-per-pair loop does. Flat and HNSW already score this way (item
	// 15, 2.05-2.82x there); late was still calling the single-row Dot for every
	// (query token x doc token) pair.
	//
	// NOT bit-identical to the per-pair Dot: the 8-row kernel accumulates in a
	// different order, so a score can move by ~1 float32 ULP — the same
	// reassociation tradeoff Flat and HNSW document. Here it can in principle
	// also change WHICH token wins a near-tie in the max, but the two scores are
	// within a ULP of each other by construction, so the sum moves by at most
	// that. TestMaxSim_batchedMatchesPerPair gates it.
	var sum float64
	var sums [32]float32
	for _, q := range query {
		d := len(q)
		n4 := d / 4
		tailStart := n4 * 4
		var best float32
		// seen replaces the old `j == 0` test: cosine can be negative, so best
		// must take the FIRST candidate unconditionally rather than being floored
		// at zero. With the rows visited in groups that index test no longer
		// identifies the first one.
		seen := false
		j := 0
		for ; d > 0 && j+8 <= len(doc); j += 8 {
			d0, d1, d2, d3 := doc[j], doc[j+1], doc[j+2], doc[j+3]
			d4, d5, d6, d7 := doc[j+4], doc[j+5], doc[j+6], doc[j+7]
			if len(d0) != d || len(d1) != d || len(d2) != d || len(d3) != d ||
				len(d4) != d || len(d5) != d || len(d6) != d || len(d7) != d {
				for k := range 8 { // ragged group — defensive, as Flat and HNSW do
					if s := linalg.Dot(q, doc[j+k]); !seen || s > best {
						best, seen = s, true
					}
				}
				continue
			}
			linalg.Dot8x4(&q[0], &d0[0], &d1[0], &d2[0], &d3[0], &d4[0], &d5[0], &d6[0], &d7[0], n4, &sums)
			group := [8][]float32{d0, d1, d2, d3, d4, d5, d6, d7}
			for k := range 8 {
				// Each row's dot is spread across its 4-lane block; sum the block,
				// then add the d%4 scalar tail — the same fold Flat uses.
				b := k * 4
				sc := sums[b] + sums[b+1] + sums[b+2] + sums[b+3]
				for kk := tailStart; kk < d; kk++ {
					sc += q[kk] * group[k][kk]
				}
				if !seen || sc > best {
					best, seen = sc, true
				}
			}
		}
		for ; j < len(doc); j++ {
			if s := linalg.Dot(q, doc[j]); !seen || s > best {
				best, seen = s, true
			}
		}
		sum += float64(best)
	}
	return sum
}

// ScoreBatch scores one query against many document token-matrices, returning
// MaxSim(query, docs[i]) per document in the caller's order. concurrency <= 0
// means NumCPU.
//
// Mirrors encoder.CrossEncoder.ScoreBatch's document-parallel work-stealing
// dispatch, minus that function's longest-first sort: a cross-encoder pays a
// full BERT forward per pair, cheap enough per-pair here (plain dot products,
// no forward pass) that a work-stealing split already keeps every core busy
// without needing to pre-balance the queue.
func ScoreBatch(query [][]float32, docs [][][]float32, concurrency int) []float64 {
	if len(docs) == 0 {
		return nil
	}
	if concurrency <= 0 {
		concurrency = runtime.NumCPU()
	}
	concurrency = min(concurrency, len(docs))

	out := make([]float64, len(docs))
	var next atomic.Int64
	var wg sync.WaitGroup
	for range concurrency {
		wg.Go(func() {
			for {
				i := int(next.Add(1)) - 1
				if i >= len(docs) {
					return
				}
				out[i] = MaxSim(query, docs[i])
			}
		})
	}
	wg.Wait()
	return out
}

// Index is a MaxSim reranker over a fixed shortlist of document token-matrices
// — see the package doc for why this is sized for a shortlist, not a corpus.
// Matrices are used by reference, not copied.
type Index struct {
	docs [][][]float32
}

// New builds an Index over docs — one token-matrix per document, each row
// L2-normalized (see the package doc).
func New(docs [][][]float32) *Index {
	return &Index{docs: docs}
}

// Len returns the number of documents in the index.
func (ix *Index) Len() int { return len(ix.docs) }

// Query scores query against every document (ScoreBatch, so document-parallel)
// and returns the k highest-scoring, best first. k <= 0 or an empty index
// returns nil.
func (ix *Index) Query(query [][]float32, k int) []Hit {
	if k <= 0 || len(ix.docs) == 0 {
		return nil
	}
	scores := ScoreBatch(query, ix.docs, 0)
	sel := topk.New[int](k)
	for i, s := range scores {
		sel.Push(i, s)
	}
	res := sel.Result()
	hits := make([]Hit, len(res))
	for i, r := range res {
		hits[i] = Hit{Index: r.Item, Score: r.Score}
	}
	return hits
}
