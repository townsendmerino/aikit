package bm25

import (
	"math"
	"strings"
	"unsafe"
)

// BM25 defaults. These are the Lucene / bm25s defaults; ken's DESIGN.md
// Stage 1 validates ranking against semble's SearchMode.BM25, so the exact
// variant is pinned to what bm25s uses by default (the Lucene IDF, see
// query.go).
const (
	DefaultK1 = 1.5
	DefaultB  = 0.75
)

// posting is deliberately 8 bytes, not 16 (perf-campaign item 29). The scoring
// scan is a linear walk over posting lists and is memory-bound, so the struct
// width IS the scan cost: int/int made it 16 B where sparse's equivalent has
// always been 8. int32 is ample — Build rejects a corpus that would overflow it.
type posting struct {
	doc int32
	tf  int32
}

// sizeofPosting is asserted by TestPostingWidth; see posting's comment.
const sizeofPosting = unsafe.Sizeof(posting{})

// Index is an immutable BM25 inverted index over a fixed document set.
// Documents are referenced by their position in the slice passed to Build.
type Index struct {
	K1, B  float64
	docLen []int
	// norm[d] is docLen[d]/avgdl, precomputed at Build. The scoring loop used to
	// divide per POSTING; this makes it a load (item 10). Computed with exactly
	// the expression the loop used, so scores are bit-identical.
	norm  []float64
	avgdl float64
	// terms maps a term to its index in `entries`. ONE map, where there used to
	// be three (postings, df, and item 39's stats).
	//
	// `m[k] = append(m[k], v)` is a mapaccess PLUS a mapassign — two independent
	// hashes of the same key — and `df[k]++` was a third and fourth in a second
	// map, and the intern check a fifth. That is five hashes of the same string
	// per (document, term) where one will do, and Build does it 23.9 M times on
	// the campaign's 200k-document corpus (lens doc 3.7).
	//
	// The indirection is through a slice rather than a map of pointers so that a
	// corpus with 30k distinct terms does not also allocate 30k *termEntry.
	terms   map[string]int32
	entries []termEntry
	// postings is every term's posting list, one after another: entries[i]'s list is
	// postings[e.off : e.off+e.df], doc-ordered. One flat arena in place of a slice per
	// term (memory campaign item 3), which spent a 24 B header and ~32% append slack on
	// each of them; it is also the layout the saved index maps from disk as one block.
	postings []posting
}

// termEntry is everything the index knows about one term. Grouping them is the
// point: a single map probe now reaches the posting list, the document
// frequency, and the extrema WAND's upper bound is built from.
type termEntry struct {
	off int64 // start of this term's list in Index.postings
	df  int32 // the list's length: the number of documents containing the term
	// maxTf and minLen are the parameter-free extrema item 39's bound is
	// reconstructed from — see wand.go's termStat, which this absorbed. They are
	// tracked here as Build goes rather than in a second pass over every posting
	// list afterwards, which is where they used to come from.
	maxTf  int32
	minLen int32
}

// list returns e's posting list.
func (ix *Index) list(e *termEntry) []posting {
	return ix.postings[e.off : e.off+int64(e.df) : e.off+int64(e.df)]
}

// entry returns the term's data, or nil if the term is not indexed.
func (ix *Index) entry(term string) *termEntry {
	i, ok := ix.terms[term]
	if !ok {
		return nil
	}
	return &ix.entries[i]
}

// minNorm is the smallest length normalization over the documents containing
// this term — the value item 39's bound needs.
//
// Derived from minLen rather than stored, and it is exactly ix.norm[d] for that
// document: norm is float64(docLen[d])/avgdl, monotone in docLen for a positive
// avgdl, so the minimum over the term's documents is reached at the shortest
// one and the expression here is character-for-character the same.
func (ix *Index) minNorm(e *termEntry) float64 {
	if ix.avgdl == 0 {
		return 0 // every document empty ⇒ norm is all zeros
	}
	return float64(e.minLen) / ix.avgdl
}

// Build constructs the index from already-tokenized documents (use
// Tokenize). docs[i] is document i's token stream; empty docs are allowed
// and simply score zero.
//
// Build is O(corpus) — one pass over the tokens — and fast (measured ~1.27–1.30×
// off its own prior baseline in the 2026 perf campaign). That speed is a design
// note, not just a datum: there is deliberately no serialized-Index format. For a
// package whose build is O(corpus) and cheap, rebuilding from the already-embedded
// corpus is often the honest answer to "how do I persist this?" — a versioned
// on-disk format is a permanent compatibility promise (every future Index field
// carried, defaulted, and gated for old files) that the build time alone does not
// justify. If you need persistence, decide it as an API/format question, not as a
// perf optimization. See docs/internal/roadmap.md (N4/N6).
func Build(docs [][]string) *Index {
	// posting packs doc and tf into int32s (item 29). Both bounds are far beyond
	// any real corpus, but a silent truncation here would corrupt the index, so
	// it panics rather than wraps.
	if len(docs) > math.MaxInt32 {
		panic("bm25: corpus exceeds 2^31-1 documents")
	}
	ix := &Index{
		K1:     DefaultK1,
		B:      DefaultB,
		docLen: make([]int, len(docs)),
		terms:  make(map[string]int32),
	}
	var total int
	// One term-frequency map reused across all documents (clear per doc) instead of
	// a fresh make per document — at ~378k chunks that churned multiple GB of
	// short-lived map storage and 378k map-header allocations (audit #20). Output
	// is identical: each doc appends exactly one posting per term and docs are
	// processed in order, so postings[term] is doc-ordered regardless of the inner
	// map-iteration order.
	tf := make(map[string]int, 512)
	// Each term's postings collect in their own list while documents arrive (a term's
	// count is not known until the last one), then move into the one arena below.
	var lists [][]posting
	nPostings := 0
	for d, toks := range docs {
		ix.docLen[d] = len(toks)
		total += len(toks)
		clear(tf)
		for _, t := range toks {
			tf[t]++
		}
		for term, f := range tf {
			// Intern the retained key (task-perf-memoization §1b). toks[i] is a view
			// into the caller's document text / the tokenizer's per-call arena, so using
			// it directly as a map key would pin that whole backing array for the Index's
			// lifetime — a 356-file index retained 4.79 MB while its distinct terms were
			// 0.11 MB. On a term's FIRST occurrence we store a compact strings.Clone; every
			// later occurrence matches that existing key (Go leaves the stored key header
			// untouched on value update), so no view is ever retained. Byte-identical keys
			// ⇒ identical scores. Single-threaded here, so no lock — unlike a Tokenize-side
			// interner, this never touches the concurrent hot path.
			i, seen := ix.terms[term]
			if !seen {
				i = int32(len(ix.entries))
				ix.terms[strings.Clone(term)] = i
				ix.entries = append(ix.entries, termEntry{minLen: math.MaxInt32})
				lists = append(lists, nil)
			}
			// Indexed rather than held as a pointer: `ix.entries` may reallocate
			// on the append above, and a pointer taken before it would dangle.
			e := &ix.entries[i]
			lists[i] = append(lists[i], posting{doc: int32(d), tf: int32(f)})
			e.df++
			nPostings++
			if tfi := int32(f); tfi > e.maxTf {
				e.maxTf = tfi
			}
			if l := int32(len(toks)); l < e.minLen {
				e.minLen = l
			}
		}
	}
	// Compact into the arena: each term's list, in term order, dropped as it is copied.
	ix.postings = make([]posting, nPostings)
	var off int64
	for i := range ix.entries {
		ix.entries[i].off = off
		off += int64(copy(ix.postings[off:], lists[i]))
		lists[i] = nil
	}
	if len(docs) > 0 {
		ix.avgdl = float64(total) / float64(len(docs))
	}
	// Precompute the length normalization per document (item 10). The expression
	// is character-for-character the one the scoring loop used, so every score is
	// bit-identical; the loop just stops dividing once per posting.
	ix.norm = make([]float64, len(docs))
	if ix.avgdl > 0 {
		for d := range ix.norm {
			ix.norm[d] = float64(ix.docLen[d]) / ix.avgdl
		}
	}
	return ix
}

// N is the number of indexed documents.
func (ix *Index) N() int { return len(ix.docLen) }
