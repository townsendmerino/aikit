package bm25

import (
	"math"
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

	// mmap is the mapping a LoadIndexMmap index's postings alias (nil otherwise);
	// closed is set by Close.
	mmap   []byte
	closed bool
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
// off its own prior baseline in the 2026 perf campaign). It needs the whole
// tokenized corpus at once, which is most of a build's peak memory (about 5×
// the finished index); Builder takes the documents one at a time instead, and
// gives the same index. To keep an index across processes, save it with
// WriteTo and load it with ReadIndex or LoadIndexMmap.
func Build(docs [][]string) *Index {
	// posting packs doc and tf into int32s (item 29). Both bounds are far beyond
	// any real corpus, but a silent truncation here would corrupt the index, so
	// it panics rather than wraps.
	if len(docs) > math.MaxInt32 {
		panic("bm25: corpus exceeds 2^31-1 documents")
	}
	b := newBuilder(len(docs))
	for _, toks := range docs {
		b.Add(toks)
	}
	return b.Build()
}

// setLengths computes avgdl and the per-document length normalization from
// docLen.
func (ix *Index) setLengths() {
	var total int
	for _, l := range ix.docLen {
		total += l
	}
	if len(ix.docLen) > 0 {
		ix.avgdl = float64(total) / float64(len(ix.docLen))
	}
	// Precompute the length normalization per document (item 10). The expression
	// is character-for-character the one the scoring loop used, so every score is
	// bit-identical; the loop just stops dividing once per posting.
	ix.norm = make([]float64, len(ix.docLen))
	if ix.avgdl > 0 {
		for d := range ix.norm {
			ix.norm[d] = float64(ix.docLen[d]) / ix.avgdl
		}
	}
}

// N is the number of indexed documents.
func (ix *Index) N() int { return len(ix.docLen) }
