package bm25

import (
	"math"
	"strings"
)

// Builder builds an Index one document at a time, so a build holds the index
// under construction plus one document rather than the whole tokenized corpus
// (Build's input is about 5× the finished index). Documents are numbered in
// the order they are added, from 0. The result is identical to Build over the
// same documents in the same order; Build is this loop.
//
//	b := bm25.NewBuilder()
//	for _, text := range corpus {
//		b.Add(bm25.Tokenize(text))
//	}
//	ix := b.Build()
//
// A Builder is not safe for concurrent use. After Build it must not be used
// again.
type Builder struct {
	ix *Index
	// One term-frequency map reused across all documents (clear per doc) instead of
	// a fresh make per document — at ~378k chunks that churned multiple GB of
	// short-lived map storage and 378k map-header allocations (audit #20). Output
	// is identical: each doc appends exactly one posting per term and docs are
	// processed in order, so a term's postings are doc-ordered regardless of the
	// inner map-iteration order.
	tf map[string]int
	// Each term's postings collect in their own list while documents arrive (a
	// term's count is not known until the last one), then move into the Index's
	// one arena at Build.
	lists     [][]posting
	nPostings int
	built     bool
}

// NewBuilder returns an empty Builder with the default K1 and B.
func NewBuilder() *Builder { return newBuilder(0) }

// newBuilder is NewBuilder with room for n documents' lengths (Build knows n).
func newBuilder(n int) *Builder {
	return &Builder{
		ix: &Index{
			K1:     DefaultK1,
			B:      DefaultB,
			docLen: make([]int, 0, n),
			terms:  make(map[string]int32),
		},
		tf: make(map[string]int, 512),
	}
}

// Add indexes tokens as the next document. An empty document is allowed and
// simply scores zero. tokens is not retained.
func (b *Builder) Add(tokens []string) {
	if b.built {
		panic("bm25: Builder.Add after Build")
	}
	ix := b.ix
	d := len(ix.docLen)
	if d == math.MaxInt32 {
		panic("bm25: corpus exceeds 2^31-1 documents")
	}
	ix.docLen = append(ix.docLen, len(tokens))
	clear(b.tf)
	for _, t := range tokens {
		b.tf[t]++
	}
	for term, f := range b.tf {
		// Intern the retained key (task-perf-memoization §1b). tokens[i] is a view
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
			b.lists = append(b.lists, nil)
		}
		// Indexed rather than held as a pointer: `ix.entries` may reallocate
		// on the append above, and a pointer taken before it would dangle.
		e := &ix.entries[i]
		b.lists[i] = append(b.lists[i], posting{doc: int32(d), tf: int32(f)})
		e.df++
		b.nPostings++
		if tfi := int32(f); tfi > e.maxTf {
			e.maxTf = tfi
		}
		if l := int32(len(tokens)); l < e.minLen {
			e.minLen = l
		}
	}
}

// N is the number of documents added so far.
func (b *Builder) N() int { return len(b.ix.docLen) }

// Build finishes the index. The Builder must not be used afterwards.
func (b *Builder) Build() *Index {
	if b.built {
		panic("bm25: Builder.Build called twice")
	}
	b.built = true
	ix := b.ix
	// Compact into the arena: each term's list, in term order, dropped as it is
	// copied.
	ix.postings = make([]posting, b.nPostings)
	var off int64
	for i := range ix.entries {
		ix.entries[i].off = off
		off += int64(copy(ix.postings[off:], b.lists[i]))
		b.lists[i] = nil
	}
	b.lists, b.tf, b.ix = nil, nil, nil
	ix.setLengths()
	return ix
}
