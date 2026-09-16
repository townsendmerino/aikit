// Package accum is a pooled scoring accumulator shared by bm25 and sparse: both
// packages score a query by walking postings and accumulating per-document
// contributions, then selecting over only the documents actually touched — this
// makes a query cost O(postings touched) instead of O(corpus). Extracted from
// what was bm25/accum.go and sparse/accum.go, near-byte-identical copies with
// nothing structurally guaranteeing they stayed in sync (perf-campaign-2026-07-28.md
// finding A / item 11 landed it in bm25 first as item 11/44; sparse item 39 caught
// the same fix up after it drifted).
//
// The old shape allocated `make([]float64, N)` per query (a fresh memclr of the
// whole span on the large-object path), wrote a handful of slots, then RE-READ
// all N for selection. Measured on bm25 at 200k docs with a 3-term query touching
// 2,335 postings: ~65% allocate+zero, ~35% selection sweep, and under 1% actual
// scoring — the query itself was lost in the noise. On sparse, a 30-term SPLADE
// query over 22,185 postings cost 654 µs, almost none of it scoring.
//
// The dense array stays (random-access `+=` beats a map), but it is pooled and
// only the touched slots are cleared, and selection runs over the touched set.
//
// WHY GENERATION STAMPS rather than "Scores[doc] != 0 means touched": a caller's
// contribution can legitimately be zero (sparse: a zero-weight posting, or
// contributions that cancel; bm25: not today, since the Lucene idf is > 0 for any
// known term, tf >= 1, denom > 0 — but a zero test would break SILENTLY the day a
// scorer admits one, and the doc would vanish from results rather than score 0). A
// stamp costs 4 bytes per doc and cannot be wrong for either caller.
package accum

import (
	"slices"
	"sync"
)

// Accum is a pooled scoring accumulator. Scores and Touched are exported for
// callers to read after OrderTouched; everything else is bookkeeping.
type Accum struct {
	Scores  []float64
	Touched []int32
	stamp   []uint32
	gen     uint32
	// runs holds the start offset in Touched of each contributing term's
	// first-touch run, plus a terminating len(Touched). Each run is ascending
	// by construction (a caller appends postings in document order and Add
	// records a doc on first touch only), so ordering Touched is a MERGE of
	// sorted runs, not a sort — see OrderTouched.
	runs  []int32
	merge []int32 // scratch destination for the merge
}

var pool = sync.Pool{New: func() any { return new(Accum) }}

// Get returns an accumulator sized for n docs, with a fresh generation. Clearing
// is O(1): bumping gen invalidates every stamp at once.
func Get(n int) *Accum {
	a := pool.Get().(*Accum)
	if cap(a.Scores) < n {
		a.Scores = make([]float64, n)
		a.stamp = make([]uint32, n)
		a.gen = 0
	}
	a.Scores = a.Scores[:n]
	a.stamp = a.stamp[:n]
	a.Touched = a.Touched[:0]
	a.runs = a.runs[:0]
	a.gen++
	if a.gen == 0 { // wrapped: the only case where stamps must actually be cleared
		clear(a.stamp)
		a.gen = 1
	}
	return a
}

// Put returns a to the pool.
func Put(a *Accum) { pool.Put(a) }

// Add applies one posting's contribution, recording the doc on its first touch.
func (a *Accum) Add(doc int32, v float64) {
	if a.stamp[doc] != a.gen {
		a.stamp[doc] = a.gen
		a.Scores[doc] = 0
		a.Touched = append(a.Touched, doc)
	}
	a.Scores[doc] += v
}

// BeginRun marks the start of a new term's postings in Touched.
func (a *Accum) BeginRun() { a.runs = append(a.runs, int32(len(a.Touched))) }

// OrderTouched puts the touched ids in ascending doc order.
//
// This is not cosmetic. topk's selector keeps the FIRST-SEEN item of a tie (a
// tied newcomer is rejected at capacity), so which member of a tie survives
// depends on push order. Both callers push in ascending doc order when they
// range over a dense array directly, so reproducing that order here reproduces
// the identical result set, not merely an equally-valid one.
//
// It used to be a full slices.Sort — O(T log T), and on bm25 measured at ~18%
// of a common 200k-doc query, on sparse the single largest cost of a 30-term
// query touching ~9,200 documents. But Touched is never arbitrary: each term
// appends its first-touch docs in ascending order, so the slice is a
// concatenation of Q ascending runs for a Q-term query. Merging them is
// O(T log Q), and for the single-term case — one run, already sorted — it is
// free.
//
// The output is byte-for-byte what slices.Sort produced: both yield the
// ascending permutation, and the ids are distinct (a doc enters Touched exactly
// once), so there is no tie for a merge to order differently.
func (a *Accum) OrderTouched() {
	// Close the final run.
	a.runs = append(a.runs, int32(len(a.Touched)))
	// Drop empty runs (a term whose postings were all already touched).
	dst := a.runs[:0]
	for i := 0; i+1 < len(a.runs); i++ {
		if a.runs[i] < a.runs[i+1] {
			dst = append(dst, a.runs[i])
		}
	}
	dst = append(dst, int32(len(a.Touched)))
	a.runs = dst

	nRuns := len(a.runs) - 1
	if nRuns <= 1 {
		return // one run (or none): already ascending
	}
	if nRuns == 2 {
		a.merge = mergeTwo(a.merge[:0], a.Touched[a.runs[0]:a.runs[1]], a.Touched[a.runs[1]:a.runs[2]])
		a.Touched, a.merge = a.merge, a.Touched
		return
	}
	// Few enough terms that pairwise merging beats a heap, and Q is bounded by
	// the query length in practice.
	for nRuns > 1 {
		a.merge = a.merge[:0]
		out := a.runs[:0]
		i := 0
		for ; i+1 < nRuns; i += 2 {
			out = append(out, int32(len(a.merge)))
			a.merge = mergeTwo(a.merge, a.Touched[a.runs[i]:a.runs[i+1]], a.Touched[a.runs[i+1]:a.runs[i+2]])
		}
		if i < nRuns { // odd run out: carry it forward unchanged
			out = append(out, int32(len(a.merge)))
			a.merge = append(a.merge, a.Touched[a.runs[i]:a.runs[i+1]]...)
		}
		out = append(out, int32(len(a.merge)))
		a.Touched, a.merge = a.merge, a.Touched
		a.runs = out
		nRuns = len(a.runs) - 1
	}
}

// mergeTwo appends the ascending merge of two ascending slices to dst.
func mergeTwo(dst, x, y []int32) []int32 {
	nx, ny := len(x), len(y)
	if nx == 0 {
		return append(dst, y...)
	}
	if ny == 0 {
		return append(dst, x...)
	}
	orig := len(dst)
	target := orig + nx + ny
	dst = slices.Grow(dst, nx+ny)[:target]
	out := dst[orig:target:target]

	i, j, k := 0, 0, 0
	for i < nx && j < ny {
		xi, yj := x[i], y[j]
		if xi <= yj {
			out[k] = xi
			i++
		} else {
			out[k] = yj
			j++
		}
		k++
	}
	k += copy(out[k:], x[i:])
	copy(out[k:], y[j:])
	return dst
}
