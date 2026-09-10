package bm25

import (
	"math"
	"slices"
	"sync"

	"github.com/townsendmerino/aikit/topk"
)

// WAND (Weak AND) dynamic pruning — perf-campaign item 39.
//
// scoreQuery's (query.go) scoring loop is term-at-a-time: every posting of every query
// term is accumulated, and only then is the top-k selected. That is O(postings)
// no matter how few documents can actually reach the top k. WAND is
// document-at-a-time and skips documents whose score CANNOT beat the current
// k-th best, using a per-term upper bound on what that term can contribute.
//
// The pivot rule, which is the whole algorithm: keep the cursors ordered by
// their current document, walk them accumulating upper bounds, and stop at the
// first cursor where the running sum exceeds the threshold θ. Every document
// strictly before that cursor's current document can only contain terms from
// the cursors already passed, whose bounds sum to at most θ — so none of them
// can be retained, and every cursor before the pivot may be advanced straight
// to the pivot document without scoring anything in between.
//
// EXACT, not approximate. Unlike ann.FlatBinary (item 38), this returns the
// same result as the full scan — same documents, same scores, same order, bit
// for bit. Two properties carry that, and both are gated:
//
//   - A skipped document is one whose upper bound cannot exceed θ, so it could
//     never have been retained. Pruning changes what is COMPUTED, never what is
//     selected.
//   - Candidates are evaluated in ascending document order, and each one's
//     contributions are summed in QUERY order — the same order the term-at-a-time
//     loop produces. Float addition is not associative, so summing in cursor
//     order (which is doc-ordered, not query-ordered) would give different
//     low bits on multi-term queries and silently break ties differently.
type wandCursor struct {
	post []posting
	i    int
	// doc caches post[i].doc, or sentinelDoc once exhausted. It is not a
	// convenience: the pivot loop and the ordering pass read the current
	// document of every cursor several times per iteration, and reading it
	// through a bounds-checked index plus an exhaustion test — originally
	// through a closure — cost more than the scoring it was protecting.
	doc int32
	idf float64
	ub  float64 // upper bound on this term's contribution to any document
}

// sentinelDoc orders an exhausted cursor after every real document, so the
// ordering pass needs no special case and the pivot walk terminates on it.
const sentinelDoc = int32(math.MaxInt32)

// setPos moves a cursor to position i and refreshes its cached document.
func (c *wandCursor) setPos(i int) {
	c.i = i
	if i < len(c.post) {
		c.doc = c.post[i].doc
	} else {
		c.doc = sentinelDoc
	}
}

// The per-term extrema this bound is built from — maxTf and minLen — live on
// termEntry and are tracked as Build goes (index.go). They used to be a
// termStat map filled by a second pass over every posting list; folding them in
// removed both the pass and a third map probe per query term.
//
// ubSlack widens each term's upper bound by a relative 1e-12.
//
// The bound is derived from monotonicity in exact arithmetic: the BM25 impact
// rises with tf and falls with the length normalization, so evaluating it at
// (maxTf, minNorm) dominates every posting. Evaluated in float64, both the
// bound and the score it is compared against carry rounding of order one ulp
// (~2.2e-16 relative), so an adversarial term could in principle produce a
// bound a hair BELOW the true maximum and prune a document that belonged in the
// results.
//
// 1e-12 is roughly 4500 ulps — far past any rounding of a four-operation
// expression, and far too small to cost pruning power, since the bound is loose
// by construction anyway (maxTf and minNorm rarely occur in the same document).
// The alternative, reasoning about the exact rounding of each operation, buys
// nothing here.
const ubSlack = 1 + 1e-12

// maxWandTerms is the largest number of distinct query terms for which dynamic
// pruning is used; longer queries take the exhaustive scan.
//
// The reason is structural, not a tuning accident. The pivot loop re-orders and
// re-walks EVERY cursor on every iteration, so its cost per iteration grows
// linearly with the query length, while the exhaustive scan costs a constant
// per posting. Measured on a 200k-document index with mixed-selectivity queries
// (BenchmarkWANDQueryLength), pruning is worth 3.98x at 2 terms, 2.83x at 3,
// 1.25x at 6, and then LOSES: 35% slower at 12 terms and 65% slower at 24.
//
// 8 sits just past the last measured win, so the guard never blocks a shape
// that pays. The crossover itself is data-dependent — it moves with how much
// pruning the query's selectivity mix allows — so this is a measured constant,
// not a derived one, and it is the amd64 number.
//
// The long-query case looked like a different algorithm — MaxScore, which
// partitions terms into essential and non-essential by cumulative bound and never
// advances the non-essential cursors. It was BUILT, MEASURED, and REVERTED
// (2026-07-31): exact (== topKExhaustive bit-for-bit over 1000 random long queries,
// mutation-checked) but 2.5–5.7× SLOWER at 12–48 terms. Two structural reasons it
// loses here, and they are not implementation bugs: topKExhaustive is a TAAT
// accumulator — it walks each term's postings ONCE into a scores[] array, O(total
// postings) and cache-friendly — whereas MaxScore is document-at-a-time, whose
// per-document overhead only repays when pruning skips many documents; and on the
// uniform-selectivity queries the benchmarks generate, θ rises slowly so the
// essential set stays large and little is pruned. MaxScore's real edge is a SKEWED
// impact distribution (a few dominant terms lift θ fast) — i.e. trained SPLADE
// weights — which nothing here measures, so shipping it would regress the general
// long-query case for an unmeasured gain. Left to the exhaustive accumulator.
const maxWandTerms = 8

// wandUsable reports whether the pruning bound is sound for the current
// parameters. The bound relies on the impact being monotone in tf and in the
// normalization, and on the denominator tf + K1*(1-B+B*norm) keeping one sign
// across the posting list. That holds for K1 >= 0 and 0 <= B <= 1 — the
// standard BM25 range. A caller outside it gets the exhaustive path instead of
// a wrong answer.
//
// B > 1 is the case that is easy to miss (audit C-03). The bound is evaluated
// at minNorm, and since d/dnorm of (1-B+B*norm) is B > 0, minNorm does minimise
// the denominator — so the formula looks right. What breaks is that for B > 1
// the term 1-B is negative, so across a corpus with skewed document lengths the
// denominator can cross ZERO inside the list's norm range: the true impact is
// unbounded near the crossing and negative past it, while the value computed at
// minNorm is small or negative and bounds nothing. Measured before this
// condition existed: a corpus of 200 long docs plus 20 two-token docs sharing a
// term, K1=1.5 B=2.0, made TopK return 0 results where the exhaustive scan
// returns 10 — silent, total loss, not a reordering.
func (ix *Index) wandUsable() bool { return ix.K1 >= 0 && ix.B >= 0 && ix.B <= 1 }

// topKWAND is TopK's pruning implementation. It returns ok=false when it
// declines, and the caller falls back to the exhaustive scan.
func (ix *Index) topKWAND(query []string, k int) ([]Result, bool) {
	w := wandPool.Get().(*wandState)
	defer wandPool.Put(w)
	return ix.topKWANDState(query, k, w)
}

// topKWANDState is topKWAND with the scratch supplied by the caller, so a test
// can read back how many documents were actually evaluated (w.evaluated) — the
// property TestWAND_actuallyPrunes needs.
//
// The cursor positions are NOT that measure, which cost a wrong conclusion
// once: a skip advances the cursor index PAST the postings it declined to read,
// so a fully-pruning query leaves its cursor at the end of the list looking
// exactly like an exhaustive walk. What is saved is documents evaluated, and
// only a counter sees it.
func (ix *Index) topKWANDState(query []string, k int, w *wandState) ([]Result, bool) {
	if !ix.wandUsable() {
		return nil, false
	}
	w.evaluated = 0
	cur := w.cur[:0]

	k1, bb := ix.K1, ix.B
	k1p1 := k1 + 1
	for i, term := range query {
		if dupTerm(query, i) {
			continue // one contribution per distinct term, exactly as scoreQuery
		}
		e := ix.entry(term)
		if e == nil || len(e.postings) == 0 {
			continue
		}
		idf := ix.idfOf(e)
		if idf == 0 {
			continue
		}
		pl := e.postings
		tf := float64(e.maxTf)
		ub := idf * (tf * k1p1) / (tf + k1*(1-bb+bb*ix.minNorm(e)))
		cur = append(cur, wandCursor{post: pl, doc: pl[0].doc, idf: idf, ub: ub * ubSlack})
	}
	w.cur = cur
	if len(cur) == 0 {
		return []Result{}, true
	}
	if len(cur) > maxWandTerms {
		return nil, false
	}

	ord := w.ord[:0]
	for i := range cur {
		ord = append(ord, int32(i))
	}
	w.ord = ord

	sel := topk.New[int](k)
	// theta is the score a candidate must strictly exceed. It starts at 0 rather
	// than -Inf because TopK's contract is "Score > 0" — which also means the
	// zero-score documents the exhaustive path filters out are never even
	// visited here.
	theta := 0.0

	for {
		// Order the cursors by current document. Insertion sort: the list is as
		// long as the query and nearly sorted every iteration after the first,
		// since only the cursors that just moved are out of place.
		for a := 1; a < len(ord); a++ {
			v := ord[a]
			d := cur[v].doc
			b := a - 1
			for b >= 0 && cur[ord[b]].doc > d {
				ord[b+1] = ord[b]
				b--
			}
			ord[b+1] = v
		}

		// Pivot: the first cursor at which the accumulated upper bound can beat
		// theta. Exhausted cursors sort last and terminate the walk — they can
		// contribute nothing, so their bounds must not enter the sum.
		sum, piv := 0.0, -1
		for j := range ord {
			c := &cur[ord[j]]
			if c.doc == sentinelDoc {
				break
			}
			sum += c.ub
			if sum > theta {
				piv = j
				break
			}
		}
		if piv < 0 {
			break // no remaining document can beat theta
		}
		pivotDoc := cur[ord[piv]].doc

		if cur[ord[0]].doc != pivotDoc {
			// Not every cursor has reached the pivot document. Everything
			// strictly before it is unreachable — its terms' bounds sum to at
			// most theta — so every cursor ahead of the pivot skips straight
			// there.
			for j := range piv {
				c := &cur[ord[j]]
				c.setPos(advanceTo(c.post, c.i, pivotDoc))
			}
			continue
		}

		// A real candidate. Sum in QUERY order — see the type comment.
		w.evaluated++
		score := 0.0
		norm := ix.norm[pivotDoc]
		for t := range cur {
			c := &cur[t]
			if c.doc == pivotDoc {
				tf := float64(c.post[c.i].tf)
				score += c.idf * (tf * k1p1) / (tf + k1*(1-bb+bb*norm))
			}
		}
		if score > theta {
			sel.Push(int(pivotDoc), score)
			if th := sel.Threshold(); th > theta {
				theta = th
			}
		}
		for t := range cur {
			if c := &cur[t]; c.doc == pivotDoc {
				c.setPos(c.i + 1)
			}
		}
	}

	items := sel.Result()
	slices.SortFunc(items, topk.ItemCmp[int])
	out := make([]Result, len(items))
	for j, s := range items {
		out[j] = Result{Doc: s.Item, Score: s.Score}
	}
	return out, true
}

// advanceTo returns the first index >= i whose document is >= target.
//
// Three regimes, in the order they are hit. A short LINEAR probe first, because
// most skips land within a few postings and a sequential read is what the
// hardware prefetcher is already doing; then galloping to bracket a long skip;
// then binary search inside the bracket.
//
// The binary searches are written out rather than calling sort.Search. That is
// not micro-optimizing for its own sake: sort.Search takes a closure, so every
// probe of every skip is an indirect call, and the first version of this
// function made pruning SLOWER than the exhaustive scan it replaced despite
// evaluating 21x fewer documents.
func advanceTo(post []posting, i int, target int32) int {
	n := len(post)
	// Linear probe. 8 covers the overwhelming majority of skips on a dense
	// posting list, where the pivot is only a document or two ahead.
	for lim := min(i+8, n); i < lim; i++ {
		if post[i].doc >= target {
			return i
		}
	}
	if i >= n {
		return n
	}
	// The linear probe ended by exhausting its budget, not by testing this
	// position, so it still has to be checked — otherwise the gallop below
	// starts from a `lo` that already satisfies the target and overshoots it.
	if post[i].doc >= target {
		return i
	}
	// Gallop to bracket the target, then bisect. lo is known to be below target.
	lo, step := i, 1
	hi := n
	for {
		p := lo + step
		if p >= n {
			break
		}
		if post[p].doc >= target {
			hi = p
			break
		}
		lo = p
		step *= 2
	}
	for lo+1 < hi {
		mid := int(uint(lo+hi) >> 1)
		if post[mid].doc >= target {
			hi = mid
		} else {
			lo = mid
		}
	}
	return hi
}

// wandState is one query's cursor scratch, pooled: a query allocates nothing
// beyond its result slice.
type wandState struct {
	cur []wandCursor
	ord []int32
	// evaluated counts the documents fully scored — one increment per candidate,
	// against a candidate's own handful of float operations and a random read of
	// ix.norm. It is what makes "did pruning actually happen" testable.
	evaluated int
}

var wandPool = sync.Pool{New: func() any { return new(wandState) }}
