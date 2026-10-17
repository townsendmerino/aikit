package ann

import (
	"slices"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/aikit/topk"
)

// Neighbor selection: trimming a candidate set down to at most m edges, at both
// insert time (Add's back-edge loop, via prune) and build time proper (Add's
// per-layer link step, via selectNeighbors). Two algorithms — the paper's
// Algorithm 3 (selectNearest, plain M-nearest) and Algorithm 4 (selectHeuristic,
// the diversity heuristic that is HNSW.go's package doc and Config.SimpleNeighbors
// both describe) — plus the concrete binary heap searchLayer and the selection
// code both build candidate sets on top of.
//
// Split out of hnsw.go (which still holds the struct, construction, and search)
// the same way hnsw_persist.go splits out serialization: one file per concern in
// an 800+ line type, so a reader after "how are neighbors chosen" or "what heap
// backs search" does not have to first skim past Add/Query/searchLayer to find it.

// simIDsInto fills dst[i] with simIDs(base, ids[i]).
//
// Same 8-row batching as scoreInto, for the BUILD path: selectHeuristic compares
// one candidate against every already-selected neighbour and prune scores one
// node against all of its neighbours, both of which are one-vector-vs-many —
// exactly Dot8x4's shape. simIDs was 50% of build CPU in single-row dots
// (perf-campaign item 17).
//
// Like scoreInto this is not bit-identical to per-pair Dot, and here the
// consequence is larger: a moved score can flip selectHeuristic's `> e.sim`
// test and change the GRAPH, not just one query's traversal. The gate is
// therefore recall, not equality — see TestHNSW_batchedBuildKeepsRecall.
func (h *HNSW) simIDsInto(base int, ids []int, dst []float64) []float64 {
	dst = dst[:0]
	if h.scoreUnbatched || h.int8 || len(h.vecs) == 0 {
		for _, id := range ids {
			dst = append(dst, h.simIDs(base, id))
		}
		return dst
	}
	return h.scoreInto(queryVec{f32: h.vecs[base]}, ids, dst)
}

// prune trims node id's neighbor list at layer to the mmax most similar.
// bIDs2 returns a length-n slice of the secondary id scratch.
func (h *HNSW) bIDs2(n int) []int {
	if cap(h.bOut2) < n {
		h.bOut2 = make([]int, n)
	}
	h.bOut2 = h.bOut2[:n]
	return h.bOut2
}

func (h *HNSW) prune(id, layer int) {
	nbrs := h.nodes[id].nbrs[layer]
	ids := h.bIDs2(len(nbrs))
	for i, n := range nbrs {
		ids[i] = int(n)
	}
	h.bSims = h.simIDsInto(id, ids, h.bSims) // node-node, both modes
	cands := h.bCand[:0]
	for i, n := range ids {
		cands = append(cands, cand{id: n, sim: h.bSims[i]})
	}
	h.bCand = cands
	kept := h.selectNeighbors(cands, h.mmax(layer))
	// Rewrite the list in place (memory campaign item 4): nbrs has been read
	// into ids above and is not read again, and len(kept) ≤ len(nbrs). A fresh
	// exact-size slice here was an allocation per prune, and the next back-link
	// append then had to regrow it — two allocations per pruned neighbor.
	out := nbrs[:len(kept)]
	for i, c := range kept {
		out[i] = int32(c.id)
	}
	h.nodes[id].nbrs[layer] = out
}

// selectNearest returns the m highest-similarity candidates from w
// (which searchLayer already returns sorted desc, but prune passes an
// unsorted slice, so re-rank via a K-selector for O(N log m)).
// selectNeighbors picks up to m edges from the candidate set w, dispatching to
// the diversity heuristic (Algorithm 4) unless Config.SimpleNeighbors is set, else the
// plain M-nearest (Algorithm 3). Both return the result descending by sim, so the
// caller's ep = result[0] handoff is unaffected.
func (h *HNSW) selectNeighbors(w []cand, m int) []cand {
	if h.heuristic {
		return h.selectHeuristic(w, m)
	}
	return selectNearest(w, m)
}

// simIDs is the cosine similarity (dot product on unit vectors) between two
// indexed vectors — the heuristic's candidate-vs-candidate comparison.
//
// ⚠ Its result is compared directly against sim's in selectHeuristic. See the
// matching-units invariant on sim: the two must stay on the same scale, and
// changing either one alone silently produces a different graph.
func (h *HNSW) simIDs(a, b int) float64 {
	if h.int8 {
		return float64(linalg.DotI8(h.code(a), h.code(b))) * float64(h.scales[a]) * float64(h.scales[b])
	}
	return float64(linalg.Dot(h.vecs[a], h.vecs[b]))
}

// selectHeuristic is the HNSW neighbor-diversity heuristic (paper Algorithm 4):
// processing candidates nearest-first, it keeps one only if it is closer to the
// base element than to every neighbor already chosen — so an edge is dropped when
// a closer, already-selected neighbor lies in the same direction. The kept edges
// fan out across directions instead of piling into one cluster, which is what
// gives long-range connectivity (and recall-per-ef) on clustered data, where
// plain M-nearest (selectNearest) links a node to a tight cluster of near-clones
// and never reaches the rest of the graph. keepPrunedConnections then tops the
// result back up to m from the discards so node degree is preserved.
//
// Cost: O(|w|·m) similarity computations vs selectNearest's heap — heavier, paid
// once at build time for the recall win.
//
// ALIASING CONTRACT: the returned slice points into build scratch and is valid
// only until the next selectHeuristic call — including one made INDIRECTLY. Both
// callers copy the ids into a fresh []int32 immediately and then use that copy;
// Add in particular must, because its back-edge loop calls prune, which re-enters
// here. Getting this wrong does not crash — it silently rewrites the slice being
// iterated and degrades the graph (recall@10 1.00 → 0.83, caught only by the
// recall gates). A future caller that keeps the result must copy it.
func (h *HNSW) selectHeuristic(w []cand, m int) []cand {
	h.bSort = sortCandsDescInto(h.bSort, w)
	cands := h.bSort
	if len(cands) <= m {
		return cands
	}
	r := h.bKeep[:0]
	discarded := h.bDisc[:0]
	rIDs := h.bIDs[:0]
	for _, e := range cands {
		if len(r) >= m {
			break
		}
		// e is closer to an already-selected neighbor than to the base ⇒
		// redundant (same direction), discard it. Scored EIGHT selected
		// neighbours at a time, with the early exit kept between groups so a
		// candidate rejected by its first comparison still costs at most one
		// group instead of the whole of r (item 17).
		// `s` here is a node–node similarity and `e.sim` is a node–query one.
		// They are comparable only under the matching-units invariant documented
		// on sim — in Add the query and the base node are quantized from the same
		// row, so the two scales are equal. This is THE line that invariant
		// exists for.
		keep := true
		for j := 0; j < len(rIDs) && keep; j += 8 {
			end := min(j+8, len(rIDs))
			h.bSims = h.simIDsInto(e.id, rIDs[j:end], h.bSims)
			for _, s := range h.bSims {
				if s > e.sim {
					keep = false
					break
				}
			}
		}
		if keep {
			r = append(r, e)
			rIDs = append(rIDs, e.id)
		} else {
			discarded = append(discarded, e)
		}
	}
	h.bKeep, h.bDisc, h.bIDs = r, discarded, rIDs
	for _, e := range discarded { // keepPrunedConnections: maintain degree
		if len(r) >= m {
			break
		}
		r = append(r, e)
	}
	h.bKeep = r
	return r
}

// sortCandsDesc returns a copy of w ordered by descending sim, ties by ascending
// id — deterministic, for the heuristic's nearest-first pass.
// candCmp orders candidates by similarity descending, then id ascending — a total
// order (ids are unique), so slices.SortFunc matches the previous sort.Slice output
// exactly while avoiding its reflect-based Swapper (audit #24). Same algorithm
// flat.go's hitCmp and bm25/sparse's own comparators reimplemented independently;
// now the one shared topk.Cmp.
func candCmp(a, b cand) int { return topk.Cmp(a.id, b.id, a.sim, b.sim) }

func sortCandsDesc(w []cand) []cand {
	out := make([]cand, len(w))
	copy(out, w)
	slices.SortFunc(out, candCmp)
	return out
}

// sortCandsDescInto is sortCandsDesc reusing dst (item 17).
func sortCandsDescInto(dst, w []cand) []cand {
	dst = append(dst[:0], w...)
	slices.SortFunc(dst, candCmp)
	return dst
}

func selectNearest(w []cand, m int) []cand {
	if len(w) <= m {
		return sortCandsDesc(w)
	}
	sel := topk.New[int](m)
	for _, c := range w {
		sel.Push(c.id, c.sim)
	}
	items := sel.Result() // descending by score
	out := make([]cand, len(items))
	for i, it := range items {
		out[i] = cand{id: it.Item, sim: it.Score}
	}
	return out
}

// cand is one (node id, similarity-to-query) pair used in the search and
// selection heaps.
type cand struct {
	id  int
	sim float64
}

// candHeap is a binary heap over cands. min=true → min-heap on sim (used
// for the bounded result set, worst on top); min=false → max-heap (used
// for the candidate frontier, closest expanded first).
type candHeap struct {
	items []cand
	min   bool
}

// Concrete typed heap — push/pop take/return cand directly. container/heap's
// Push(any)/Pop()any box every element into an interface, which during a build
// was ~23M heap allocations (1.8 GB); this version allocates only when the items
// slice grows.
func (h *candHeap) len() int { return len(h.items) }

func (h *candHeap) less(i, j int) bool {
	if h.min {
		return h.items[i].sim < h.items[j].sim
	}
	return h.items[i].sim > h.items[j].sim
}

func (h *candHeap) push(c cand) {
	h.items = append(h.items, c)
	for i := len(h.items) - 1; i > 0; {
		p := (i - 1) / 2
		if !h.less(i, p) {
			break
		}
		h.items[i], h.items[p] = h.items[p], h.items[i]
		i = p
	}
}

func (h *candHeap) pop() cand {
	top := h.items[0]
	n := len(h.items) - 1
	h.items[0] = h.items[n]
	h.items = h.items[:n]
	for i := 0; ; {
		l, r, best := 2*i+1, 2*i+2, i
		if l < n && h.less(l, best) {
			best = l
		}
		if r < n && h.less(r, best) {
			best = r
		}
		if best == i {
			break
		}
		h.items[i], h.items[best] = h.items[best], h.items[i]
		i = best
	}
	return top
}
