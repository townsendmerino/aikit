package ann

// HNSW is the approximate, sublinear sibling of Flat — a Hierarchical
// Navigable Small World graph (Malkov & Yashunin, 2016). Flat is exact
// but O(N) per query, fine to the low hundreds of thousands of vectors;
// HNSW trades a small recall hit for ~O(log N) search, the lever for
// millions of vectors. It deliberately lives behind the SAME Hit /
// Query(q, k) shape as Flat (see the package doc), so a consumer can
// swap one for the other without touching call sites.
//
// Same invariants as Flat carry over:
//
//   - Input vectors are L2-normalized; similarity is the dot product
//     ("higher = better"), reported in Hit.Score, NOT a distance.
//   - Build (NewHNSW + Add, or BuildHNSW) is single-writer: not safe for
//     concurrent Add. Query is read-only and safe to call concurrently
//     across goroutines ONCE building has finished — same contract as
//     Flat (New not thread-safe, Query is). There is no internal locking
//     on the hot path.
//
// Neighbor selection defaults to the diversity heuristic (paper Algorithm 4,
// selectHeuristic): edges fan out across directions instead of clustering, which
// is what holds recall up on clustered data — a real Model2Vec code corpus went
// from 0.68 (plain M-nearest, Algorithm 3) to 1.00 recall@10 with it (bench/).
// Config.SimpleNeighbors opts back to the cheaper-to-build Algorithm 3.
// Determinism: level assignment is seeded (Config.Seed), so a given
// (vectors, Config) builds the same graph every time — important for
// reproducible tests.
//
// This file holds the struct, construction (Add/BuildHNSW), and search
// (searchLayer/Query*). Neighbor selection (Algorithm 3/selectNearest and
// Algorithm 4/selectHeuristic) plus the binary heap they both use live in
// hnsw_select.go; serialization lives in hnsw_persist.go.

import (
	"math"
	"math/rand/v2"
	"slices"
	"sync"

	"github.com/townsendmerino/aikit/linalg"
)

// visitTracker is searchLayer's visited set as a generation-stamped slice instead
// of a map: reset() bumps a generation counter (O(1), no clearing), seen/mark are
// plain array reads/writes (no hashing, no per-search allocation). A map here cost
// ~16% of build CPU. NOT safe for concurrent use — build reuses one (single-writer
// Add); each Query borrows one from a pool.
type visitTracker struct {
	stamp []uint32
	gen   uint32
	// Scratch for the batched neighbour scoring (item 15). It lives here rather
	// than in searchLayer because this struct is already pooled per query, so
	// batching costs no new allocation.
	ids    []int
	scores []float64
}

func (v *visitTracker) reset(n int) {
	if cap(v.stamp) < n {
		// Grow with headroom — during a build n rises by 1 each insert, so an
		// exact-size make would reallocate every call.
		c := max(2*cap(v.stamp), n)
		v.stamp = make([]uint32, n, c)
	} else {
		v.stamp = v.stamp[:n]
	}
	v.gen++
	if v.gen == 0 { // counter wrapped — clear once and restart
		for i := range v.stamp {
			v.stamp[i] = 0
		}
		v.gen = 1
	}
}

func (v *visitTracker) seen(id int) bool { return v.stamp[id] == v.gen }
func (v *visitTracker) mark(id int)      { v.stamp[id] = v.gen }

// searchScratch is everything one searchLayer call needs to scribble on: the
// visited-set stamps and the two heaps. They are pooled together because they
// have the same lifetime and the same sharing rule — one per in-flight search —
// and because the heaps' backing arrays were the bulk of a query's allocations
// (they grow by append from nil to ef on every call, ~7 doublings each).
type searchScratch struct {
	vis     visitTracker
	cands   candHeap // frontier, max-heap
	results candHeap // bounded result set, min-heap
}

// getScratch / putScratch lend a searchScratch to a Query from the per-index
// pool, so concurrent queries don't share state and don't each re-grow an
// N-sized stamp buffer and two ef-sized heaps.
func (h *HNSW) getScratch() *searchScratch {
	if v, ok := h.queryVis.Get().(*searchScratch); ok {
		return v
	}
	return &searchScratch{}
}
func (h *HNSW) putScratch(v *searchScratch) { h.queryVis.Put(v) }

// Config tunes the graph. Zero values fall back to the documented
// defaults, so Config{} is a sensible build.
type Config struct {
	// M is the max neighbors per node per layer above 0; layer 0 uses
	// 2*M (the standard M0). Higher M ⇒ better recall, more memory and
	// slower build. Default 16.
	M int
	// EfConstruction is the candidate-list size during insertion. Higher
	// ⇒ better graph quality (recall) at higher build cost. Default 200.
	EfConstruction int
	// EfSearch is the default candidate-list size during Query; the
	// effective value is max(EfSearch, k). Higher ⇒ better recall, slower
	// query. Default 64. Override per-query with QueryEf.
	EfSearch int
	// Seed seeds the level-assignment RNG for reproducible builds.
	Seed uint64
	// SimpleNeighbors opts into plain M-nearest neighbor selection (paper
	// Algorithm 3) instead of the DEFAULT diversity heuristic (Algorithm 4). The
	// heuristic spreads each node's edges across directions rather than piling
	// them into one cluster, which sharply improves recall on CLUSTERED data — on
	// a real Model2Vec code corpus it lifted recall@10 from 0.68 to 1.00 — at
	// roughly 2× build cost (query cost unchanged). Set this only to trade that
	// recall for a faster build. See selectHeuristic.
	SimpleNeighbors bool

	// Int8 stores the vectors as int8 (per-vector symmetric quantization) instead
	// of float32 — ¼ the vector memory, and the persisted/embedded blob shrinks to
	// match. Build, search, and persistence all run in the integer domain. Recall
	// is essentially unchanged on real embeddings (the int8 quantization is the
	// same that FlatI8 uses; see TestHNSW_int8RecallGate). Experimental.
	Int8 bool
}

type hnswNode struct {
	layer int       // top layer this node belongs to
	nbrs  [][]int32 // nbrs[l] = neighbor ids at layer l, len == layer+1
}

// HNSW is an approximate cosine index. Build with NewHNSW + Add (or
// BuildHNSW); query with Query. See the type doc for thread-safety.
type HNSW struct {
	// Vectors are stored as float32 (vecs) OR, when int8 is set, as row-major int8
	// codes (bq, [count*dim]) + per-vector scales. Exactly one is populated.
	vecs           [][]float32
	int8           bool
	bq             []int8
	scales         []float32
	nodes          []hnswNode
	dim            int
	m, m0          int
	efConstruction int
	efSearch       int
	mL             float64 // level-generation normalizer = 1/ln(M)
	entry          int     // entry-point node id (top of the graph)
	maxLayer       int
	seed           uint64 // Config.Seed, retained so a loaded index re-seeds rng
	heuristic      bool   // !Config.SimpleNeighbors — Alg-4 diversity neighbor selection
	rng            *rand.Rand
	buildScratch   searchScratch // reused across searchLayer calls during Add (single-writer)
	queryVis       sync.Pool     // *searchScratch per concurrent Query
	// scoreUnbatched forces the per-candidate scoring path. Test-only: it exists
	// so the differential gate for item 15 can run the pristine and batched
	// scorers against the SAME graph and compare full ranked result lists, which
	// is the only way to check the transformation preserved push order.
	scoreUnbatched bool

	// Build-time scratch. Add is documented "not safe for concurrent use", so
	// these need no pool or lock — they exist because selectHeuristic, prune and
	// sortCandsDesc allocated fresh slices on EVERY call, which at 10k×d256 was
	// 225 allocations per insert (perf-campaign item 17).
	bSort []cand
	bKeep []cand
	bDisc []cand
	bIDs  []int
	bSims []float64
	bCand []cand
	bOut2 []int

	// mmap is the backing mapping when built by LoadHNSWMmap (vecs/bq alias it);
	// nil for a NewHNSW/Add or Load index. closed is set by Close. Same lifetime
	// contract as FlatI8's identical fields (ann/flat_i8.go).
	mmap   []byte
	closed bool
}

// NewHNSW creates an empty index. Add vectors with Add, or use BuildHNSW
// to bulk-load.
func NewHNSW(cfg Config) *HNSW {
	m := cfg.M
	if m <= 0 {
		m = 16
	}
	if m < 2 {
		// M=1 gives mL = 1/ln(1) = +Inf, so randomLevel's int(-ln(U)·mL)
		// overflows (MinInt64/MaxInt64) and make([][]int32, level+1) panics on
		// the first Add. A 1-neighbor graph is degenerate anyway — clamp to the
		// minimum viable connectivity.
		m = 2
	}
	efc := cfg.EfConstruction
	if efc <= 0 {
		efc = 200
	}
	efs := cfg.EfSearch
	if efs <= 0 {
		efs = 64
	}
	return &HNSW{
		m:              m,
		m0:             2 * m,
		efConstruction: efc,
		efSearch:       efs,
		mL:             1.0 / math.Log(float64(m)),
		entry:          -1,
		seed:           cfg.Seed,
		heuristic:      !cfg.SimpleNeighbors,
		int8:           cfg.Int8,
		rng:            rand.New(rand.NewPCG(cfg.Seed, cfg.Seed^0x9e3779b97f4a7c15)),
	}
}

// count is the number of indexed vectors, from whichever storage is active.
func (h *HNSW) count() int {
	if h.int8 {
		return len(h.scales)
	}
	return len(h.vecs)
}

// BuildHNSW builds an index over vecs (used by reference, not copied),
// added in order. Equivalent to NewHNSW + a loop of Add.
func BuildHNSW(vecs [][]float32, cfg Config) *HNSW {
	h := NewHNSW(cfg)
	for _, v := range vecs {
		h.Add(v)
	}
	return h
}

// Len is the number of indexed vectors.
func (h *HNSW) Len() int { return h.count() }

// mmax is the per-layer neighbor cap: 2*M at layer 0, M above.
func (h *HNSW) mmax(layer int) int {
	if layer == 0 {
		return h.m0
	}
	return h.m
}

// queryVec is a search query prepared once for the index's storage mode: f32
// holds the raw vector (float32 mode), or q8 + scale its int8 quantization (int8
// mode). It threads through greedyClosest/searchLayer into sim, so the query is
// quantized once per search rather than per similarity.
type queryVec struct {
	f32   []float32
	q8    []int8
	scale float64
}

func (h *HNSW) prepare(q []float32) queryVec {
	if !h.int8 {
		return queryVec{f32: q}
	}
	q8 := make([]int8, len(q))
	scale := linalg.QuantizeRowInt8(q, q8)
	return queryVec{q8: q8, scale: float64(scale)}
}

// code returns node id's int8 row (int8 mode only).
func (h *HNSW) code(id int) []int8 { return h.bq[id*h.dim : id*h.dim+h.dim] }

// sim is the cosine similarity (dot on unit vectors) between the prepared query
// and indexed vector id — the SIMD f32 dot (float32 mode) or the int8 dot rescaled
// by the query and per-vector scales (int8 mode). HNSW is approximate by contract,
// so the sub-ULP difference from a float64 scalar sum is immaterial to recall.
//
// ⚠ MATCHING-UNITS INVARIANT — sim and simIDs must stay on the same scale.
//
// selectHeuristic cross-compares the two: it tests a node–node similarity from
// simIDs against `e.sim`, which came from sim(qv, ·). In the int8 mode those are
// only comparable because of an equality that nothing in the type system
// enforces — in Add, `qv = h.prepare(vec)` and `h.bq[id]` are quantized from the
// SAME row by the same pure function, so `qv.scale == h.scales[id]` exactly and
// sim(qv, c) ≡ simIDs(id, c).
//
// The consequence is worth stating plainly because it is the kind of change that
// looks obviously safe: `qv.scale` is loop-invariant inside sim, so hoisting it
// out is a textbook optimization — and hoisting it out of sim ALONE changes the
// units of one side of that comparison only. The result is a DIFFERENT GRAPH.
//
// It is worse than "no test failure" — that was measured. Dropping `qv.scale`
// here passes the whole ann suite AND an int8-mode recall gate built for it,
// scoring 0.978 on clustered data, because on L2-normalized vectors every
// per-vector scale is nearly the same number and the mutation is close to a
// uniform rescale of one side. The gate that does catch it,
// TestHNSW_simAndSimIDsAgree, asserts the invariant itself: sim(prepare(v), c)
// and simIDs(id, c) must be the SAME NUMBER for the node built from v. If you
// factor a scale out of one of these, factor it out of both, and run that test.
//
// (The related simplification is sound and is the right way to read the test:
// h.scales[e.id] appears as a positive factor on BOTH sides and cancels exactly.
// It has also been measured, at three dimensions, at 0% — see the lens doc's
// dead ends.)
func (h *HNSW) sim(qv queryVec, id int) float64 {
	if h.int8 {
		return float64(linalg.DotI8(qv.q8, h.code(id))) * qv.scale * float64(h.scales[id])
	}
	return float64(linalg.Dot(qv.f32, h.vecs[id]))
}

// scoreInto fills dst[i] with sim(qv, ids[i]).
//
// The f32 path scores EIGHT candidates per call through linalg.Dot8x4, which
// holds the query strip in registers and amortizes it across the 8 rows instead
// of re-streaming it per neighbour (perf-campaign item 15). Flat already used
// this kernel; hnsw was calling linalg.Dot once per neighbour.
//
// NOT bit-identical to the per-neighbour Dot: the 8-row kernel accumulates in a
// different order, so scores can move by ~1 float32 ULP. That is the same
// reassociation tradeoff Flat documents, and HNSW is approximate by contract —
// but the traversal is threshold-driven, so a moved score could in principle
// change which nodes are explored. TestHNSW_batchedScoringMatchesPristine
// checks it does not, over a grid of dims, sizes and ef values.
//
// The int8 path ALSO scores eight candidates per call, through linalg.DotI8x8
// (audit M-21): linalg had no gathered 8-row int8 kernel when this comment was
// written, so the mode was left scalar even though DotI8 already runs a SIMD
// reduction per candidate — DotI8x8 shares the QUERY's widening across all
// eight the same way Dot8x4 shares the query strip, on top of each candidate
// already being SIMD-scored.
//
// Bit-identical to the per-candidate path: DotI8x8 is exactly eight DotI8
// calls' worth of integer arithmetic (associative, no overflow risk — see
// linalg's own doc comment), and the rescale below is the same
// scale*h.scales[id] multiply h.sim does per candidate, just applied to
// DotI8x8's eight raw sums instead of DotI8's one.
func (h *HNSW) scoreInto(qv queryVec, ids []int, dst []float64) []float64 {
	dst = dst[:0]
	if h.scoreUnbatched {
		for _, id := range ids {
			dst = append(dst, h.sim(qv, id))
		}
		return dst
	}
	if h.int8 {
		// int8 mode never populates h.vecs (Add stores it in h.bq instead —
		// see Add's if h.int8 branch), so the len(h.vecs)==0 guard below,
		// which exists for the f32 path's empty-index case, was ALWAYS true
		// here and silently forced every int8 score through the scalar path
		// regardless of h.scoreUnbatched. Caught by measurement, not
		// inspection: the correctness gate (TestHNSW_int8BatchedScoringMatchesPristine)
		// could not catch it because the bypassed scalar path computes the
		// same right answer — see perf-dead-ends.md for the full chase.
		q := qv.q8
		i := 0
		for ; i+8 <= len(ids); i += 8 {
			c0, c1, c2, c3 := h.code(ids[i]), h.code(ids[i+1]), h.code(ids[i+2]), h.code(ids[i+3])
			c4, c5, c6, c7 := h.code(ids[i+4]), h.code(ids[i+5]), h.code(ids[i+6]), h.code(ids[i+7])
			sums := linalg.DotI8x8(q, c0, c1, c2, c3, c4, c5, c6, c7)
			for j := range 8 {
				id := ids[i+j]
				dst = append(dst, float64(sums[j])*qv.scale*float64(h.scales[id]))
			}
		}
		for ; i < len(ids); i++ {
			dst = append(dst, h.sim(qv, ids[i]))
		}
		return dst
	}
	if len(h.vecs) == 0 {
		for _, id := range ids {
			dst = append(dst, h.sim(qv, id))
		}
		return dst
	}
	q := qv.f32
	d := len(q)
	n4 := d / 4
	tailStart := n4 * 4
	var sums [32]float32
	i := 0
	for ; d > 0 && i+8 <= len(ids); i += 8 {
		v0, v1, v2, v3 := h.vecs[ids[i]], h.vecs[ids[i+1]], h.vecs[ids[i+2]], h.vecs[ids[i+3]]
		v4, v5, v6, v7 := h.vecs[ids[i+4]], h.vecs[ids[i+5]], h.vecs[ids[i+6]], h.vecs[ids[i+7]]
		if len(v0) != d || len(v1) != d || len(v2) != d || len(v3) != d ||
			len(v4) != d || len(v5) != d || len(v6) != d || len(v7) != d {
			for j := range 8 { // ragged (defensive; index vectors share one dim)
				dst = append(dst, h.sim(qv, ids[i+j]))
			}
			continue
		}
		linalg.Dot8x4(&q[0], &v0[0], &v1[0], &v2[0], &v3[0], &v4[0], &v5[0], &v6[0], &v7[0], n4, &sums)
		group := [8][]float32{v0, v1, v2, v3, v4, v5, v6, v7}
		for j := range 8 {
			// Each row's dot is spread across its 4-lane block; sum the block,
			// then add the d%4 scalar tail — same fold Flat uses.
			b := j * 4
			sc := sums[b] + sums[b+1] + sums[b+2] + sums[b+3]
			for kk := tailStart; kk < d; kk++ {
				sc += q[kk] * group[j][kk]
			}
			dst = append(dst, float64(sc))
		}
	}
	for ; i < len(ids); i++ {
		dst = append(dst, h.sim(qv, ids[i]))
	}
	return dst
}

func (h *HNSW) randomLevel() int {
	// floor(-ln(U) * mL), U in (0,1].
	r := h.rng.Float64()
	if r <= 0 {
		r = math.SmallestNonzeroFloat64
	}
	return int(-math.Log(r) * h.mL)
}

// Add inserts vec (by reference) and returns its assigned index. Not
// safe for concurrent use. Panics on a dimension mismatch with vectors
// already added (a programmer error, like topk's negative-k panic).
func (h *HNSW) Add(vec []float32) int {
	if h.closed {
		panic("ann: Add on a closed HNSW (mmap released by Close)")
	}
	id := h.count()
	if id == 0 {
		h.dim = len(vec)
	} else if len(vec) != h.dim {
		panic("ann: HNSW.Add dimension mismatch")
	}
	if h.int8 {
		q8 := make([]int8, h.dim)
		h.bq = append(h.bq, q8...) // grow, then quantize into the tail
		s := linalg.QuantizeRowInt8(vec, h.bq[id*h.dim:id*h.dim+h.dim])
		h.scales = append(h.scales, s)
	} else {
		h.vecs = append(h.vecs, vec)
	}

	l := h.randomLevel()
	h.nodes = append(h.nodes, hnswNode{layer: l, nbrs: make([][]int32, l+1)})

	if id == 0 {
		h.entry = 0
		h.maxLayer = l
		return id
	}

	qv := h.prepare(vec) // quantize the query once (int8 mode); raw f32 otherwise
	ep := h.entry
	// Greedy descent through the layers above l: refine the single best
	// entry point, no link changes.
	for layer := h.maxLayer; layer > l; layer-- {
		ep = h.greedyClosest(qv, ep, layer)
	}
	// Insert layers min(l, maxLayer) … 0.
	start := min(h.maxLayer, l)
	for layer := start; layer >= 0; layer-- {
		w := h.searchLayer(qv, []int{ep}, h.efConstruction, layer, &h.buildScratch, nil)
		neighbors := h.selectNeighbors(w, h.mmax(layer))
		// Link id → neighbors.
		ids := make([]int32, len(neighbors))
		for i, c := range neighbors {
			ids[i] = int32(c.id)
		}
		h.nodes[id].nbrs[layer] = ids
		// Link neighbors → id, pruning any that overflow mmax.
		//
		// Iterating `ids` rather than `neighbors` is load-bearing: prune re-enters
		// selectHeuristic, which overwrites the build scratch `neighbors` aliases
		// (item 17). `ids` is a freshly allocated []int32 that prune cannot touch
		// — it only rewrites h.nodes[c.id], never h.nodes[id]. Ranging over
		// `neighbors` here silently corrupted the graph: recall@10 fell from 1.00
		// to 0.83 with no test failing except the recall gates.
		for _, nid := range ids {
			c := int(nid)
			h.nodes[c].nbrs[layer] = append(h.nodes[c].nbrs[layer], int32(id))
			if len(h.nodes[c].nbrs[layer]) > h.mmax(layer) {
				h.prune(c, layer)
			}
		}
		if len(w) > 0 {
			ep = w[0].id // best candidate carries to the next layer down
		}
	}
	if l > h.maxLayer {
		h.maxLayer = l
		h.entry = id
	}
	return id
}

// greedyClosest walks from ep along layer-`layer` links, always moving
// to the neighbor most similar to q, until no neighbor improves. Used
// for the ef=1 descent above the insertion/query layer.
func (h *HNSW) greedyClosest(qv queryVec, ep, layer int) int {
	best := ep
	bestSim := h.sim(qv, ep)
	for {
		improved := false
		for _, n := range h.nodes[best].nbrs[layer] {
			s := h.sim(qv, int(n))
			if s > bestSim {
				bestSim, best, improved = s, int(n), true
			}
		}
		if !improved {
			return best
		}
	}
}

// searchLayer returns the ef vectors in `layer` most similar to q,
// reachable from entryPoints, as a slice sorted by descending
// similarity. This is the HNSW inner loop (paper Algorithm 2).
// keep, if non-nil, restricts the COLLECTED results to ids where keep(id) is
// true. Filtered-out nodes still route the search (pushed to the frontier) — graph
// connectivity is preserved — they're just not added to the result set.
func (h *HNSW) searchLayer(qv queryVec, entryPoints []int, ef, layer int, sc *searchScratch, keep func(int) bool) []cand {
	vis := &sc.vis
	vis.reset(h.count())
	// candidates: max-heap on sim (expand the closest-to-q first).
	// results:    min-heap on sim (the worst of the ef-best sits on top for
	//             O(1) eviction / comparison).
	// Both reuse the scratch's backing arrays: truncating to [:0] keeps the
	// capacity a previous search grew, so a warm scratch allocates nothing here.
	candidates, results := &sc.cands, &sc.results
	candidates.items, candidates.min = candidates.items[:0], false
	results.items, results.min = results.items[:0], true

	for _, ep := range entryPoints {
		s := h.sim(qv, ep)
		vis.mark(ep)
		candidates.push(cand{id: ep, sim: s})
		if keep == nil || keep(ep) {
			results.push(cand{id: ep, sim: s})
		}
	}

	for candidates.len() > 0 {
		c := candidates.pop()
		// If the closest remaining candidate is worse than the worst
		// result and we already have ef, no unexplored node can improve
		// the result set — stop.
		if results.len() >= ef && c.sim < results.items[0].sim {
			break
		}
		// Two phases, deliberately. Collecting the unseen neighbours first (marking
		// them exactly as before) lets them be scored eight at a time; the push
		// logic then runs over them IN THE ORIGINAL ORDER, so the evolving
		// results.items[0].sim threshold sees the identical sequence it saw when
		// scoring was interleaved (item 15).
		vis.ids = vis.ids[:0]
		for _, nb := range h.nodes[c.id].nbrs[layer] {
			n := int(nb)
			if vis.seen(n) {
				continue
			}
			vis.mark(n)
			vis.ids = append(vis.ids, n)
		}
		vis.scores = h.scoreInto(qv, vis.ids, vis.scores)
		for i, n := range vis.ids {
			s := vis.scores[i]
			if results.len() < ef || s > results.items[0].sim {
				candidates.push(cand{id: n, sim: s}) // route through it
				if keep == nil || keep(n) {
					results.push(cand{id: n, sim: s})
					if results.len() > ef {
						results.pop() // drop the current worst
					}
				}
			}
		}
	}

	out := make([]cand, results.len())
	copy(out, results.items)
	slices.SortFunc(out, candCmp)
	return out
}

// Query returns the k highest-cosine-similarity vectors to q, descending,
// ties broken by ascending index — matching Flat.Query's contract. Uses
// the configured EfSearch (effective ef = max(EfSearch, k)). Returns nil
// for an empty index or a dimension-mismatched q.
func (h *HNSW) Query(q []float32, k int) []Hit {
	return h.queryEf(q, k, h.efSearch, nil)
}

// QueryEf is Query with an explicit ef (candidate-list size) for this
// call, the recall/latency knob: larger ef ⇒ higher recall, slower.
// Effective ef is max(ef, k).
func (h *HNSW) QueryEf(q []float32, k, ef int) []Hit {
	return h.queryEf(q, k, ef, nil)
}

// QueryFilter is Query restricted to the documents for which keep(id) is true — a
// logical-delete / live-set filter applied at query time, so the index stays
// immutable (the cornerstone). Filtered-out nodes still ROUTE the search, so graph
// connectivity is intact and live recall holds; they're simply not returned. Under
// heavy deletion the live result can fall short of k (the search runs out of live
// nodes within ef) — rebuild the index to purge tombstones when that bites. A nil
// keep is exactly Query.
// keep must be a pure predicate that is safe for concurrent use, per
// Flat.QueryFilter — see the note there. This implementation applies it
// serially today; the requirement is stated uniformly across the indexes.
func (h *HNSW) QueryFilter(q []float32, k int, keep func(id int) bool) []Hit {
	return h.queryEf(q, k, h.efSearch, keep)
}

func (h *HNSW) queryEf(q []float32, k, ef int, keep func(int) bool) []Hit {
	if h.closed {
		panic("ann: Query on a closed HNSW (mmap released by Close)")
	}
	if h.Len() == 0 || k <= 0 || len(q) != h.dim {
		return nil
	}
	if ef < k {
		ef = k
	}
	// Descend greedily from the top entry point to layer 1 (routing is never
	// filtered — only the final collected set is).
	qv := h.prepare(q) // quantize the query once (int8 mode); raw f32 otherwise
	ep := h.entry
	for layer := h.maxLayer; layer >= 1; layer-- {
		ep = h.greedyClosest(qv, ep, layer)
	}
	sc := h.getScratch()
	found := h.searchLayer(qv, []int{ep}, ef, 0, sc, keep)
	h.putScratch(sc)
	if len(found) > k {
		found = found[:k]
	}
	hits := make([]Hit, len(found))
	for i, c := range found {
		hits[i] = Hit{Index: c.id, Score: c.sim}
	}
	return hits
}
