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

import (
	"container/heap"
	"math"
	"math/rand/v2"
	"sort"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/aikit/topk"
)

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
}

type hnswNode struct {
	layer int       // top layer this node belongs to
	nbrs  [][]int32 // nbrs[l] = neighbor ids at layer l, len == layer+1
}

// HNSW is an approximate cosine index. Build with NewHNSW + Add (or
// BuildHNSW); query with Query. See the type doc for thread-safety.
type HNSW struct {
	vecs           [][]float32
	nodes          []hnswNode
	dim            int
	m, m0          int
	efConstruction int
	efSearch       int
	mL             float64 // level-generation normalizer = 1/ln(M)
	entry          int     // entry-point node id (top of the graph)
	maxLayer       int
	seed           uint64 // Config.Seed, retained so a loaded index re-seeds rng
	heuristic      bool   // Config.Heuristic — Alg-4 diversity neighbor selection
	rng            *rand.Rand
}

// NewHNSW creates an empty index. Add vectors with Add, or use BuildHNSW
// to bulk-load.
func NewHNSW(cfg Config) *HNSW {
	m := cfg.M
	if m <= 0 {
		m = 16
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
		rng:            rand.New(rand.NewPCG(cfg.Seed, cfg.Seed^0x9e3779b97f4a7c15)),
	}
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
func (h *HNSW) Len() int { return len(h.vecs) }

// mmax is the per-layer neighbor cap: 2*M at layer 0, M above.
func (h *HNSW) mmax(layer int) int {
	if layer == 0 {
		return h.m0
	}
	return h.m
}

// sim is the cosine similarity (dot product on unit vectors) between q
// and indexed vector id. Scored with the SIMD dot kernel (linalg.Dot,
// float32 accumulation); HNSW is approximate by contract, so the sub-ULP
// difference from a float64 scalar sum is immaterial to recall.
func (h *HNSW) sim(q []float32, id int) float64 {
	return float64(linalg.Dot(q, h.vecs[id]))
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
	id := len(h.vecs)
	if id == 0 {
		h.dim = len(vec)
	} else if len(vec) != h.dim {
		panic("ann: HNSW.Add dimension mismatch")
	}
	h.vecs = append(h.vecs, vec)

	l := h.randomLevel()
	h.nodes = append(h.nodes, hnswNode{layer: l, nbrs: make([][]int32, l+1)})

	if id == 0 {
		h.entry = 0
		h.maxLayer = l
		return id
	}

	ep := h.entry
	// Greedy descent through the layers above l: refine the single best
	// entry point, no link changes.
	for layer := h.maxLayer; layer > l; layer-- {
		ep = h.greedyClosest(vec, ep, layer)
	}
	// Insert layers min(l, maxLayer) … 0.
	start := min(h.maxLayer, l)
	for layer := start; layer >= 0; layer-- {
		w := h.searchLayer(vec, []int{ep}, h.efConstruction, layer)
		neighbors := h.selectNeighbors(w, h.mmax(layer))
		// Link id → neighbors.
		ids := make([]int32, len(neighbors))
		for i, c := range neighbors {
			ids[i] = int32(c.id)
		}
		h.nodes[id].nbrs[layer] = ids
		// Link neighbors → id, pruning any that overflow mmax.
		for _, c := range neighbors {
			h.nodes[c.id].nbrs[layer] = append(h.nodes[c.id].nbrs[layer], int32(id))
			if len(h.nodes[c.id].nbrs[layer]) > h.mmax(layer) {
				h.prune(c.id, layer)
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

// prune trims node id's neighbor list at layer to the mmax most similar.
func (h *HNSW) prune(id, layer int) {
	nbrs := h.nodes[id].nbrs[layer]
	self := h.vecs[id]
	cands := make([]cand, len(nbrs))
	for i, n := range nbrs {
		cands[i] = cand{id: int(n), sim: h.sim(self, int(n))}
	}
	kept := h.selectNeighbors(cands, h.mmax(layer))
	out := make([]int32, len(kept))
	for i, c := range kept {
		out[i] = int32(c.id)
	}
	h.nodes[id].nbrs[layer] = out
}

// greedyClosest walks from ep along layer-`layer` links, always moving
// to the neighbor most similar to q, until no neighbor improves. Used
// for the ef=1 descent above the insertion/query layer.
func (h *HNSW) greedyClosest(q []float32, ep, layer int) int {
	best := ep
	bestSim := h.sim(q, ep)
	for {
		improved := false
		for _, n := range h.nodes[best].nbrs[layer] {
			s := h.sim(q, int(n))
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
func (h *HNSW) searchLayer(q []float32, entryPoints []int, ef, layer int) []cand {
	visited := make(map[int]struct{}, ef*2)
	// candidates: max-heap on sim (expand the closest-to-q first).
	candidates := &candHeap{min: false}
	// results: min-heap on sim (the worst of the ef-best sits on top for
	// O(1) eviction / comparison).
	results := &candHeap{min: true}

	for _, ep := range entryPoints {
		s := h.sim(q, ep)
		visited[ep] = struct{}{}
		heap.Push(candidates, cand{id: ep, sim: s})
		heap.Push(results, cand{id: ep, sim: s})
	}

	for candidates.Len() > 0 {
		c := heap.Pop(candidates).(cand)
		// If the closest remaining candidate is worse than the worst
		// result and we already have ef, no unexplored node can improve
		// the result set — stop.
		if results.Len() >= ef && c.sim < results.items[0].sim {
			break
		}
		for _, nb := range h.nodes[c.id].nbrs[layer] {
			n := int(nb)
			if _, seen := visited[n]; seen {
				continue
			}
			visited[n] = struct{}{}
			s := h.sim(q, n)
			if results.Len() < ef || s > results.items[0].sim {
				heap.Push(candidates, cand{id: n, sim: s})
				heap.Push(results, cand{id: n, sim: s})
				if results.Len() > ef {
					heap.Pop(results) // drop the current worst
				}
			}
		}
	}

	out := make([]cand, results.Len())
	copy(out, results.items)
	sort.Slice(out, func(a, b int) bool {
		if out[a].sim != out[b].sim {
			return out[a].sim > out[b].sim
		}
		return out[a].id < out[b].id
	})
	return out
}

// Query returns the k highest-cosine-similarity vectors to q, descending,
// ties broken by ascending index — matching Flat.Query's contract. Uses
// the configured EfSearch (effective ef = max(EfSearch, k)). Returns nil
// for an empty index or a dimension-mismatched q.
func (h *HNSW) Query(q []float32, k int) []Hit {
	return h.QueryEf(q, k, h.efSearch)
}

// QueryEf is Query with an explicit ef (candidate-list size) for this
// call, the recall/latency knob: larger ef ⇒ higher recall, slower.
// Effective ef is max(ef, k).
func (h *HNSW) QueryEf(q []float32, k, ef int) []Hit {
	if h.Len() == 0 || k <= 0 || len(q) != h.dim {
		return nil
	}
	if ef < k {
		ef = k
	}
	// Descend greedily from the top entry point to layer 1.
	ep := h.entry
	for layer := h.maxLayer; layer >= 1; layer-- {
		ep = h.greedyClosest(q, ep, layer)
	}
	// Full ef search at layer 0.
	found := h.searchLayer(q, []int{ep}, ef, 0)
	if len(found) > k {
		found = found[:k]
	}
	hits := make([]Hit, len(found))
	for i, c := range found {
		hits[i] = Hit{Index: c.id, Score: c.sim}
	}
	return hits
}

// selectNearest returns the m highest-similarity candidates from w
// (which searchLayer already returns sorted desc, but prune passes an
// unsorted slice, so re-rank via a K-selector for O(N log m)).
// selectNeighbors picks up to m edges from the candidate set w, dispatching to
// the diversity heuristic (Algorithm 4) when Config.Heuristic is set, else the
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
func (h *HNSW) simIDs(a, b int) float64 { return float64(linalg.Dot(h.vecs[a], h.vecs[b])) }

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
func (h *HNSW) selectHeuristic(w []cand, m int) []cand {
	cands := sortCandsDesc(w)
	if len(cands) <= m {
		return cands
	}
	r := make([]cand, 0, m)
	discarded := make([]cand, 0, len(cands))
	for _, e := range cands {
		if len(r) >= m {
			break
		}
		keep := true
		for _, sel := range r {
			// e is closer to an already-selected neighbor than to the base ⇒
			// redundant (same direction), discard it.
			if h.simIDs(e.id, sel.id) > e.sim {
				keep = false
				break
			}
		}
		if keep {
			r = append(r, e)
		} else {
			discarded = append(discarded, e)
		}
	}
	for _, e := range discarded { // keepPrunedConnections: maintain degree
		if len(r) >= m {
			break
		}
		r = append(r, e)
	}
	return r
}

// sortCandsDesc returns a copy of w ordered by descending sim, ties by ascending
// id — deterministic, for the heuristic's nearest-first pass.
func sortCandsDesc(w []cand) []cand {
	out := make([]cand, len(w))
	copy(out, w)
	sort.Slice(out, func(a, b int) bool {
		if out[a].sim != out[b].sim {
			return out[a].sim > out[b].sim
		}
		return out[a].id < out[b].id
	})
	return out
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

func (h *candHeap) Len() int { return len(h.items) }
func (h *candHeap) Less(i, j int) bool {
	if h.min {
		return h.items[i].sim < h.items[j].sim
	}
	return h.items[i].sim > h.items[j].sim
}
func (h *candHeap) Swap(i, j int) { h.items[i], h.items[j] = h.items[j], h.items[i] }
func (h *candHeap) Push(x any)    { h.items = append(h.items, x.(cand)) }
func (h *candHeap) Pop() any {
	n := len(h.items)
	x := h.items[n-1]
	h.items = h.items[:n-1]
	return x
}
