// Package ann is the dense (semantic) retriever. v1 is a flat brute-force
// cosine scan — the "vicinity" equivalent in ken's DESIGN.md §1. HNSW lands
// later behind this same Hit/Query shape; flat is exact and fine at repo
// scale.
//
// This package moved here from ken (ADR-034); "DESIGN.md" refers to
// https://github.com/townsendmerino/ken/blob/main/docs/DESIGN.md.
//
// Invariants the rest of the codebase depends on:
//
//   - **Input vectors are L2-normalized.** embed.StaticModel.Encode
//     normalizes its output before returning, so cosine similarity is
//     just the dot product — Query computes that, not a full
//     ‖a‖‖b‖-divided cosine. Passing non-normalized vectors silently
//     produces incorrect rankings; the precision contract lives at the
//     embed boundary, not here.
//   - **Scores are float32-precision.** Query scores each vector with the
//     SIMD dot kernel (linalg.Dot, float32 accumulation), not a float64
//     scalar sum. For unit-norm float32 inputs the per-element error is
//     bounded and recall is unaffected; only sub-ULP near-ties may order
//     differently than an exact float64 scan would. The ascending-Index
//     tie-break still makes the result deterministic.
//   - **Similarity, not distance.** semble's dense backend (vicinity)
//     returns cosine *distance* (1 − sim) and search.py flips it back to
//     similarity; ken skips the round-trip and scores similarity
//     directly, with "higher = better." Anything reading the Score
//     field must treat it that way.
//   - **Goroutine-safety.** A built *Flat is read-only — Query takes no
//     locks and is safe to call concurrently across goroutines. New is
//     not thread-safe (single builder); Query is.
//   - **No mutation.** *Flat is immutable after New, by design. v0.3's
//     incremental indexing (internal/search/watch.go, ADR-012) does not
//     mutate an existing *Flat — instead the writer builds a brand-new
//     *Flat alongside a new *bm25.Index + chunks slice, wraps them in a
//     new *search.Index snapshot, and publishes the pointer atomically.
//     Readers load the snapshot pointer once at query entry. So Flat
//     stays goroutine-safe-by-immutability; what changes is that the
//     containing search.Index can be swapped wholesale between queries.
package ann

import (
	"runtime"
	"slices"
	"sync"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/aikit/topk"
)

// Hit is one scored item, highest Score (cosine similarity) first.
type Hit struct {
	Index int
	Score float64
}

// Flat is an exhaustive cosine index over a fixed set of unit vectors.
type Flat struct {
	vecs [][]float32 // each assumed L2-normalized (embed.Encode guarantees this)
	dim  int
}

// New builds a flat index. Vectors are used by reference, not copied.
func New(vecs [][]float32) *Flat {
	d := 0
	if len(vecs) > 0 {
		d = len(vecs[0])
	}
	return &Flat{vecs: vecs, dim: d}
}

// Len is the number of indexed vectors.
func (f *Flat) Len() int { return len(f.vecs) }

// Query returns the k highest cosine-similarity vectors to q, descending,
// ties broken by ascending index for determinism. k<=0 or k>=Len returns
// all, sorted.
//
// Two paths by design:
//
//   - k<=0 || k>=Len: full sort over every dot-product result. Preserves
//     the documented "return all, sorted" semantic for callers that want
//     the complete ranked list.
//   - 0 < k < Len: min-heap of size K via internal/topk. O(N log K) vs
//     the full-sort path's O(N log N) — at medium scale (~378k chunks,
//     k=10) this was 30.88% of hybrid search CPU per ADR-025. Final
//     sort.SliceStable imposes the ascending-Index tie-break the doc
//     comment promises, which the heap on its own doesn't guarantee
//     (heap only sorts by score; ties within result come out in
//     heap-internal order). The K-sized stable sort is O(K log K) —
//     cheap at K=10.
func (f *Flat) Query(q []float32, k int) []Hit {
	return f.query(q, k, nil)
}

// QueryFilter is Query restricted to documents for which keep(id) is true — a
// logical-delete / live-set filter applied at query time, so the index stays
// immutable. Exact for Flat (it scores every vector and filters before selecting),
// unlike the approximate HNSW.QueryFilter. A nil keep is exactly Query.
//
// keep MUST be a pure predicate that is safe for concurrent use. The scan is
// sharded across cores, so keep is called from several goroutines at once, and
// — because each shard maintains its own running top-k threshold — the SET of
// ids it is asked about, and their order, differ from any serial
// implementation. A read-only live-set lookup (the intended use) satisfies this;
// a closure that counts calls or memoizes into shared state does not.
//
// The returned hits do not depend on any of that: they are the k highest by
// score with ties broken by ascending index, identical to a serial scan.
func (f *Flat) QueryFilter(q []float32, k int, keep func(id int) bool) []Hit {
	return f.query(q, k, keep)
}

func (f *Flat) query(q []float32, k int, keep func(int) bool) []Hit {
	// Full-sort path: k<=0 returns everything; k>=Len would have nothing
	// to discard anyway. Either way the heap buys us nothing.
	if k <= 0 || k >= len(f.vecs) {
		hits := make([]Hit, 0, len(f.vecs))
		scanFlat(q, f.vecs, func(i int, score float64) {
			if keep == nil || keep(i) {
				hits = append(hits, Hit{Index: i, Score: score})
			}
		})
		slices.SortFunc(hits, hitCmp)
		return hits
	}

	// Heap path: 0 < k < Len. Score every vector, push into the K-sized
	// min-heap; the heap only retains the K highest seen so far.
	var items []topk.ItemWithScore[int]
	if w := flatQueryWorkers(len(f.vecs), f.dim); w > 1 {
		items = f.queryShards(q, k, w, keep)
	} else {
		items = scanTopK(q, f.vecs, 0, k, keep)
	}
	// Stable secondary sort by ascending Index to honor the doc-comment
	// tie-break contract. K is small (typically 10), so this is cheap.
	slices.SortFunc(items, topk.ItemCmp[int])
	hits := make([]Hit, len(items))
	for j, s := range items {
		hits[j] = Hit{Index: s.Item, Score: s.Score}
	}
	return hits
}

// scanFlat dots q against every candidate vector and calls emit(index, score)
// for each (dimension mismatches are skipped, never panicked). It streams 8
// vectors per pass through the register-blocked SIMD kernel (linalg.Dot8x4): the
// shared query is loaded once and reused across 8 candidates — the a-reuse trick
// the blocked matmul uses — which beats one linalg.Dot call per vector. The
// kernel covers the first ⌊d/4⌋·4 dims; a scalar tail handles the d%4 remainder,
// and the final <8 vectors (plus any ragged-dim group) fall back to linalg.Dot.
// scanTopK scores vecs against q and returns the k best as (score desc, index
// asc), with indices offset by base. It is the shared body of both the serial
// and the sharded query path — one implementation, so a shard cannot drift from
// the whole.
func scanTopK(q []float32, vecs [][]float32, base, k int, keep func(int) bool) []topk.ItemWithScore[int] {
	sel := topk.New[int](k)
	// See ann/flat_i8.go's topHits: the threshold is hoisted so the non-inlinable
	// Push is only called for candidates that can actually be retained.
	th := sel.Threshold()
	scanFlat(q, vecs, func(i int, score float64) {
		gi := base + i
		if score > th && (keep == nil || keep(gi)) {
			sel.Push(gi, score)
			th = sel.Threshold()
		}
	})
	return sel.Result()
}

// flatParallelThreshold is the scored-element count (N·dim) at or above which
// sharding pays for the goroutine spawn. Below it the scan is a few hundred
// microseconds and the fan-out is pure overhead.
const flatParallelThreshold = 1 << 19

// flatQueryWorkers decides the fan-out for one query.
func flatQueryWorkers(n, dim int) int {
	if n < 2 {
		return 1
	}
	if int64(n)*int64(dim) < flatParallelThreshold {
		return 1
	}
	w := min(runtime.NumCPU(), n)
	return w
}

// queryShards splits the scan across workers, each with its OWN k-selector, then
// merges.
//
// The merge is exact, not approximate. Flat's contract is "the k highest by
// score, ties broken by ascending index", and the serial path implements it by
// scanning in ascending index order with topk's strict-> rule, so a tie at the
// boundary keeps the first-seen (lowest) index. That is precisely a global sort
// by (score desc, index asc) truncated to k. Any element in that global top-k is
// also in its own shard's top-k under the same ordering — a shard is a subset —
// so it survives to the merge, and the merge re-applies the same total order.
// Sharding therefore cannot change which elements are returned, only how they
// are found. TestFlatQuery_shardedMatchesSerial gates it on adversarial
// all-ties input.
func (f *Flat) queryShards(q []float32, k, workers int, keep func(int) bool) []topk.ItemWithScore[int] {
	n := len(f.vecs)
	per := (n + workers - 1) / workers
	parts := make([][]topk.ItemWithScore[int], workers)
	var wg sync.WaitGroup
	for w := range workers {
		lo := w * per
		if lo >= n {
			break
		}
		hi := min(lo+per, n)
		wg.Add(1)
		go func(w, lo, hi int) {
			defer wg.Done()
			parts[w] = scanTopK(q, f.vecs[lo:hi], lo, k, keep)
		}(w, lo, hi)
	}
	wg.Wait()

	merged := make([]topk.ItemWithScore[int], 0, workers*k)
	for _, p := range parts {
		merged = append(merged, p...)
	}
	slices.SortFunc(merged, topk.ItemCmp[int])
	if len(merged) > k {
		merged = merged[:k]
	}
	return merged
}

func scanFlat(q []float32, vecs [][]float32, emit func(i int, score float64)) {
	d := len(q)
	n4 := d / 4
	tailStart := n4 * 4
	var sums [32]float32
	i := 0
	// d>0 guards &v[0] in the streamed path; d==0 (empty query) falls straight
	// to the scalar remainder, where linalg.Dot handles zero-length safely.
	for ; d > 0 && i+8 <= len(vecs); i += 8 {
		v0, v1, v2, v3 := vecs[i], vecs[i+1], vecs[i+2], vecs[i+3]
		v4, v5, v6, v7 := vecs[i+4], vecs[i+5], vecs[i+6], vecs[i+7]
		if len(v0) != d || len(v1) != d || len(v2) != d || len(v3) != d ||
			len(v4) != d || len(v5) != d || len(v6) != d || len(v7) != d {
			// Ragged group (defensive; index vectors normally share one dim).
			for j := range 8 {
				if v := vecs[i+j]; len(v) == d {
					emit(i+j, float64(linalg.Dot(q, v)))
				}
			}
			continue
		}
		linalg.Dot8x4(&q[0], &v0[0], &v1[0], &v2[0], &v3[0], &v4[0], &v5[0], &v6[0], &v7[0], n4, &sums)
		if tailStart == d {
			for j := range 8 {
				b := j * 4
				s := sums[b] + sums[b+1] + sums[b+2] + sums[b+3]
				emit(i+j, float64(s))
			}
		} else {
			group := [8][]float32{v0, v1, v2, v3, v4, v5, v6, v7}
			for j := range 8 {
				// Each row's dot is spread across its 4-lane block (the arm64 kernel
				// leaves 4 partial sums; the generic puts it all in lane 0 + zeros).
				// Sum the block, then add the d%4 scalar tail.
				b := j * 4
				s := sums[b] + sums[b+1] + sums[b+2] + sums[b+3]
				for kk := tailStart; kk < d; kk++ {
					s += q[kk] * group[j][kk]
				}
				emit(i+j, float64(s))
			}
		}
	}
	for ; i < len(vecs); i++ {
		if v := vecs[i]; len(v) == d {
			emit(i, float64(linalg.Dot(q, v)))
		}
	}
}

// hitCmp is shared by every Hit-selection site in this package, and it is a
// `slices.SortFunc` comparator rather than a `sort.Slice` closure for two
// reasons — one mechanical, one about correctness.
//
// Mechanical: sort.Slice and sort.SliceStable go through reflect.Swapper, so
// every swap is an indirect call through reflection. slices.SortFunc is generic
// and swaps typed values directly. That is perf-campaign A5, and audit #24
// already made the same change in hnsw.go; it simply never reached the other
// eight sites.
//
// Correctness: SortFunc is NOT stable, and this does not need it to be. The
// second key is Hit.Index, which is unique within the slice being sorted, so
// this is a strict TOTAL order — there are no ties for stability to resolve,
// and the output is identical to what a stable sort would produce. This is the
// argument candCmp already makes in hnsw.go.
//
// Note the contrast with fuse.RRF, changed in the same item, where stability IS
// load-bearing because its tie-break lives in the element positions. Same
// package-level change, opposite conclusion, for a reason visible in the
// comparator.
//
// The int/int32 ItemWithScore comparators this package used to define
// alongside it (itemCmp, itemCmp32) were the exact same algorithm bm25 and
// sparse also wrote out over their own result types — now topk.Cmp /
// topk.ItemCmp, called directly at each site rather than re-wrapped here.
func hitCmp(a, b Hit) int { return topk.Cmp(a.Index, b.Index, a.Score, b.Score) }
