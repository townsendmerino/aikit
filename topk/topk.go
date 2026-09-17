// Package topk is a min-heap-of-size-K selector for "keep the K
// highest-scoring items from a stream" without sorting the full input.
// O(N log K) vs sort.Slice's O(N log N); the win grows with N relative
// to K.
//
// Use in place of "score everything into a slice, sort by score
// descending, take the first K" — a pattern that's invisible at toy
// scale but dominates search CPU at production scale.
//
// The implementation hand-rolls heap up/down rather than wrapping
// container/heap because (a) container/heap takes interface{} via
// heap.Interface which would force boxing and defeat the perf goal,
// and (b) the heap-of-fixed-K pattern is small enough that direct
// implementation is clearer than the heap.Interface dance.
//
// Generic typing (rather than interface{}) avoids the boxing allocations that
// interface{} would impose and per-callsite concrete-type copies. The
// production callers (aikit/ann and aikit/bm25) use different item types, so
// generics are the right shape.
//
// It's a standalone leaf package so a future caller (a reranker, a
// find_related reimplementation, …) that needs top-K selection can import it
// without the import path suggesting false coupling to BM25 or ANN.
package topk

import "math"

// scored is the internal heap element: an item plus its score. Kept
// non-generic-friendly (struct-of-T) so the slice backing the heap
// is contiguous and benefits from CPU-cache locality on the up/down
// sift loops.
type scored[T any] struct {
	item  T
	score float64
	seq   uint64 // insertion order, for the tie-break (audit #6)
}

// heapLess orders the min-heap on (score asc, seq desc): among equal scores the
// LATER-inserted item is "smaller" and sits at the root, so an eviction removes
// the newest tied item and the first-seen ones survive — the "ties broken by
// ascending document id / first-seen wins" contract bm25.TopK and sparse.Query
// document. Ordering only on score left heap[0] as an arbitrary tied minimum, so a
// higher score arriving later could evict an OLDER tied item and keep a newer one.
func heapLess[T any](a, b scored[T]) bool {
	if a.score != b.score {
		return a.score < b.score
	}
	return a.seq > b.seq
}

// Selector is a min-heap of fixed capacity. Push items as you score
// them; the heap retains the K highest-scoring observations seen so
// far. Read via Result() — returned in descending-score order.
//
// Tie-breaking: a strict greater-than comparison in Push ensures that
// when a new item ties the current minimum's score, the new item is
// discarded and the older one stays. Callers that iterate input in
// some natural order (ascending index, etc.) thereby inherit a stable
// "first-seen wins on tie" behavior without paying a secondary-key
// sort cost.
type Selector[T any] struct {
	k    int
	seq  uint64 // monotonic offer counter feeding the tie-break
	heap []scored[T]
}

// New returns a Selector that keeps the top k items by score.
// k=0 is valid (Result returns an empty slice; Push always discards).
// Negative k panics — caller error, not a runtime condition.
func New[T any](k int) *Selector[T] {
	if k < 0 {
		panic("topk.New: k must be non-negative")
	}
	return &Selector[T]{
		k:    k,
		heap: make([]scored[T], 0, k),
	}
}

// Reset clears the selector for a new sequence of pushes with capacity k,
// reusing the allocated backing array whenever possible.
func (s *Selector[T]) Reset(k int) {
	if k < 0 {
		panic("topk.Reset: k must be non-negative")
	}
	s.k = k
	s.seq = 0
	if cap(s.heap) < k {
		s.heap = make([]scored[T], 0, k)
	} else {
		s.heap = s.heap[:0]
	}
}

// Push offers (item, score) to the selector. If the heap hasn't reached
// capacity, item is added. Otherwise item replaces the current minimum
// iff score > min_score (strict; ties favor the older item per the
// tie-breaking note above). Returns true if item was retained, false
// if discarded.
func (s *Selector[T]) Push(item T, score float64) bool {
	if s.k == 0 {
		return false
	}
	// Reject NaN: every comparison against NaN is false, so at capacity a NaN
	// item would slip past the `score <= min` guard and evict the true minimum,
	// then poison every later sift (all comparisons against it false) — Result
	// could return NaN entries while genuinely high scorers were dropped. BM25 /
	// dot scorers can produce NaN from degenerate vectors, and this is the
	// terminal retrieval stage.
	if score != score {
		return false
	}
	if len(s.heap) < s.k {
		seq := s.seq
		s.seq++
		s.heap = append(s.heap, scored[T]{item: item, score: score, seq: seq})
		s.siftUp(len(s.heap) - 1)
		return true
	}
	// At capacity: only retain if strictly greater than current minimum.
	// Strict > still rejects a tied newcomer (score == min); the heap's
	// (score asc, seq desc) ordering makes heap[0] the latest-inserted tied
	// minimum, so evicting it keeps the first-seen tied items (audit #6).
	if score <= s.heap[0].score {
		return false
	}
	seq := s.seq
	s.seq++
	s.heap[0] = scored[T]{item: item, score: score, seq: seq}
	s.siftDown(0)
	return true
}

// Threshold is the score a new item must STRICTLY EXCEED to be retained: the current
// minimum once the selector is at capacity, and -Inf before that (everything is
// retained while there is room).
//
// It exists so a hot scan can skip the Push call entirely for the overwhelming
// majority of candidates. Push cannot be inlined — it contains the siftUp/siftDown
// calls — so every rejected candidate in an N-element scan pays a full call just to
// fail one comparison. Threshold is small enough to inline, which turns that into a
// compare in the caller's loop:
//
//	th := sel.Threshold()
//	for i, s := range scores {
//	    if s > th {
//	        sel.Push(i, s)
//	        th = sel.Threshold()   // only re-read when the heap actually changed
//	    }
//	}
//
// The guard must be `>`, matching Push's own strict comparison: a tied newcomer is
// rejected either way, so hoisting it changes nothing about which items are kept.
func (s *Selector[T]) Threshold() float64 {
	// k == 0 discards everything, and the heap is permanently empty — so the
	// threshold is +Inf ("nothing can be retained"), NOT heap[0], which would index
	// an empty slice. Push handles k == 0 with its own early return, so a caller
	// that hoists the threshold is the only way to reach this.
	if s.k == 0 {
		return math.Inf(1)
	}
	if len(s.heap) < s.k {
		return math.Inf(-1)
	}
	return s.heap[0].score
}

// ItemWithScore is the public read shape returned by Result.
type ItemWithScore[T any] struct {
	Item  T
	Score float64
}

// Extract invokes fn on each retained item in descending score order without allocating
// an intermediate slice of ItemWithScore.
func (s *Selector[T]) Extract(fn func(i int, item T, score float64)) {
	n := len(s.heap)
	if n == 0 {
		return
	}
	// Repeated extract-min would give ascending order — we reverse-fill
	// the output to land descending without a second pass.
	// For k <= 64 (standard retrieval k=10), use a stack scratch slice to
	// avoid a heap allocation on every Extract call.
	var stackTmp [64]scored[T]
	var tmp []scored[T]
	if n <= len(stackTmp) {
		tmp = stackTmp[:n]
	} else {
		tmp = make([]scored[T], n)
	}
	copy(tmp, s.heap)
	for i := n - 1; i >= 0; i-- {
		minEl := tmp[0]
		tmp[0] = tmp[len(tmp)-1]
		tmp = tmp[:len(tmp)-1]
		siftDownSlice(tmp, 0)
		fn(i, minEl.item, minEl.score)
	}
}

// Result returns the retained items in descending-score order. May be
// shorter than k if Push was called fewer than k times. Returned slice
// is freshly allocated; safe for callers to retain or mutate.
//
// Sort cost is O(K log K) — by construction K is small (typically 10
// for ken's search), so this is cheap relative to the N pushes that
// fed the heap.
//
// Tie-breaking on Result: the heap's internal ordering on ties is not
// defined, but because Push uses strict > (see Push comment), the
// retained K items at the tie boundary are the first-seen of any tied
// group. Result then sorts strictly by score; equal-score items emerge
// in heap-internal order, which is deterministic for a given input
// sequence but not lexically ordered by item.
func (s *Selector[T]) Result() []ItemWithScore[T] {
	n := len(s.heap)
	out := make([]ItemWithScore[T], n)
	s.Extract(func(i int, item T, score float64) {
		out[i] = ItemWithScore[T]{Item: item, Score: score}
	})
	return out
}

// Len is the current number of retained items (0 ≤ Len() ≤ k).
func (s *Selector[T]) Len() int { return len(s.heap) }

// ── heap operations ───────────────────────────────────────────────
// Min-heap: parent ≤ children. heap[0] is the smallest score.

func (s *Selector[T]) siftUp(i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if !heapLess(s.heap[i], s.heap[parent]) {
			return
		}
		s.heap[i], s.heap[parent] = s.heap[parent], s.heap[i]
		i = parent
	}
}

func (s *Selector[T]) siftDown(i int) { siftDownSlice(s.heap, i) }

// siftDownSlice is shared by Selector.siftDown and Result's
// extract-min loop (which operates on a scratch slice).
func siftDownSlice[T any](heap []scored[T], i int) {
	n := len(heap)
	for {
		l := 2*i + 1
		if l >= n {
			return
		}
		smallest := i
		if heapLess(heap[l], heap[smallest]) {
			smallest = l
		}
		if r := l + 1; r < n && heapLess(heap[r], heap[smallest]) {
			smallest = r
		}
		if smallest == i {
			return
		}
		heap[i], heap[smallest] = heap[smallest], heap[i]
		i = smallest
	}
}
