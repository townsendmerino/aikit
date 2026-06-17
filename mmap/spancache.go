package mmap

import "container/list"

// SpanCache bounds the resident RAM of a read-only mapping by paging spans of it in
// and out under a byte budget. It is the generic, demand-signal-agnostic core of a
// weight/index pager: the caller registers each member's page-aligned spans with
// Add and calls Touch(key) when a member is needed now. Touch faults the member in
// (Advise WILLNEED) and, if that pushes resident bytes over budget, releases the
// least-recently-touched members (Advise DONTNEED) until back under it.
//
// Releasing is lossless: a MapReadOnly mapping is read-only and file-backed, so an
// evicted-then-reused member merely re-faults from the file — its bytes are
// identical, only the cold-miss fault costs anything. So a budget-paged traversal
// returns the same results as a fully-resident one; the cap trades faults for RAM.
//
// SpanCache holds NO model-specific logic. It does not know what a member is or when
// to Touch it — the demand signal (a MoE router's top-k, an ANN query's scanned
// blocks, a layer-order prefetch) lives entirely in the caller. The key type K is
// whatever the caller uses to identify a member (a tensor pointer, a block index).
//
// Not goroutine-safe — one traversal touches it at a time, like a KV cache.
type SpanCache[K comparable] struct {
	budget   int64                    // resident-bytes cap; ≤ 0 means unbounded (never evict)
	resident int64                    // bytes currently held resident (our accounting)
	spans    map[K][][]byte           // page-aligned spans of the mapping, per member
	bytes    map[K]int64              // resident bytes per member (Σ aligned span lens)
	lru      *list.List               // K, most-recently-touched at front
	pos      map[K]*list.Element      // resident membership + O(1) promotion
	advise   func([]byte, bool) error // residency hint; Advise in production, a fake in tests

	hits, misses, evictions int64
}

// NewSpanCache returns a cache that caps resident registered spans at budget bytes.
// A budget ≤ 0 means unbounded: members are still tracked and prefetched on Touch,
// but nothing is ever evicted. Register members with Add before touching them.
func NewSpanCache[K comparable](budget int64) *SpanCache[K] {
	return &SpanCache[K]{
		budget: budget,
		spans:  map[K][][]byte{},
		bytes:  map[K]int64{},
		lru:    list.New(),
		pos:    map[K]*list.Element{},
		advise: Advise,
	}
}

// Add registers a member's spans under key. spans should be page-aligned (see
// PageAlignedInterior) so eviction of one member never disturbs another's pages;
// empty spans are dropped. A member starts non-resident — call Touch to fault it in.
// Re-adding a key that was already registered is ignored (the first registration
// wins); Add a distinct key per member.
func (c *SpanCache[K]) Add(key K, spans [][]byte) {
	if _, ok := c.spans[key]; ok {
		return
	}
	var kept [][]byte
	var n int64
	for _, s := range spans {
		if len(s) == 0 {
			continue
		}
		kept = append(kept, s)
		n += int64(len(s))
	}
	if n == 0 {
		return // nothing mapping-backed to page for this member
	}
	c.spans[key] = kept
	c.bytes[key] = n
}

// Touch records that key is needed now: it becomes most-recently-touched and, if it
// wasn't resident, is faulted in (Advise WILLNEED) and the LRU tail released to stay
// within budget. A no-op for keys that were never Added.
func (c *SpanCache[K]) Touch(key K) {
	spans, managed := c.spans[key]
	if !managed {
		return
	}
	if el, ok := c.pos[key]; ok {
		c.lru.MoveToFront(el)
		c.hits++
		return
	}
	c.misses++
	for _, s := range spans {
		_ = c.advise(s, true) // WILLNEED: hint the fault we're about to take
	}
	c.pos[key] = c.lru.PushFront(key)
	c.resident += c.bytes[key]
	for c.budget > 0 && c.resident > c.budget && c.lru.Len() > 1 {
		back := c.lru.Back()
		victim := back.Value.(K)
		c.lru.Remove(back)
		delete(c.pos, victim)
		c.resident -= c.bytes[victim]
		c.evictions++
		for _, s := range c.spans[victim] {
			_ = c.advise(s, false) // DONTNEED: release the victim's pages
		}
	}
}

// Resident reports the bytes the cache currently holds resident (its own accounting,
// Σ aligned span lengths of touched-and-not-evicted members). It never exceeds
// Budget once a member larger than the budget isn't in play.
func (c *SpanCache[K]) Resident() int64 { return c.resident }

// Budget reports the resident-bytes cap (≤ 0 means unbounded).
func (c *SpanCache[K]) Budget() int64 { return c.budget }

// Registered reports the total bytes of all Added members (resident or not) — the
// full footprint were nothing evicted. Useful for choosing a budget and for the
// load banner.
func (c *SpanCache[K]) Registered() int64 {
	var total int64
	for _, n := range c.bytes {
		total += n
	}
	return total
}

// Stats returns cumulative (hits, misses, evictions) over all Touch calls. A
// non-zero eviction count means the budget was actually enforced (the LRU tail was
// released), as opposed to mere cold-start misses.
func (c *SpanCache[K]) Stats() (hits, misses, evictions int64) {
	return c.hits, c.misses, c.evictions
}
