package mmap

import "testing"

// fakeAdvise records WILLNEED/DONTNEED calls so the LRU/eviction logic can be tested
// without a real mapping. The span pointers it sees are irrelevant — SpanCache's
// accounting is driven by registered byte counts, not by what Advise does.
type fakeAdvise struct {
	willNeed, dontNeed int
}

func (f *fakeAdvise) advise(_ []byte, willNeed bool) error {
	if willNeed {
		f.willNeed++
	} else {
		f.dontNeed++
	}
	return nil
}

// span returns a non-empty byte slice of length n to register as a member's span.
// SpanCache only reads len(span) for accounting, so the contents don't matter here.
func span(n int) []byte { return make([]byte, n) }

func TestSpanCache_evictsLRUTailOverBudget(t *testing.T) {
	fa := &fakeAdvise{}
	// Budget holds 2 of the 4 equal-sized members (100 each, budget 250).
	c := NewSpanCache[int](250)
	c.advise = fa.advise
	for i := range 4 {
		c.Add(i, [][]byte{span(100)})
	}

	c.Touch(0)
	c.Touch(1) // resident {1,0}, 200 ≤ 250, no eviction yet
	if _, _, ev := c.Stats(); ev != 0 {
		t.Fatalf("no eviction expected yet, got %d", ev)
	}
	c.Touch(2) // resident would be 300 > 250 → evict LRU tail (0)
	if got := c.Resident(); got != 200 {
		t.Fatalf("resident = %d, want 200 (two 100-byte members)", got)
	}
	if _, _, ev := c.Stats(); ev != 1 {
		t.Fatalf("evictions = %d, want 1", ev)
	}

	// 0 was the LRU tail and should have been evicted; touching it again is a miss.
	hitsBefore, _, _ := c.Stats()
	c.Touch(1) // 1 is still resident → hit, promotes it
	if h, _, _ := c.Stats(); h != hitsBefore+1 {
		t.Fatalf("touching resident member should be a hit")
	}
	_, missBefore, _ := c.Stats()
	c.Touch(0) // evicted earlier → miss, faults back in, evicts new tail (2)
	if _, m, _ := c.Stats(); m != missBefore+1 {
		t.Fatalf("touching evicted member should be a miss")
	}
	if got := c.Resident(); got > c.Budget() {
		t.Fatalf("resident %d exceeds budget %d", got, c.Budget())
	}

	// Every fault hinted WILLNEED; every eviction hinted DONTNEED.
	_, misses, evictions := c.Stats()
	if int64(fa.willNeed) != misses {
		t.Fatalf("WILLNEED calls = %d, want one per miss (%d)", fa.willNeed, misses)
	}
	if int64(fa.dontNeed) != evictions {
		t.Fatalf("DONTNEED calls = %d, want one per eviction (%d)", fa.dontNeed, evictions)
	}
}

func TestSpanCache_alwaysKeepsTouchedMember(t *testing.T) {
	fa := &fakeAdvise{}
	// Budget (50) is smaller than a single member (100): the just-touched member
	// must stay resident anyway — never evict the thing you just asked for.
	c := NewSpanCache[int](50)
	c.advise = fa.advise
	c.Add(0, [][]byte{span(100)})
	c.Add(1, [][]byte{span(100)})

	c.Touch(0)
	if c.Resident() != 100 {
		t.Fatalf("a single touched member must stay resident, got %d", c.Resident())
	}
	c.Touch(1) // evicts 0, keeps 1 — Len never drops below 1
	if c.Resident() != 100 {
		t.Fatalf("resident = %d, want 100 (only the latest member)", c.Resident())
	}
	if _, _, ev := c.Stats(); ev != 1 {
		t.Fatalf("evictions = %d, want 1", ev)
	}
}

func TestSpanCache_unboundedNeverEvicts(t *testing.T) {
	fa := &fakeAdvise{}
	c := NewSpanCache[int](0) // ≤ 0 ⇒ unbounded
	c.advise = fa.advise
	for i := range 5 {
		c.Add(i, [][]byte{span(100)})
		c.Touch(i)
	}
	if c.Resident() != 500 {
		t.Fatalf("unbounded cache resident = %d, want 500", c.Resident())
	}
	if _, _, ev := c.Stats(); ev != 0 {
		t.Fatalf("unbounded cache must never evict, got %d", ev)
	}
	if fa.dontNeed != 0 {
		t.Fatalf("unbounded cache must never DONTNEED, got %d", fa.dontNeed)
	}
}

func TestSpanCache_touchUnregisteredIsNoOp(t *testing.T) {
	c := NewSpanCache[int](1000)
	c.Touch(42) // never Added
	if h, m, e := c.Stats(); h != 0 || m != 0 || e != 0 {
		t.Fatalf("touching an unregistered key must not move any counter, got %d/%d/%d", h, m, e)
	}
	if c.Resident() != 0 {
		t.Fatalf("resident = %d, want 0", c.Resident())
	}
}

func TestSpanCache_addDropsEmptySpansAndDedups(t *testing.T) {
	c := NewSpanCache[int](1000)
	c.Add(0, [][]byte{span(40), nil, span(60)}) // empty span dropped
	if c.Registered() != 100 {
		t.Fatalf("Registered = %d, want 100 (empty span dropped)", c.Registered())
	}
	c.Add(0, [][]byte{span(999)}) // re-add ignored: first registration wins
	if c.Registered() != 100 {
		t.Fatalf("re-adding a key must be ignored, Registered = %d", c.Registered())
	}

	// A member with only empty spans is not registered at all (nothing to page).
	c.Add(1, [][]byte{nil, {}})
	c.Touch(1)
	if _, m, _ := c.Stats(); m != 0 {
		t.Fatalf("a member with no real spans should not be touchable, misses = %d", m)
	}
}
