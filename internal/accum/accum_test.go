package accum

import (
	"fmt"
	"math/rand"
	"slices"
	"testing"
)

// TestAccum_orderTouchedMatchesSort exercises the multi-run merge path (the part
// bm25 and sparse both separately got wrong once — item 44 / item 39) against the
// reference: sort the touched set and compare. Runs from 1 to 9 terms so the merge
// hits the single-run passthrough, the 2-run direct-merge, and the pairwise-reduce
// paths.
func TestAccum_orderTouchedMatchesSort(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for nTerms := 1; nTerms <= 9; nTerms++ {
		for iter := range 50 {
			const n = 500
			a := Get(n)
			var want []int32
			for range nTerms {
				a.BeginRun()
				// Each "term" touches an ascending run of doc ids, like a real
				// posting list — the invariant OrderTouched's merge relies on.
				start := rng.Intn(n - 10)
				for _, doc := range randAscending(rng, start, n) {
					a.Add(int32(doc), 1)
					want = append(want, int32(doc))
				}
			}
			a.OrderTouched()

			wantSorted := slices.Clone(want)
			slices.Sort(wantSorted)
			wantSorted = slices.Compact(wantSorted) // Add dedupes on first touch too
			if !slices.Equal(a.Touched, wantSorted) {
				t.Fatalf("nTerms=%d iter=%d: OrderTouched() = %v, want %v", nTerms, iter, a.Touched, wantSorted)
			}
			Put(a)
		}
	}
}

// randAscending returns a random-length ascending run of distinct doc ids in [start, n).
func randAscending(rng *rand.Rand, start, n int) []int {
	var out []int
	for d := start; d < n; d++ {
		if rng.Intn(4) == 0 {
			out = append(out, d)
		}
	}
	return out
}

// TestAccum_generationReuse confirms a doc from a stale generation reads as
// untouched — the whole point of the generation stamp — without an explicit clear,
// across a pooled Get/Put cycle.
func TestAccum_generationReuse(t *testing.T) {
	a := Get(10)
	a.BeginRun()
	a.Add(3, 5)
	a.OrderTouched()
	if got := a.Scores[3]; got != 5 {
		t.Fatalf("Scores[3] = %v, want 5", got)
	}
	Put(a)

	b := Get(10) // same underlying buffer from the pool, next generation
	b.BeginRun()
	b.Add(7, 9)
	b.OrderTouched()
	// Touched is the only promised read surface (Scores[d] is meaningful only
	// for d in Touched — bm25 and sparse never read it otherwise) — doc 3's
	// stale value from the prior generation must not appear here even though
	// the underlying array is reused unzeroed.
	if len(b.Touched) != 1 || b.Touched[0] != 7 {
		t.Fatalf("Touched = %v, want [7] — doc 3 from the prior generation leaked through", b.Touched)
	}
	Put(b)
}

func BenchmarkOrderTouched(b *testing.B) {
	for _, nTerms := range []int{2, 3, 8, 30} {
		b.Run(fmt.Sprintf("terms%d", nTerms), func(b *testing.B) {
			rng := rand.New(rand.NewSource(42))
			const n = 200_000
			a := Get(n)
			defer Put(a)
			for range nTerms {
				a.BeginRun()
				start := rng.Intn(1000)
				for d := start; d < n; d += 5 + rng.Intn(20) {
					a.Add(int32(d), 1.0)
				}
			}
			savedTouched := slices.Clone(a.Touched)
			savedRuns := slices.Clone(a.runs)
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				a.Touched = append(a.Touched[:0], savedTouched...)
				a.runs = append(a.runs[:0], savedRuns...)
				a.OrderTouched()
			}
		})
	}
}
