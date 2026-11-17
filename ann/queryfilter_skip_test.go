package ann

import (
	"math/rand/v2"
	"reflect"
	"slices"
	"testing"

	"github.com/townsendmerino/aikit/topk"
)

// The filtered scans skip a disallowed row before scoring it. Two gates:
//
//   - identity: the hits are exactly what score-every-row-then-filter returns
//     (same indices, same float64 scores, same order);
//   - speed: a 10% scope is measurably cheaper than an unfiltered query, which
//     it cannot be while every row is still scored.

// refFiltered is the score-everything-then-filter reference: the store's own
// full ranking, filtered, truncated to k.
func refFiltered(all []Hit, k int, keep func(int) bool) []Hit {
	var out []Hit
	for _, h := range all {
		if keep(h.Index) {
			out = append(out, h)
		}
	}
	if k > 0 && len(out) > k {
		out = out[:k]
	}
	return out
}

// flatRefFilter is Flat.QueryFilter as it was before rows were skipped: score
// every row (same shard layout, same kernels), filter, then select. It is the
// baseline "bit-identical to today" is measured against; Query(q, -1) is not,
// because a sharded scan puts the d%4 / <8-row tails at different rows than the
// serial one does and those can differ in the last bit.
func flatRefFilter(f *Flat, q []float32, k int, keep func(int) bool) []Hit {
	if k <= 0 || k >= len(f.vecs) {
		var hits []Hit
		scanFlat(q, f.vecs, func(i int, s float64) {
			if keep(i) {
				hits = append(hits, Hit{Index: i, Score: s})
			}
		})
		slices.SortFunc(hits, hitCmp)
		return hits
	}
	var items []topk.ItemWithScore[int]
	scan := func(lo, hi int) []topk.ItemWithScore[int] {
		sel := topk.New[int](k)
		scanFlat(q, f.vecs[lo:hi], func(i int, s float64) {
			if keep(lo + i) {
				sel.Push(lo+i, s)
			}
		})
		return sel.Result()
	}
	w := flatQueryWorkers(len(f.vecs), f.dim)
	per := (len(f.vecs) + w - 1) / w
	for lo := 0; lo < len(f.vecs); lo += per {
		items = append(items, scan(lo, min(lo+per, len(f.vecs)))...)
	}
	slices.SortFunc(items, topk.ItemCmp[int])
	if len(items) > k {
		items = items[:k]
	}
	hits := make([]Hit, len(items))
	for j, s := range items {
		hits[j] = Hit{Index: s.Item, Score: s.Score}
	}
	return hits
}

func TestFlatQueryFilter_skipIdentical(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	for _, tc := range []struct{ n, d int }{
		{1, 8}, {7, 8}, {8, 8}, {9, 8}, {100, 13}, {997, 64}, {5000, 96}, {20001, 30},
	} {
		vecs := randUnitSet(rng, tc.n, tc.d)
		f := New(vecs)
		fi := NewFlatI8(vecs)
		for _, sel := range []struct {
			name string
			keep func(int) bool
		}{
			{"all", func(int) bool { return true }},
			{"none", func(int) bool { return false }},
			{"tenth", func(id int) bool { return id%10 == 3 }},
			{"half", func(id int) bool { return id%2 == 0 }},
			{"one", func(id int) bool { return id == tc.n/2 }},
			{"prefix", func(id int) bool { return id < 13 }},
			{"suffix", func(id int) bool { return id >= tc.n-13 }},
		} {
			q := randUnit(rng, tc.d)
			for _, st := range []struct {
				name string
				ix   queryFilterer
			}{{"Flat", f}, {"FlatI8", fi}} {
				all := st.ix.Query(q, -1)
				for _, k := range []int{-1, 1, 10, 64, tc.n + 5} {
					want := refFiltered(all, k, sel.keep)
					if fx, ok := st.ix.(*Flat); ok {
						want = flatRefFilter(fx, q, k, sel.keep)
					}
					got := st.ix.QueryFilter(q, k, sel.keep)
					if len(got) == 0 && len(want) == 0 {
						continue
					}
					if !reflect.DeepEqual(got, want) {
						t.Fatalf("%s n=%d d=%d %s k=%d:\n got %v\nwant %v", st.name, tc.n, tc.d, sel.name, k, got, want)
					}
				}
			}
		}
	}
}

// TestFlatQueryFilter_scoresOnlyAllowedRows is the deterministic form of "a
// narrow scope does less work": the scan emits a score for exactly the allowed
// rows and for no others. (Wall-clock ratios are in BenchmarkQueryFilterScope,
// not asserted here: a timing assertion flakes on shared CI runners.)
func TestFlatQueryFilter_scoresOnlyAllowedRows(t *testing.T) {
	rng := rand.New(rand.NewPCG(31, 32))
	vecs := randUnitSet(rng, 5003, 30)
	q := randUnit(rng, 30)
	for _, keep := range []func(int) bool{
		func(id int) bool { return id%10 == 0 },
		func(id int) bool { return id >= 1000 && id < 1500 },
		func(int) bool { return false },
	} {
		want, got := 0, 0
		for i := range vecs {
			if keep(i) {
				want++
			}
		}
		seen := map[int]bool{}
		scanFlatKeep(q, vecs, 0, keep, func(gi int, _ float64) {
			got++
			if !keep(gi) || seen[gi] {
				t.Fatalf("row %d scored but not allowed, or scored twice", gi)
			}
			seen[gi] = true
		})
		if got != want {
			t.Fatalf("scored %d rows, want exactly the %d allowed", got, want)
		}
	}
}

// BenchmarkQueryFilterScope reports unfiltered vs a 10% scope (clustered and
// scattered) per store, for the release perf gate and for A/B by hand.
func BenchmarkQueryFilterScope(b *testing.B) {
	rng := rand.New(rand.NewPCG(21, 22))
	const n = 60000
	vecs := randUnitSet(rng, n, 256)
	q := randUnit(rng, 256)
	clustered := func(id int) bool { return id >= n/3 && id < n/3+n/10 }
	scattered := func(id int) bool { return id%10 == 0 }
	for _, st := range []struct {
		name string
		ix   queryFilterer
	}{{"Flat", New(vecs)}, {"FlatI8", NewFlatI8(vecs)}} {
		b.Run(st.name+"/unfiltered", func(b *testing.B) {
			for b.Loop() {
				st.ix.Query(q, 10)
			}
		})
		b.Run(st.name+"/clustered10pct", func(b *testing.B) {
			for b.Loop() {
				st.ix.QueryFilter(q, 10, clustered)
			}
		})
		b.Run(st.name+"/scattered10pct", func(b *testing.B) {
			for b.Loop() {
				st.ix.QueryFilter(q, 10, scattered)
			}
		})
	}
}
