package ann

import (
	"math/rand/v2"
	"reflect"
	"testing"
)

// queryFilterer is the shared QueryFilter surface across the dense indexes.
type queryFilterer interface {
	Query(q []float32, k int) []Hit
	QueryFilter(q []float32, k int, keep func(int) bool) []Hit
}

func TestQueryFilter_excludesDeleted(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	vecs := randUnitSet(rng, 2000, 64)
	deleted := func(id int) bool { return id%7 == 0 }
	keep := func(id int) bool { return !deleted(id) }

	idx := map[string]queryFilterer{
		"Flat":   New(vecs),
		"HNSW":   BuildHNSW(vecs, Config{M: 16, EfConstruction: 200, EfSearch: 128, Seed: 1}),
		"FlatI8": NewFlatI8(vecs),
	}
	for range 20 {
		q := randUnit(rng, 64)
		for name, ix := range idx {
			for _, h := range ix.QueryFilter(q, 10, keep) {
				if deleted(h.Index) {
					t.Errorf("%s.QueryFilter returned deleted doc %d", name, h.Index)
				}
			}
		}
	}
}

// Flat.QueryFilter is exact: it must equal Query over the live subset.
func TestFlatQueryFilter_exactVsManualFilter(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	vecs := randUnitSet(rng, 1000, 64)
	keep := func(id int) bool { return id%3 != 0 }
	f := New(vecs)

	for qi := range 10 {
		q := randUnit(rng, 64)
		got := f.QueryFilter(q, 10, keep)
		// Reference: full ranking, drop filtered, take 10.
		var want []Hit
		for _, h := range f.Query(q, -1) {
			if keep(h.Index) {
				want = append(want, h)
				if len(want) == 10 {
					break
				}
			}
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("query %d: QueryFilter != manual-filtered Query\n got %v\nwant %v", qi, got, want)
		}
	}
}

// HNSW.QueryFilter keeps high recall under sparse deletion: filtered nodes still
// route the search, so the live top-k is still reachable.
func TestHNSW_QueryFilter_recallUnderDeletion(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	vecs := randUnitSet(rng, 3000, 64)
	keep := func(id int) bool { return id%5 != 0 } // 20% deleted
	flat := New(vecs)
	hnsw := BuildHNSW(vecs, Config{M: 16, EfConstruction: 200, EfSearch: 128, Seed: 1})

	const Q, k = 60, 10
	var sum float64
	for range Q {
		q := randUnit(rng, 64)
		truth := make(map[int]bool, k)
		for _, h := range flat.QueryFilter(q, k, keep) { // exact live top-k
			truth[h.Index] = true
		}
		got := 0
		for _, h := range hnsw.QueryFilter(q, k, keep) {
			if truth[h.Index] {
				got++
			}
		}
		sum += float64(got) / float64(k)
	}
	mean := sum / Q
	t.Logf("HNSW.QueryFilter recall@%d under 20%% deletion = %.4f", k, mean)
	if mean < 0.90 {
		t.Errorf("recall %.4f < 0.90 — routing through deleted nodes regressed?", mean)
	}
}

// A nil keep is exactly Query, across all three indexes.
func TestQueryFilter_nilEqualsQuery(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	vecs := randUnitSet(rng, 500, 64)
	q := randUnit(rng, 64)
	for name, ix := range map[string]queryFilterer{
		"Flat": New(vecs), "HNSW": BuildHNSW(vecs, Config{Seed: 1}), "FlatI8": NewFlatI8(vecs),
	} {
		if !reflect.DeepEqual(ix.QueryFilter(q, 10, nil), ix.Query(q, 10)) {
			t.Errorf("%s: QueryFilter(nil) != Query", name)
		}
	}
}
