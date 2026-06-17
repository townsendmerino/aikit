package ann

import (
	"math/rand/v2"
	"os"
	"reflect"
	"testing"
)

// TestLoadFlatI8MmapPaged_matchesResidentAndEvicts is the D3 gate: a budget-paged
// FlatI8 index returns results IDENTICAL to the fully-resident one over the same
// queries (paging is lossless — the cap costs faults, never wrong codes), and the
// eviction count is > 0, proving the budget actually bit rather than everything
// merely fitting.
func TestLoadFlatI8MmapPaged_matchesResidentAndEvicts(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 22))
	const dim = 64
	pg := os.Getpagesize()
	// 4-page blocks so each has a non-empty page-aligned interior to page; 16 blocks
	// total so a few-block budget forces eviction.
	blockRows := 4 * pg / dim
	n := 16 * blockRows
	path := writeFlatI8Blob(t, NewFlatI8(randUnitSet(rng, n, dim)))

	resident, err := LoadFlatI8Mmap(path)
	if err != nil {
		t.Fatalf("resident load: %v", err)
	}
	defer resident.Close()

	// Budget holds ~4 of the 16 blocks resident → the other 12 must be paged.
	budget := int64(4 * blockRows * dim)
	paged, err := loadFlatI8MmapPaged(path, budget, blockRows)
	if err != nil {
		t.Fatalf("paged load: %v", err)
	}
	defer paged.Close()

	if _, _, _, isPaged := paged.PageStats(); !isPaged {
		t.Fatal("expected a paged index")
	}

	for i := range 30 {
		q := randUnit(rng, dim)
		want := resident.Query(q, 10)
		got := paged.Query(q, 10)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("query %d: paged result differs from resident\n got: %v\nwant: %v", i, got, want)
		}
		// Also exercise the unfiltered full-result path.
		if !reflect.DeepEqual(paged.Query(q, 0), resident.Query(q, 0)) {
			t.Fatalf("query %d: paged full-scan differs from resident", i)
		}
	}

	hits, misses, evictions, _ := paged.PageStats()
	if evictions == 0 {
		t.Fatalf("expected evictions > 0 (budget should have bitten); stats h=%d m=%d e=%d", hits, misses, evictions)
	}
	if got := paged.pager.Resident(); got > budget {
		t.Fatalf("resident bytes %d exceed budget %d", got, budget)
	}
}

// TestLoadFlatI8MmapPaged_autoBudget smoke-tests the public entry point (auto block
// size + auto budget). The corpus fits comfortably under the auto budget, so this
// just checks paging is transparent: same results as resident, no crash.
func TestLoadFlatI8MmapPaged_autoBudget(t *testing.T) {
	rng := rand.New(rand.NewPCG(33, 44))
	const dim = 48
	path := writeFlatI8Blob(t, NewFlatI8(randUnitSet(rng, 500, dim)))

	resident, err := LoadFlatI8Mmap(path)
	if err != nil {
		t.Fatal(err)
	}
	defer resident.Close()

	paged, err := LoadFlatI8MmapPaged(path, 0) // 0 ⇒ auto budget
	if err != nil {
		t.Fatal(err)
	}
	defer paged.Close()

	for range 15 {
		q := randUnit(rng, dim)
		if !reflect.DeepEqual(paged.Query(q, 10), resident.Query(q, 10)) {
			t.Fatal("auto-budget paged result differs from resident")
		}
	}
}

// TestLoadFlatI8MmapPaged_concurrentQueriesSafe checks that concurrent Query on a
// paged index is race-free (the stateful pager is guarded by pagerMu) and still
// returns the resident answer. Run under -race to exercise the guard.
func TestLoadFlatI8MmapPaged_concurrentQueriesSafe(t *testing.T) {
	rng := rand.New(rand.NewPCG(77, 88))
	const dim = 64
	pg := os.Getpagesize()
	blockRows := 4 * pg / dim
	n := 12 * blockRows
	path := writeFlatI8Blob(t, NewFlatI8(randUnitSet(rng, n, dim)))

	resident, err := LoadFlatI8Mmap(path)
	if err != nil {
		t.Fatal(err)
	}
	defer resident.Close()
	paged, err := loadFlatI8MmapPaged(path, int64(3*blockRows*dim), blockRows)
	if err != nil {
		t.Fatal(err)
	}
	defer paged.Close()

	// Precompute queries + expected answers single-threaded, then hammer concurrently.
	const G = 8
	queries := make([][]float32, G)
	want := make([][]Hit, G)
	for g := range G {
		queries[g] = randUnit(rng, dim)
		want[g] = resident.Query(queries[g], 10)
	}
	done := make(chan struct{})
	for g := range G {
		go func(g int) {
			defer func() { done <- struct{}{} }()
			for range 20 {
				if !reflect.DeepEqual(paged.Query(queries[g], 10), want[g]) {
					t.Errorf("goroutine %d: concurrent paged query differs from resident", g)
					return
				}
			}
		}(g)
	}
	for range G {
		<-done
	}
}

// TestFlatI8_nonPagedPageStats: a non-paged index reports paged=false, zero counts.
func TestFlatI8_nonPagedPageStats(t *testing.T) {
	f := NewFlatI8(randUnitSet(rand.New(rand.NewPCG(5, 6)), 20, 8))
	if h, m, e, paged := f.PageStats(); paged || h != 0 || m != 0 || e != 0 {
		t.Fatalf("non-paged PageStats = (%d,%d,%d,%v), want all zero/false", h, m, e, paged)
	}
}
