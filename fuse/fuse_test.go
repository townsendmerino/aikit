package fuse

import (
	"math"
	"testing"
)

func approx(a, b float64) bool { return math.Abs(a-b) < 1e-12 }

// TestRRF_basic: a key ranked high in both lists must beat keys ranked
// high in only one. Hand-computes the canonical RRF scores.
func TestRRF_basic(t *testing.T) {
	// list A: [1, 2, 3]   list B: [2, 1, 4]
	// k=60. score(x) = Σ 1/(60+rank).
	//   1: 1/61 (A r1) + 1/62 (B r2)
	//   2: 1/62 (A r2) + 1/61 (B r1)   -> ties with 1
	//   3: 1/63 (A r3)
	//   4: 1/63 (B r3)
	got := RRF(60, []int{1, 2, 3}, []int{2, 1, 4})

	want := map[int]float64{
		1: 1.0/61 + 1.0/62,
		2: 1.0/62 + 1.0/61,
		3: 1.0 / 63,
		4: 1.0 / 63,
	}
	if len(got) != 4 {
		t.Fatalf("got %d results, want 4", len(got))
	}
	for _, r := range got {
		if !approx(r.Score, want[r.Key]) {
			t.Errorf("key %d: score %v want %v", r.Key, r.Score, want[r.Key])
		}
	}
	// 1 and 2 tie on score; 1 appears first (ranking A, rank2 vs B...),
	// actually 1 is first-seen at order 0 (A[0]), 2 at order 1 (A[1]).
	// So tie-break puts 1 before 2.
	if got[0].Key != 1 || got[1].Key != 2 {
		t.Errorf("tie order: got [%d,%d], want [1,2]", got[0].Key, got[1].Key)
	}
	// 3 was first-seen (order 2) before 4 (order 3); equal scores → 3 first.
	if got[2].Key != 3 || got[3].Key != 4 {
		t.Errorf("tail order: got [%d,%d], want [3,4]", got[2].Key, got[3].Key)
	}
}

// TestRRF_consensusWins: an item ranked modestly in BOTH lists should
// outrank an item ranked #1 in only one — the core value of fusion.
func TestRRF_consensusWins(t *testing.T) {
	// X is rank 2 in both; Y is rank 1 in A only.
	// X: 1/62 + 1/62 = 2/62 ≈ 0.03226
	// Y: 1/61        ≈ 0.01639
	got := RRF(60, []int{9, 5, 7}, []int{8, 5, 6})
	if got[0].Key != 5 {
		t.Fatalf("consensus item 5 should rank first, got %d", got[0].Key)
	}
}

// TestRRFWeighted: weighting one list up changes the winner.
func TestRRFWeighted(t *testing.T) {
	dense := []int{100, 200}   // dense likes 100
	lexical := []int{200, 100} // lexical likes 200
	// Unweighted: 100 and 200 tie; 100 first-seen → first.
	if got := RRF(60, dense, lexical); got[0].Key != 100 {
		t.Errorf("unweighted tie: want 100 first, got %d", got[0].Key)
	}
	// Weight lexical 3×: 200 should win.
	//  100: 1/61 + 3/62
	//  200: 1/62 + 3/61   (bigger)
	got := RRFWeighted(60, []float64{1, 3}, dense, lexical)
	if got[0].Key != 200 {
		t.Errorf("lexical-weighted: want 200 first, got %d", got[0].Key)
	}
}

// TestRRF_stringKeys: keys are generic.
func TestRRF_stringKeys(t *testing.T) {
	got := RRF(60, []string{"a", "b"}, []string{"b", "c"})
	// b is in both → wins.
	if got[0].Key != "b" {
		t.Errorf("want \"b\" first, got %q", got[0].Key)
	}
	if len(got) != 3 {
		t.Errorf("want 3 distinct keys, got %d", len(got))
	}
}

// TestRRF_singleAndEmpty: edge cases.
func TestRRF_singleAndEmpty(t *testing.T) {
	if got := RRF[int](60); len(got) != 0 {
		t.Errorf("no rankings → empty, got %d", len(got))
	}
	if got := RRF(60, []int{}); len(got) != 0 {
		t.Errorf("empty ranking → empty, got %d", len(got))
	}
	// Single ranking just reproduces its order (scores strictly decreasing).
	got := RRF(60, []int{7, 8, 9})
	if len(got) != 3 || got[0].Key != 7 || got[2].Key != 9 {
		t.Fatalf("single ranking should preserve order: %+v", got)
	}
	if !(got[0].Score > got[1].Score && got[1].Score > got[2].Score) {
		t.Errorf("single ranking scores should strictly decrease: %+v", got)
	}
}

func TestRRF_panics(t *testing.T) {
	assertPanic(t, "k<=0", func() { RRF(0, []int{1}) })
	assertPanic(t, "weight mismatch", func() { RRFWeighted(60, []float64{1}, []int{1}, []int{2}) })
}

func assertPanic(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s: expected panic", name)
		}
	}()
	fn()
}

// TestKeys: the projection helper.
func TestKeys(t *testing.T) {
	type hit struct {
		Index int
		Score float64
	}
	hits := []hit{{Index: 3}, {Index: 1}, {Index: 4}}
	ids := Keys(hits, func(h hit) int { return h.Index })
	want := []int{3, 1, 4}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("Keys[%d]=%d want %d", i, ids[i], want[i])
		}
	}
}
