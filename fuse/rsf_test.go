package fuse

import (
	"math"
	"testing"
)

func keyOrder[K comparable](rs []Result[K]) []K {
	out := make([]K, len(rs))
	for i, r := range rs {
		out[i] = r.Key
	}
	return out
}

func TestRSF_normalizesAndSums(t *testing.T) {
	// A: x=0.9 y=0.5 z=0.1 → span 0.8 → x=1.0 y=0.5 z=0.0
	// B: y=10  z=6   w=2   → span 8   → y=1.0 z=0.5 w=0.0
	a := []Scored[string]{{"x", 0.9}, {"y", 0.5}, {"z", 0.1}}
	b := []Scored[string]{{"y", 10}, {"z", 6}, {"w", 2}}
	got := RSF(a, b)

	want := map[string]float64{"y": 1.5, "x": 1.0, "z": 0.5, "w": 0.0}
	for _, r := range got {
		if math.Abs(r.Score-want[r.Key]) > 1e-9 {
			t.Errorf("%s fused = %.6f, want %.6f", r.Key, r.Score, want[r.Key])
		}
	}
	if order := keyOrder(got); !equalK(order, []string{"y", "x", "z", "w"}) {
		t.Errorf("order = %v, want [y x z w]", order)
	}
}

func TestRSF_weighted(t *testing.T) {
	a := []Scored[string]{{"x", 1}, {"y", 0}} // norms x=1, y=0
	b := []Scored[string]{{"y", 1}, {"x", 0}} // norms y=1, x=0
	// weight B 3×: x=1*1+0*3=1, y=0*1+1*3=3 → y first.
	got := RSFWeighted([]float64{1, 3}, a, b)
	if got[0].Key != "y" || math.Abs(got[0].Score-3) > 1e-9 {
		t.Errorf("weighted top = %+v, want y score 3", got[0])
	}
}

func TestRSF_allEqualAndSingle(t *testing.T) {
	// All-equal scores → every member normalizes to 1.0 (order, no within-signal).
	got := RSF([]Scored[int]{{1, 5}, {2, 5}, {3, 5}})
	for _, r := range got {
		if r.Score != 1.0 {
			t.Errorf("all-equal: key %d score %.3f, want 1.0", r.Key, r.Score)
		}
	}
	// Single-item ranking → that item is 1.0.
	if s := RSF([]Scored[int]{{9, 0.3}}); len(s) != 1 || s[0].Score != 1.0 {
		t.Errorf("single-item = %+v, want one item score 1.0", s)
	}
}

func TestRSF_negativeScores(t *testing.T) {
	// Cosine-like negatives: min→0, max→1.
	got := RSF([]Scored[int]{{1, -0.5}, {2, 0.5}, {3, 0.0}})
	want := map[int]float64{1: 0.0, 2: 1.0, 3: 0.5}
	for _, r := range got {
		if math.Abs(r.Score-want[r.Key]) > 1e-9 {
			t.Errorf("key %d = %.4f, want %.4f", r.Key, r.Score, want[r.Key])
		}
	}
}

func TestRSF_emptyAndMismatchedWeights(t *testing.T) {
	if got := RSF[int](); len(got) != 0 {
		t.Errorf("no rankings: got %d", len(got))
	}
	if got := RSF([]Scored[int]{}, []Scored[int]{{1, 1}}); len(got) != 1 {
		t.Errorf("empty + one: got %d, want 1", len(got))
	}
	defer func() {
		if recover() == nil {
			t.Error("mismatched weights should panic")
		}
	}()
	RSFWeighted([]float64{1}, []Scored[int]{{1, 1}}, []Scored[int]{{2, 2}})
}

func TestScores_helper(t *testing.T) {
	type hit struct {
		id int
		s  float64
	}
	hits := []hit{{7, 0.9}, {3, 0.4}}
	got := Scores(hits, func(h hit) int { return h.id }, func(h hit) float64 { return h.s })
	if len(got) != 2 || got[0] != (Scored[int]{7, 0.9}) || got[1] != (Scored[int]{3, 0.4}) {
		t.Errorf("Scores = %+v", got)
	}
}

func equalK[K comparable](a, b []K) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
