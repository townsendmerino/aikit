package linalg

import (
	"math/rand"
	"slices"
	"strings"
	"testing"
)

func TestLayout_String(t *testing.T) {
	for l, want := range map[Layout]string{
		LayoutCanonical: "canonical", LayoutRow4: "row4", LayoutSplitHalf: "splithalf", Layout(9): "Layout(9)",
	} {
		if got := l.String(); got != want {
			t.Errorf("Layout(%d).String() = %q, want %q", uint8(l), got, want)
		}
	}
}

// TestPreferredInt4Layout_followsTheCPU holds PreferredInt4Layout to the kernels' CPU gates (row4Usable:
// arm64 DotProd; splitHalfUsable: amd64 AVX2 without AVX-512 VNNI) for a shape every repacked layout
// accepts, and to canonical for shapes none does.
func TestPreferredInt4Layout_followsTheCPU(t *testing.T) {
	want := LayoutCanonical
	switch {
	case row4Usable():
		want = LayoutRow4
	case splitHalfUsable():
		want = LayoutSplitHalf
	}
	if row4Usable() && splitHalfUsable() {
		t.Fatal("row4Usable and splitHalfUsable both true: the two layouts are for different arches")
	}
	if got := PreferredInt4Layout(8, 64, 32); got != want {
		t.Errorf("PreferredInt4Layout(8, 64, 32) = %v, want %v", got, want)
	}
	for _, s := range []struct{ rows, cols, group int }{{8, 64, 16}, {8, 48, 32}} {
		if got := PreferredInt4Layout(s.rows, s.cols, s.group); got != LayoutCanonical {
			t.Errorf("PreferredInt4Layout(%d, %d, %d) = %v, want canonical", s.rows, s.cols, s.group, got)
		}
	}
	if want == LayoutRow4 {
		if got := PreferredInt4Layout(6, 64, 32); got != LayoutCanonical {
			t.Errorf("PreferredInt4Layout(6, 64, 32) = %v, want canonical (rows%%4 != 0)", got)
		}
	}
}

func int4Fixture(t *testing.T, rows, cols, group int) (q []byte, s16 []uint16, a []float32) {
	t.Helper()
	rng := rand.New(rand.NewSource(int64(rows*cols + group)))
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	q4, q4s := QuantizeGroupsInt4(w, rows, cols, group)
	a = make([]float32, 3*cols)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	return q4, F32ToF16Scales(q4s), a
}

// TestWrapInt4_everyLayoutMatchesCanonical builds the same weight four ways (canonical; repacked in place to
// the preferred layout; that layout's bytes wrapped by WrapInt4) and holds every matmul result bit-equal.
func TestWrapInt4_everyLayoutMatchesCanonical(t *testing.T) {
	const rows, cols, group, M = 8, 64, 32, 3
	q, s16, a := int4Fixture(t, rows, cols, group)
	canon, ok := WrapInt4(q, s16, rows, cols, group, LayoutCanonical)
	if !ok || canon.Int4Layout() != "canonical" {
		t.Fatalf("WrapInt4(canonical): ok=%v layout=%q", ok, canon.Int4Layout())
	}
	var ws Workspace
	want := make([]float32, M*rows)
	canon.MatmulBTInto(&ws, a, want, M)

	pref := PreferredInt4Layout(rows, cols, group)
	qIn, sIn := slices.Clone(q), slices.Clone(s16)
	inPlace, ok := RepackInt4InPlace(qIn, sIn, rows, cols, group, pref)
	if !ok || inPlace.Int4Layout() != pref.String() {
		t.Fatalf("RepackInt4InPlace(%v): ok=%v layout=%q", pref, ok, inPlace.Int4Layout())
	}
	wrapped, ok := WrapInt4(slices.Clone(qIn), slices.Clone(sIn), rows, cols, group, pref)
	if !ok || wrapped.Int4Layout() != pref.String() {
		t.Fatalf("WrapInt4(%v) of the repacked bytes: ok=%v layout=%q", pref, ok, wrapped.Int4Layout())
	}
	for name, w := range map[string]WeightMat{"RepackInt4InPlace": inPlace, "WrapInt4": wrapped} {
		got := make([]float32, M*rows)
		w.MatmulBTInto(&ws, a, got, M)
		if !slices.Equal(got, want) {
			t.Errorf("%s(%v): matmul differs from the canonical weight's", name, pref)
		}
	}
}

// TestWrapInt4_declinesALayoutThisCPUDoesNotRun: a repacked layout other than the preferred one is refused,
// retains nothing, and RepackInt4InPlace leaves the caller's bytes as they were.
func TestWrapInt4_declinesALayoutThisCPUDoesNotRun(t *testing.T) {
	const rows, cols, group = 8, 64, 32
	q, s16, _ := int4Fixture(t, rows, cols, group)
	pref := PreferredInt4Layout(rows, cols, group)
	for _, l := range []Layout{LayoutRow4, LayoutSplitHalf} {
		if l == pref {
			continue
		}
		if w, ok := WrapInt4(q, s16, rows, cols, group, l); ok || w.IsInt4() {
			t.Errorf("WrapInt4(%v) on a CPU that prefers %v: ok=%v int4=%v, want declined", l, pref, ok, w.IsInt4())
		}
		qIn, sIn := slices.Clone(q), slices.Clone(s16)
		if w, ok := RepackInt4InPlace(qIn, sIn, rows, cols, group, l); ok || w.IsInt4() {
			t.Errorf("RepackInt4InPlace(%v) on a CPU that prefers %v: ok=%v, want declined", l, pref, ok)
		}
		if !slices.Equal(qIn, q) || !slices.Equal(sIn, s16) {
			t.Errorf("RepackInt4InPlace(%v) declined but wrote to its input", l)
		}
	}
}

func TestWrapInt4_panicsOnAnUnknownLayoutAndABadLength(t *testing.T) {
	const rows, cols, group = 8, 64, 32
	q, s16, _ := int4Fixture(t, rows, cols, group)
	for name, call := range map[string]func(){
		"WrapInt4 unknown layout":          func() { WrapInt4(q, s16, rows, cols, group, Layout(7)) },
		"RepackInt4InPlace unknown layout": func() { RepackInt4InPlace(q, s16, rows, cols, group, Layout(7)) },
		"WrapInt4 short q":                 func() { WrapInt4(q[1:], s16, rows, cols, group, LayoutCanonical) },
	} {
		func() {
			defer func() {
				r := recover()
				if r == nil {
					t.Errorf("%s: no panic", name)
					return
				}
				if msg, _ := r.(string); !strings.Contains(msg, "WrapInt4") && !strings.Contains(msg, "RepackInt4InPlace") {
					t.Errorf("%s: the panic should name the constructor; got %v", name, r)
				}
			}()
			call()
		}()
	}
}
