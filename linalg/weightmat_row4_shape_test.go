//go:build arm64

package linalg

import (
	"testing"

	"github.com/townsendmerino/aikit/internal/cpufeat"
)

// TestWrapInt4Row4_declinesBadShapeAtWrap covers task-simd-audit.md S-10's hardening item for the door a
// prebuilt row4 blob comes through: a blob with the right byte count and a shape the row4 kernel cannot
// express (rows%4 != 0, a ragged final group) must be refused at the wrap, which names the blob, and not
// reach MatmulBTW4A8Row4Into, which would panic far from it. WrapInt4 reports it as ok=false, the same
// answer PreferredInt4Layout gives for that shape.
func TestWrapInt4Row4_declinesBadShapeAtWrap(t *testing.T) {
	if !cpufeat.HasDotProd {
		t.Skip("DotProd required: without it the row4 layout is declined for every shape")
	}
	const group = 32
	cases := []struct {
		name       string
		rows, cols int
	}{
		{"rows not a multiple of 4", 6, 64},     // the 4-row interleave cannot express it
		{"cols not a multiple of group", 8, 48}, // 48 % 32 != 0: a ragged final group
		{"both wrong", 5, 40},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			// Lengths from the caller's own rows/cols, so only the shape is wrong.
			nGroups, bpr := groupsFor(c.cols, group)
			if got := PreferredInt4Layout(c.rows, c.cols, group); got != LayoutCanonical {
				t.Fatalf("PreferredInt4Layout(%d, %d, %d) = %v, want canonical", c.rows, c.cols, group, got)
			}
			w, ok := WrapInt4(make([]byte, c.rows*bpr), make([]uint16, c.rows*nGroups), c.rows, c.cols, group, LayoutRow4)
			if ok || w.IsInt4() {
				t.Fatalf("rows=%d cols=%d: WrapInt4(LayoutRow4) accepted a shape the row4 kernel cannot express", c.rows, c.cols)
			}
		})
	}
}

// TestWrapInt4Row4_acceptsValidShape is the other half: the gate must not refuse what the row4 kernel
// handles, or it would silently disable the layout it protects.
func TestWrapInt4Row4_acceptsValidShape(t *testing.T) {
	if !cpufeat.HasDotProd {
		t.Skip("DotProd required: without it the row4 layout is declined for every shape")
	}
	const group, rows, cols = 32, 8, 64
	nGroups, bpr := groupsFor(cols, group)
	if got := PreferredInt4Layout(rows, cols, group); got != LayoutRow4 {
		t.Fatalf("PreferredInt4Layout = %v, want row4", got)
	}
	w, ok := WrapInt4(make([]byte, rows*bpr), make([]uint16, rows*nGroups), rows, cols, group, LayoutRow4)
	if !ok || w.rows != rows || w.cols != cols || w.q4Row4 == nil {
		t.Fatalf("valid shape on a row4-capable host: ok=%v rows=%d cols=%d row4=%v", ok, w.rows, w.cols, w.q4Row4 != nil)
	}
}
