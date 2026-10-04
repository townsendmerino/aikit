//go:build arm64

package linalg

import (
	"strings"
	"testing"
)

// TestWrapInt4Row4_rejectsBadShapeAtWrap covers task-simd-audit.md S-10's hardening
// item: WrapInt4Row4 validated the byte LENGTHS of the row4 arrays but not the SHAPE
// they imply, so a blob with the right count and the wrong shape was stored and
// panicked later, inside MatmulBTW4A8Row4Into, on the first M=1 matmul.
//
// The distinction this test turns on is WHERE it fails, not whether. A panic at the
// matmul names a kernel; a panic at the wrap names the blob. So each case asserts the
// message identifies WrapInt4Row4 and reports the offending dimensions — a test that
// only caught "it panics" would pass against the old behaviour too.
func TestWrapInt4Row4_rejectsBadShapeAtWrap(t *testing.T) {
	if !hasDotProd {
		t.Skip("DotProd required: the row4 layout declines without it (WrapInt4Row4Only reports false, WrapInt4Row4 returns the plain int4 matrix)")
	}
	const group = 32
	// Lengths are computed from the caller's OWN rows/cols so requireExactLen is
	// satisfied. That is the whole point: the old code got this far and stored it.
	mk := func(rows, cols int) (q4 []byte, q4s []float32, r4 []byte, r4s []float32) {
		nGroups, bpr := groupsFor(cols, group)
		return make([]byte, rows*bpr), make([]float32, rows*nGroups),
			make([]byte, rows*bpr), make([]float32, rows*nGroups)
	}

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
			q4, q4s, r4, r4s := mk(c.rows, c.cols)
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("rows=%d cols=%d: want a panic at WrapInt4Row4, got none — "+
						"the blob would be stored and panic later in the matmul instead", c.rows, c.cols)
				}
				msg, _ := r.(string)
				if !strings.Contains(msg, "WrapInt4Row4") {
					t.Errorf("panic should name WrapInt4Row4 (it names the blob, not a kernel); got: %v", r)
				}
				if !strings.Contains(msg, "rows=") || !strings.Contains(msg, "group=") {
					t.Errorf("panic should report the offending dimensions; got: %v", r)
				}
			}()
			WrapInt4Row4(q4, q4s, c.rows, c.cols, group, r4, r4s)
		})
	}
}

// TestWrapInt4Row4_acceptsValidShape is the other half: the guard must not reject
// what the row4 kernel can actually handle, or it would silently disable the layout
// it exists to protect.
func TestWrapInt4Row4_acceptsValidShape(t *testing.T) {
	const group, rows, cols = 32, 8, 64
	nGroups, bpr := groupsFor(cols, group)
	w := WrapInt4Row4(
		make([]byte, rows*bpr), make([]float32, rows*nGroups),
		rows, cols, group,
		make([]byte, rows*bpr), make([]float32, rows*nGroups),
	)
	if w.rows != rows || w.cols != cols {
		t.Fatalf("wrapped dims rows=%d cols=%d, want %d/%d", w.rows, w.cols, rows, cols)
	}
	// The row4 arrays are only retained when the host can actually use them
	// (row4Usable gates on DotProd); on a host that cannot, the canonical wrap is
	// the correct outcome and this assertion would be wrong to make.
	if row4Usable() && w.q4Row4 == nil {
		t.Error("valid shape on a row4-capable host: the row4 layout was dropped")
	}
}

// TestWrapInt4Row4_nilArraysStillCanonical pins that the shape guard did not change
// the documented escape hatch: passing nil row4 arrays means "no row4 layout", and
// must stay legal for ANY shape — including shapes the row4 kernel cannot express.
// Guarding before that early return would have broken every canonical caller.
func TestWrapInt4Row4_nilArraysStillCanonical(t *testing.T) {
	const group, rows, cols = 32, 6, 48 // deliberately a shape the guard rejects
	nGroups, bpr := groupsFor(cols, group)
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("nil row4 arrays must be legal at any shape, got panic: %v", r)
		}
	}()
	w := WrapInt4Row4(make([]byte, rows*bpr), make([]float32, rows*nGroups), rows, cols, group, nil, nil)
	if w.q4Row4 != nil {
		t.Error("nil row4 arrays should leave the canonical wrap untouched")
	}
}
