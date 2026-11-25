package linalg

import "fmt"

// Layout is the order an int4 weight's nibbles and scales are stored in. Scales are binary16 in every layout.
type Layout uint8

const (
	// LayoutCanonical is row-major packed nibbles (two a byte, ⌈cols/2⌉ bytes a row) and row-major per-group
	// scales (⌈cols/group⌉ a row): QuantizeInt4's order, and the one every GPU backend reads.
	LayoutCanonical Layout = iota
	// LayoutRow4 is the arm64 DotProd kernel's: each group split in halves and four rows interleaved, in
	// RepackW4A8Row4's and RepackW4A8Row4ScalesF16's output order. It needs rows%4 == 0, group 32 and
	// cols%group == 0.
	LayoutRow4
	// LayoutSplitHalf is the amd64 AVX2 kernel's: each 32-column group's nibbles split in halves
	// (RepackW4A8SplitHalf's order), the scales canonical. It needs group 32 and cols%32 == 0.
	LayoutSplitHalf
)

// String returns "canonical", "row4" or "splithalf", the names WeightMat.Int4Layout reports.
func (l Layout) String() string {
	switch l {
	case LayoutCanonical:
		return "canonical"
	case LayoutRow4:
		return "row4"
	case LayoutSplitHalf:
		return "splithalf"
	}
	return fmt.Sprintf("Layout(%d)", uint8(l))
}

// PreferredInt4Layout reports the layout this CPU's int4 kernel runs for a [rows, cols] weight in groups of
// group: LayoutRow4 on arm64 with DotProd, LayoutSplitHalf on amd64 with AVX2 and without AVX-512 VNNI, each
// when the shape qualifies, and LayoutCanonical otherwise. WrapInt4 and RepackInt4InPlace take a repacked
// layout only when it is this one.
func PreferredInt4Layout(rows, cols, group int) Layout {
	switch {
	case int4Row4Usable(rows, cols, group):
		return LayoutRow4
	case int4SplitHalfUsable(cols, group):
		return LayoutSplitHalf
	}
	return LayoutCanonical
}

// WrapInt4 wraps already-quantized int4 weights stored in layout, aliasing q and scales16 (binary16 bit
// patterns, F32ToF16's) without copying, so the caller keeps them alive; they may point into a file mapping.
// q is rows·⌈cols/2⌉ bytes and scales16 rows·⌈cols/group⌉ scales in every layout.
//
// ok is false, and nothing is retained, when layout is a repacked one this CPU's kernel cannot run for this
// shape (PreferredInt4Layout(rows, cols, group) != layout): a repacked weight has no canonical bytes to fall
// back to. LayoutCanonical is always ok.
//
// Panics if rows or cols is negative, group <= 0, a length does not match, or layout is not one of the three.
func WrapInt4(q []byte, scales16 []uint16, rows, cols, group int, layout Layout) (WeightMat, bool) {
	switch layout {
	case LayoutCanonical:
		return wrapInt4Canonical(q, scales16, rows, cols, group), true
	case LayoutRow4:
		return wrapInt4Row4Only(q, scales16, rows, cols, group)
	case LayoutSplitHalf:
		return wrapInt4SplitHalfOnly(q, scales16, rows, cols, group)
	}
	panic(fmt.Sprintf("linalg: WrapInt4: unknown layout %v", layout))
}

// RepackInt4InPlace permutes canonical int4 weights, in the caller's own q and scales16, into layout and wraps
// them, so the canonical bytes never coexist with the repacked ones (the scratch is one row, or for
// LayoutRow4 one quad of rows). The input is consumed: after ok, q and scales16 hold layout's order and
// belong to the result. LayoutCanonical wraps them unchanged.
//
// ok is false, and q and scales16 are untouched, when PreferredInt4Layout(rows, cols, group) != layout.
//
// Panics on the shapes and lengths WrapInt4 panics on.
//
// Output contract: rewrites q and scales16 in place; there is no separate destination, and nothing needs zeroing.
func RepackInt4InPlace(q []byte, scales16 []uint16, rows, cols, group int, layout Layout) (WeightMat, bool) {
	switch layout {
	case LayoutCanonical:
		return wrapInt4Canonical(q, scales16, rows, cols, group), true
	case LayoutRow4:
		return repackInt4Row4InPlace(q, scales16, rows, cols, group)
	case LayoutSplitHalf:
		return repackInt4SplitHalfInPlace(q, scales16, rows, cols, group)
	}
	panic(fmt.Sprintf("linalg: RepackInt4InPlace: unknown layout %v", layout))
}
