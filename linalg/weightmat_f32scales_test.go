package linalg

// The int4 constructors over f32 scales, for tests that quantize with QuantizeGroupsInt4 (f32 scales) and
// wrap the result. Each converts the scales with F32ToF16Scales into a new slice, so the result does not
// alias them.

func wrapInt4F32(q4 []byte, q4s []float32, rows, cols, group int) WeightMat {
	checkWrapInt4("WrapInt4", q4, len(q4s), rows, cols, group)
	return wrapInt4Canonical(q4, F32ToF16Scales(q4s), rows, cols, group)
}

// repackInt4Row4InPlaceF32 permutes q4 in place; the scales it returns are a binary16 copy of q4s, so q4s
// itself is left as it was.
func repackInt4Row4InPlaceF32(q4 []byte, q4s []float32, rows, cols, group int) (WeightMat, bool) {
	if !int4Row4Usable(rows, cols, group) {
		return WeightMat{}, false
	}
	return repackInt4Row4InPlace(q4, F32ToF16Scales(q4s), rows, cols, group)
}

// repackInt4SplitHalfInPlaceF32 is repackInt4Row4InPlaceF32's split-half twin.
func repackInt4SplitHalfInPlaceF32(q4 []byte, q4s []float32, rows, cols, group int) (WeightMat, bool) {
	if !int4SplitHalfUsable(cols, group) {
		return WeightMat{}, false
	}
	return repackInt4SplitHalfInPlace(q4, F32ToF16Scales(q4s), rows, cols, group)
}
