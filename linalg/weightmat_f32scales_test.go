package linalg

// The int4 constructors over f32 scales, for tests that quantize with QuantizeGroupsInt4 (f32 scales) and
// wrap the result. Each converts the scales with F32ToF16Scales into a new slice, so the result does not
// alias them.

func wrapInt4F32(q4 []byte, q4s []float32, rows, cols, group int) WeightMat {
	checkWrapInt4("WrapInt4", q4, len(q4s), rows, cols, group)
	return wrapInt4Canonical(q4, F32ToF16Scales(q4s), rows, cols, group)
}

func wrapInt4SplitHalfOnlyF32(q4SplitHalf []byte, q4s []float32, rows, cols, group int) (WeightMat, bool) {
	if !int4SplitHalfUsable(cols, group) {
		return WeightMat{}, false
	}
	return wrapInt4SplitHalfOnly(q4SplitHalf, F32ToF16Scales(q4s), rows, cols, group)
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

// int4Widened and int4Row4Widened are the int4 accessors with the scales widened to f32 in a new slice, for tests
// written against f32 scales. Production reads WeightMat.Int4F16 and Int4Row4F16.
func (w *WeightMat) int4Widened() (q4 []byte, q4s []float32, group int, ok bool) {
	return w.q4, widenScalesCopy(w.q4s16), w.group, w.q4 != nil
}

func (w *WeightMat) int4Row4Widened() (packed4 []byte, scales4 []float32, ok bool) {
	return w.q4Row4, widenScalesCopy(w.q4Row4Scales16), w.q4Row4 != nil
}
