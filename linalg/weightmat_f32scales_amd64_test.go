package linalg

// wrapInt4SplitHalfOnlyF32 is WrapInt4(LayoutSplitHalf) over f32 scales (see weightmat_f32scales_test.go).
func wrapInt4SplitHalfOnlyF32(q4SplitHalf []byte, q4s []float32, rows, cols, group int) (WeightMat, bool) {
	if !int4SplitHalfUsable(cols, group) {
		return WeightMat{}, false
	}
	return wrapInt4SplitHalfOnly(q4SplitHalf, F32ToF16Scales(q4s), rows, cols, group)
}
