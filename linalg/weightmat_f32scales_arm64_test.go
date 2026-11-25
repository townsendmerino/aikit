package linalg

// wrapInt4Row4OnlyF32 is WrapInt4(LayoutRow4) over f32 scales (see weightmat_f32scales_test.go).
func wrapInt4Row4OnlyF32(q4Row4 []byte, q4Row4Scales []float32, rows, cols, group int) (WeightMat, bool) {
	if !int4Row4Usable(rows, cols, group) {
		return WeightMat{}, false
	}
	return wrapInt4Row4Only(q4Row4, F32ToF16Scales(q4Row4Scales), rows, cols, group)
}
