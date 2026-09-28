package linalg

// f16RoundScales rounds scales in place to binary16-representable values (F32ToF16, then widened), so an
// f32-scale reference and a WeightMat (which stores binary16) hold the same scale values.
func f16RoundScales(s []float32) []float32 {
	for i := range s {
		s[i] = F16ToF32(F32ToF16(s[i]))
	}
	return s
}
