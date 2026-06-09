package linalg

// dotW4A8Scalar is the portable reference for one W4A8 output: the int8
// activation row dotted against a group-int4 weight row, each group's integer
// dot scaled by its f32 weight scale. The asm kernels (dotW4A8SDOT) must match
// this bit-for-bit on the integer accumulation; the per-group f32 fold may
// differ only in float rounding. group need not divide K (ragged final group).
func dotW4A8Scalar(act []int8, packed []byte, scales []float32, group, K int) float32 {
	var total float32
	nGroups := (K + group - 1) / group
	for g := 0; g < nGroups; g++ {
		ks := g * group
		ke := min(ks+group, K)
		var acc int32
		for k := ks; k < ke; k++ {
			b := packed[k>>1]
			nib := b & 0x0F
			if k&1 == 1 {
				nib = b >> 4
			}
			acc += int32(act[k]) * int32(int(nib)-8)
		}
		total += float32(acc) * scales[g]
	}
	return total
}
