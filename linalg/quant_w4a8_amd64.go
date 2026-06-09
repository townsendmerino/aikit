//go:build amd64

package linalg

// dotW4A8GroupsAVX2 fills out[0:nGroups] with the int32 dot of each 32-wide group
// (int8 activation · centered-int4 weight) via the fused AVX2 kernel in
// dot_w4a8_amd64.s. Only safe when hasAVX2 (the kernel uses AVX2). group is fixed
// at 32; nGroups = K/32.
//
//go:noescape
func dotW4A8GroupsAVX2(act *int8, packed *byte, out *int32, nGroups int)

// dotW4A8 computes one W4A8 output (before the activation scale). With AVX2 and
// the group-32 layout the fused kernel emits the per-group int32 dots into sums
// (caller-owned scratch, len ≥ K/32) and Go folds in the f32 weight scales; a
// scalar tail mops up any ragged final group. Mirrors the arm64 dispatch
// (quant_w4a8_arm64.go); everything else falls back to the portable reference.
func dotW4A8(act []int8, packed []byte, scales []float32, group, K int, sums []int32) float32 {
	if hasAVX2 && group == 32 && K >= 32 {
		nFull := K / 32
		dotW4A8GroupsAVX2(&act[0], &packed[0], &sums[0], nFull)
		var total float32
		for g := 0; g < nFull; g++ {
			total += float32(sums[g]) * scales[g]
		}
		if done := nFull * 32; done < K {
			// Ragged final group (K not a multiple of 32): scalar, scales[nFull].
			var acc int32
			for k := done; k < K; k++ {
				b := packed[k>>1]
				nib := b & 0x0F
				if k&1 == 1 {
					nib = b >> 4
				}
				acc += int32(act[k]) * int32(int(nib)-8)
			}
			total += float32(acc) * scales[nFull]
		}
		return total
	}
	return dotW4A8Scalar(act, packed, scales, group, K)
}
