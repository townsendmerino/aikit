//go:build amd64

package linalg

// dotW4A8FoldAVX2 returns the per-group-scaled f32 dot Σ_g scale[g]·(act·w)_g of
// one int4 weight row against the int8 activation row, via the fused AVX2 kernel
// in dot_w4a8_amd64.s. The f32 weight scales are folded IN-REGISTER (convert +
// FMA into an 8-lane accumulator, one reduce at the end) — no per-group int32
// scratch and no Go-side fold loop. Only safe when hasAVX2. group is fixed at 32;
// nGroups = K/32.
//
//go:noescape
func dotW4A8FoldAVX2(act *int8, packed *byte, scales *float32, nGroups int) float32

// dotW4A8 computes one W4A8 output (before the activation scale). The AVX2 path
// folds the per-group weight scales inside the kernel and returns the f32 dot
// directly; only a ragged final group (K % 32 ≠ 0) is mopped up in Go.
// Everything off the fast path falls back to the portable reference.
func dotW4A8(act []int8, packed []byte, scales []float32, group, K int) float32 {
	if hasAVX2 && group == 32 && K >= 32 {
		nFull := K / 32
		total := dotW4A8FoldAVX2(&act[0], &packed[0], &scales[0], nFull)
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
