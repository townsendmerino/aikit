//go:build amd64

package linalg

// dotW4A8Tile4RowAVX2 computes four W4A8 outputs in one call — four activation
// rows, actStride bytes apart, against ONE weight row — into dst[0..3] (before
// the activation scale, exactly as dotW4A8FoldAVX2 returns it). nGroups is K/32
// and any ragged final group is the caller's, as it already is for
// dotW4A8FoldAVX2. Only safe when hasAVX2. See dot_w4a8_tile_amd64.s.
//
//go:noescape
func dotW4A8Tile4RowAVX2(act *int8, actStride int, packed *byte, scales *float32, dst *float32, nGroups int)

// w4a8TileRows runs the AVX2 activation-blocked tile over the first M&^3
// activation rows for every column of this span, and returns how many rows it
// took. The leftover M%4 rows are the caller's.
//
// Column-outer, exactly like the span it replaces: for one weight row all four
// activation blocks are consumed before j advances, so B is still walked linearly
// and a weight row (K/2 bytes — 768 at K=1536) is re-read from L1 rather than
// from memory when M>4.
//
// THE hasAVX512VNNIVL EXCLUSION IS NOT REDUNDANT, and it is the subtle part.
// dotW4A8 prefers dotW4A8FoldAVX512VNNI over the AVX2 kernel where VNNI exists,
// and that kernel's int32 lane partials come from VPDPBUSD's 4-byte lanes where the
// AVX2 one's come from VPMADDWD pairs — a different summation order, so the two are
// not bit-identical (they agree to f32 ULP noise since the 2026-09-24 int32-centering
// fix, dot_w4a8_avx512vnni_amd64.go; before it they differed by far more). This tile is
// built on the AVX2 fold. Letting it run at M>1 on a VNNI host while M=1 kept the
// VNNI kernel would make the result depend on M, which is precisely the property
// TestMatmulBTW4A8_MConsistent exists to forbid and which goinfer's speculative
// verify relies on. Neither project box has VNNI, so this could not have been
// caught by measurement here; it is excluded by construction instead.
func w4a8TileRows(aq []int8, aScales []float32, w4 []byte, wScales, dst []float32, M, K, N, group, nGroups, bpr, j0, j1 int) int {
	if !hasAVX2 || hasAVX512VNNIVL || M < 4 || group != 32 || K < 32 {
		return 0
	}
	mFull := M &^ 3
	nFull := K / 32
	done := nFull * 32
	var out [4]float32
	for j := j0; j < j1; j++ {
		prow := w4[j*bpr : j*bpr+bpr]
		srow := wScales[j*nGroups : j*nGroups+nGroups]
		for i := 0; i < mFull; i += 4 {
			dotW4A8Tile4RowAVX2(&aq[i*K], K, &prow[0], &srow[0], &out[0], nFull)
			for m := range 4 {
				aScale := aScales[i+m]
				if aScale == 0 {
					dst[(i+m)*N+j] = 0
					continue
				}
				total := out[m]
				if done < K {
					// The ragged final group, mopped up exactly as dotW4A8 does
					// it after the AVX2 kernel — same nibble extraction, same
					// scales[nFull], added to the kernel's total in the same
					// place, so a K%32!=0 shape stays bit-identical too.
					arow := aq[(i+m)*K : (i+m)*K+K]
					var acc int32
					for k := done; k < K; k++ {
						b := prow[k>>1]
						nib := b & 0x0F
						if k&1 == 1 {
							nib = b >> 4
						}
						acc += int32(arow[k]) * int32(int(nib)-8)
					}
					total += float32(acc) * srow[nFull]
				}
				dst[(i+m)*N+j] = total * aScale
			}
		}
	}
	return mFull
}
