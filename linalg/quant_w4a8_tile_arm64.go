//go:build arm64

package linalg

// dotW4A8Tile4RowSDOT computes four W4A8 outputs — four activation rows,
// actStride bytes apart, against ONE weight row — into dst[0..3], before the
// activation scale, exactly as dotW4A8FoldSDOT returns it. nGroups is K/32; any
// ragged final group is the caller's, as it already is for dotW4A8FoldSDOT.
// Only safe when hasDotProd. See dot_w4a8_tile4row_arm64.s.
//
//go:noescape
func dotW4A8Tile4RowSDOT(act *int8, actStride int, packed *byte, scales *float32, dst *float32, nGroups int)

// w4a8TileRows runs the NEON activation-blocked tile over the first M&^3
// activation rows for every column of this span, and returns how many rows it
// took. The leftover M%4 rows are the caller's.
//
// WHY THIS EXISTS (audit M-02). amd64 got a layout-free 4-row tile inside the
// canonical span in S-01 (1.65-1.90x); arm64's half of S-01 was scoped to the
// ROW4 layout and reached only through WeightMat.MatmulBTW4A8Into, so every
// caller that does not hold a repacked tensor — paged MoE experts, any tensor
// that declined the repack (K%32, N%4), MatmulBTW4A8Batch at M>1, and the
// uniform WeightMat.MatmulBTInto — kept running the M=1 GEMV once per
// (row, column) pair at the recorded 24.1 GMAC/s against the tile's 69.5. This
// closes that gap on the canonical layout, where no repack is needed.
//
// Column-outer, exactly like the span it replaces and like the amd64 twin: for
// one weight row all four activation blocks are consumed before j advances, so
// B is still walked linearly and the weight row is re-read from L1 rather than
// from memory when M>4.
//
// NO VNNI-STYLE EXCLUSION IS NEEDED HERE, unlike amd64. That exclusion exists
// because dotW4A8 prefers a VNNI kernel with a different fold at M=1, so a tile
// built on the AVX2 fold would make the result depend on M. On arm64 there is
// one canonical kernel — dotW4A8FoldSDOT — and this tile reproduces its
// instruction sequence exactly, so M-invariance holds by construction.
// hasDotProd is still required: SDOT is ARMv8.2 FEAT_DotProd.
func w4a8TileRows(aq []int8, aScales []float32, w4 []byte, wScales, dst []float32, M, K, N, group, nGroups, bpr, j0, j1 int) int {
	if !hasDotProd || M < 4 || group != 32 || K < 32 {
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
			dotW4A8Tile4RowSDOT(&aq[i*K], K, &prow[0], &srow[0], &out[0], nFull)
			for m := range 4 {
				aScale := aScales[i+m]
				if aScale == 0 {
					dst[(i+m)*N+j] = 0
					continue
				}
				total := out[m]
				if done < K {
					// The ragged final group, mopped up exactly as dotW4A8 does
					// it after the SDOT kernel — same nibble extraction, same
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
