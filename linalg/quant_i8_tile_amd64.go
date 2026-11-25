//go:build amd64

package linalg

import "github.com/townsendmerino/aikit/internal/cpufeat"

// dotI8Tile4x1AVX2 computes 4 int8 dot products in one call — four activation
// rows (actStride bytes apart) against ONE weight row — into dst[0..3] as int32.
// n must be a multiple of 16; the caller mops up the remainder, as dotI8 already
// does for dotI8AVX2. See dot_i8_tile_amd64.s for why the weight dimension is
// deliberately NOT blocked here, which is a measured result and not a preference.
//
//go:noescape
func dotI8Tile4x1AVX2(act *int8, actStride int, w *int8, dst *int32, n int)

// w8a8TileRect runs the AVX2 W8A8 tile over the first M&^3 activation rows for
// EVERY column of this span, so it returns (mFull, j1): there is no column
// remainder, because the tile is one weight row wide by design.
//
// Declining is normal: M<4 leaves no full activation block (decode is M=1 and
// never reaches here) and a core without AVX2 has no kernel.
//
// VNNI hosts are excluded, and unlike the W4A8 tile's exclusion this is a
// PERFORMANCE judgement rather than a correctness one — W8A8 accumulates in
// int32, so every tier of dotI8 already returns the identical integer and no
// M-dependence is possible. The reason to decline is that dotI8AVX512VNNI does
// four times the MACs per instruction, so an AVX2 tile would very likely be a
// regression there, and neither project box has VNNI to measure it on. Building
// the VNNI tile is S-08.3's "file, do not build" case: it needs a host first.
func w8a8TileRect(aq []int8, aScales []float32, bQ []int8, bScales, dst []float32, M, K, N, j0, j1 int) (mTiled, jTiled int) {
	if !cpufeat.HasAVX2 || cpufeat.HasAVX512VNNI || M < 4 || K < 16 {
		return 0, j0
	}
	mFull := M &^ 3
	n16 := K &^ 15
	var tile [4]int32
	// Column-outer, one weight row at a time: B is walked exactly as the pre-tile
	// span walks it. Only the activation dimension is blocked.
	for j := j0; j < j1; j++ {
		bj := bQ[j*K : j*K+K]
		bScale := bScales[j]
		for i := 0; i < mFull; i += 4 {
			dotI8Tile4x1AVX2(&aq[i*K], K, &bj[0], &tile[0], n16)
			for m := range 4 {
				aScale := aScales[i+m]
				if aScale == 0 {
					dst[(i+m)*N+j] = 0
					continue
				}
				s := tile[m]
				for k := n16; k < K; k++ {
					s += int32(aq[(i+m)*K+k]) * int32(bj[k])
				}
				dst[(i+m)*N+j] = float32(s) * aScale * bScale
			}
		}
	}
	// Every column of the span is done for the tiled rows.
	return mFull, j1
}
