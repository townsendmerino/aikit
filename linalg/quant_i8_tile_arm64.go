//go:build arm64

package linalg

import "github.com/townsendmerino/aikit/internal/cpufeat"

// dotI8Tile4x4 computes 16 int8 dot products in one call — four activation rows
// (actStride bytes apart) against four weight rows (wStride bytes apart) — into
// dst[m*4+r] as int32. n must be a multiple of 16; the caller mops up the
// remainder, exactly as dotI8 does for dotI8SDOT. Implemented in
// dot_i8_tile_arm64.s; see there for why it beats dotI8SDOT and why bit-identity
// is free.
//
//go:noescape
func dotI8Tile4x4(act *int8, actStride int, w *int8, wStride int, dst *int32, n int)

// w8a8TileRect runs the register-blocked W8A8 tile (docs/task-simd-audit.md
// S-01b) over the largest 4-activation-row by 4-weight-row rectangle of this
// span, and reports how much it took: rows [0,mTiled) and columns [j0,jTiled)
// are done, everything else is the caller's to finish.
//
// Declining is normal and cheap. M<4 leaves no full row block (decode is M=1 and
// never comes here), a core without DotProd has no SDOT at all, and K<16 leaves
// the tile kernel nothing to chew — each returns (0, j0), which makes w8a8Span
// run precisely the loop it ran before this existed.
//
// The four weight rows a tile reads are CONSECUTIVE, so they are one contiguous
// 4*K-byte block of B rather than four far-apart streams, and the walk over j is
// still linear over B. That distinction is why this is worth trying at all: the
// deleted dotI8Cols8 (see w8a8Span's comment) advanced eight streams and lost
// badly once B stopped fitting in cache. Two things differ here — four streams
// inside one block, not eight, and four times the arithmetic per weight byte
// because M>=4 — but the lesson stands that this must be measured in BOTH the
// cache-resident and streamed regimes before it is believed, which is what
// TestW8A8TileVsSpanAB does.
func w8a8TileRect(aq []int8, aScales []float32, bQ []int8, bScales, dst []float32, M, K, N, j0, j1 int) (mTiled, jTiled int) {
	if !cpufeat.HasDotProd || M < 4 || K < 16 {
		return 0, j0
	}
	mFull := M &^ 3
	jFull := j0 + (j1-j0)&^3
	if jFull == j0 {
		return 0, j0
	}
	n16 := K &^ 15
	var tile [16]int32
	for j := j0; j < jFull; j += 4 {
		for i := 0; i < mFull; i += 4 {
			dotI8Tile4x4(&aq[i*K], K, &bQ[j*K], K, &tile[0], n16)
			for m := range 4 {
				aScale := aScales[i+m]
				arow := aq[(i+m)*K : (i+m)*K+K]
				for r := range 4 {
					if aScale == 0 {
						// w8a8Span's shortcut, kept exactly: a row that
						// quantized to all zeros stores a literal 0.
						dst[(i+m)*N+j+r] = 0
						continue
					}
					s := tile[m*4+r]
					// K%16 tail, the same scalar mop-up dotI8 does after
					// dotI8SDOT. Exact integer addition, so appending it to the
					// kernel's partial gives the identical int32.
					for k := n16; k < K; k++ {
						s += int32(arow[k]) * int32(bQ[(j+r)*K+k])
					}
					dst[(i+m)*N+j+r] = float32(s) * aScale * bScales[j+r]
				}
			}
		}
	}
	return mFull, jFull
}
