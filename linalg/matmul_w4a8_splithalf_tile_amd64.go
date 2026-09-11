//go:build amd64

package linalg

import "fmt"

// dotW4A8SplitHalfTile4RowAVX2 computes four W4A8 outputs in one call — four
// activation rows, actStride bytes apart, against ONE split-half-packed
// weight row — into dst[0..3], the split-half twin of dotW4A8Tile4RowAVX2.
// Only safe when hasAVX2. See dot_w4a8_splithalf_tile_amd64.s.
//
//go:noescape
func dotW4A8SplitHalfTile4RowAVX2(act *int8, actStride int, packed *byte, scales *float32, dst *float32, nGroups int)

// matmulBTW4A8SplitHalfMultiInto is matmulBTW4A8SplitHalfInto's M>1 sibling
// (audit M-22 follow-up: split-half M>1, the prerequisite for a split-half-
// only WeightMat to serve prefill). Structured exactly like
// MatmulBTW4A8Into/w4a8Span: quantize all M activation rows once, then for
// each column span run the tile over the M&^3 tile-eligible rows and the
// per-row split-half kernel over the M%4 remainder — the SAME shape
// w4a8Span uses for canonical, just with w4a8TileRows/w4a8SpanRows'
// split-half twins.
//
// K%32==0 is a hard guard, not an assumption: unlike the canonical kernel,
// neither the tile nor the M=1 split-half kernel has a ragged-tail path
// (see matmulBTW4A8SplitHalfInto's own comment) — a split-half-only
// WeightMat can only exist when Int4SplitHalfUsable already enforced this,
// but a future caller could reach this function directly, so it is checked
// again here rather than assumed.
func matmulBTW4A8SplitHalfMultiInto(ws *Workspace, a []float32, w4sh []byte, wScales, dst []float32, M, K, N, group int) {
	checkMatmulW4A8("MatmulBTW4A8SplitHalfMulti", len(a), len(w4sh), len(wScales), len(dst), M, K, N, group)
	if K%32 != 0 {
		panic(fmt.Sprintf("linalg: matmulBTW4A8SplitHalfMultiInto requires K a multiple of 32, got %d", K))
	}
	nGroups, bpr := groupsFor(K, group)
	aq := ws.int8Buf(M * K)
	aScales := ws.f32Buf(M)
	quantizeRowsInto(aq, aScales, a, M, K, ws.width)
	if M*N*K < ws.thr() || N < 2 {
		w4a8SplitHalfSpanMulti(aq, aScales, w4sh, wScales, dst, M, K, N, nGroups, bpr, 0, N)
		return
	}
	ws.parallel(N, func(j0, j1 int) {
		w4a8SplitHalfSpanMulti(aq, aScales, w4sh, wScales, dst, M, K, N, nGroups, bpr, j0, j1)
	})
}

// w4a8SplitHalfSpanMulti computes output columns [j0,j1) for all M rows,
// given the already-quantized activations — w4a8Span's split-half twin.
func w4a8SplitHalfSpanMulti(aq []int8, aScales []float32, w4sh []byte, wScales, dst []float32, M, K, N, nGroups, bpr, j0, j1 int) {
	mTiled := w4a8SplitHalfTileRows(aq, aScales, w4sh, wScales, dst, M, K, N, nGroups, bpr, j0, j1)
	if mTiled < M {
		w4a8SplitHalfSpanRows(aq, aScales, w4sh, wScales, dst, K, N, nGroups, bpr, mTiled, M, j0, j1)
	}
}

// w4a8SplitHalfTileRows is w4a8TileRows' split-half twin: runs the AVX2
// split-half tile over the first M&^3 activation rows for every column of
// this span, returning how many rows it took. The leftover M%4 rows are
// the caller's (w4a8SplitHalfSpanRows below).
//
// The hasAVX512VNNIVL exclusion mirrors w4a8TileRows' own for the same
// reason (M-04/M-consistency): a VNNI host runs the canonical dispatch at
// every M (splitHalfUsable() is false there — see weightmat_splithalf_
// amd64.go — so a split-half-only WeightMat cannot exist on such a host at
// all), but this function is checked defensively rather than assumed safe,
// the same posture w4a8TileRows takes.
func w4a8SplitHalfTileRows(aq []int8, aScales []float32, w4sh []byte, wScales, dst []float32, M, K, N, nGroups, bpr, j0, j1 int) int {
	if !hasAVX2 || hasAVX512VNNIVL || M < 4 || K < 32 {
		return 0
	}
	mFull := M &^ 3
	nFull := K / 32
	var out [4]float32
	for j := j0; j < j1; j++ {
		prow := w4sh[j*bpr : j*bpr+bpr]
		srow := wScales[j*nGroups : j*nGroups+nGroups]
		for i := 0; i < mFull; i += 4 {
			dotW4A8SplitHalfTile4RowAVX2(&aq[i*K], K, &prow[0], &srow[0], &out[0], nFull)
			for m := range 4 {
				aScale := aScales[i+m]
				if aScale == 0 {
					dst[(i+m)*N+j] = 0
					continue
				}
				dst[(i+m)*N+j] = out[m] * aScale
			}
		}
	}
	return mFull
}

// w4a8SplitHalfSpanRows is w4a8SpanRows' split-half twin, over an explicit
// row range as well as a column range — the M%4 remainder the tile above
// leaves behind, one activation row at a time through the same
// dotW4A8SplitHalfAVX2 kernel matmulBTW4A8SplitHalfInto (M=1) already uses.
// No ragged-K mop-up: K%32==0 is the caller's guard (matmulBTW4A8SplitHalf
// MultiInto), same as the M=1 path.
func w4a8SplitHalfSpanRows(aq []int8, aScales []float32, w4sh []byte, wScales, dst []float32, K, N, nGroups, bpr, i0, i1, j0, j1 int) {
	if i0 >= i1 {
		return
	}
	nFull := K / 32
	for j := j0; j < j1; j++ {
		prow := w4sh[j*bpr : j*bpr+bpr]
		srow := wScales[j*nGroups : j*nGroups+nGroups]
		for i := i0; i < i1; i++ {
			if aScales[i] == 0 {
				dst[i*N+j] = 0
				continue
			}
			dst[i*N+j] = dotW4A8SplitHalfAVX2(&aq[i*K], &prow[0], &srow[0], nFull) * aScales[i]
		}
	}
}
