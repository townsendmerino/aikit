//go:build arm64

package linalg

import "fmt"

// MatmulBTW4A8Row4TileInto is MatmulBTW4A8Row4Into's M>1 sibling: the same
// row4 weights (RepackW4A8Row4 / RepackW4A8Row4Scales), the same numerics, but
// four ACTIVATION rows are reduced against a quad's four weight rows in one
// kernel call instead of one activation row per call.
//
// docs/task-simd-audit.md S-01. The canonical M>1 path (w4a8Span) is a GEMV per
// (activation row, weight row) pair, so the 16-byte weight load, the four-op
// nibble unpack and the scale broadcast are each paid M times over the same
// bytes. That is the mechanism behind "int8int8 prefill beats int4 by 25-33%"
// despite int4 moving half the bytes: at M>1 the byte saving is served out of
// cache anyway while the unpack ALU cost stays. Here the unpack is paid once per
// weight row per group regardless of M.
//
// Bit-identical to MatmulBTW4A8Into on the same logical weights, by construction
// rather than by tolerance: see dot_w4a8_tile_arm64.s for the per-output
// instruction sequence, and TestMatmulBTW4A8_MConsistent /
// TestWeightMatW4A8_MConsistentAcrossRow4Dispatch for the gates that hold it.
//
// Contract is row4's, minus the M=1 restriction: N a multiple of 4, K a multiple
// of group=32. Any M is accepted; M%4 remainder rows fall to the existing
// four-weight-row kernel one activation row at a time, which is the shape the
// M=1 path already ships.
func MatmulBTW4A8Row4TileInto(ws *Workspace, a []float32, w4Row4 []byte, wScales4 []float32, dst []float32, M, K, N, group int) {
	checkMatmulW4A8("MatmulBTW4A8Row4Tile", len(a), len(w4Row4), len(wScales4), len(dst), M, K, N, group)
	checkGroupMatmul("MatmulBTW4A8Row4Tile", len(a), w4Row4, wScales4, len(dst), M, K, N, group)
	if group != 32 {
		panic(fmt.Sprintf("linalg: MatmulBTW4A8Row4TileInto requires group=32, got %d", group))
	}
	if N%4 != 0 {
		panic(fmt.Sprintf("linalg: MatmulBTW4A8Row4TileInto requires N a multiple of 4, got %d", N))
	}
	if K%group != 0 {
		panic(fmt.Sprintf("linalg: MatmulBTW4A8Row4TileInto requires K a multiple of group=%d, got %d", group, K))
	}
	// K=0 satisfies K%group==0, so without this it reaches the span with
	// nGroups=0 and fails as an index-out-of-range on &blk[0] — a panic that
	// names a kernel internal instead of the caller's shape. Every other bad
	// shape here reports itself; this one did not.
	if K < group {
		panic(fmt.Sprintf("linalg: MatmulBTW4A8Row4TileInto requires K >= group=%d, got %d", group, K))
	}
	nGroups, bpr := groupsFor(K, group)
	aq := ws.int8Buf(M * K)
	aScales := ws.f32Buf(M)
	for i := range M {
		aScales[i] = quantizeRowInt8(a[i*K:i*K+K], aq[i*K:i*K+K])
	}
	nQuads := N / 4
	if M*N*K < ws.thr() || nQuads < 2 {
		w4a8Row4TileSpan(aq, aScales, w4Row4, wScales4, dst, M, K, N, nGroups, bpr, 0, nQuads)
		return
	}
	ws.parallel(nQuads, func(q0, q1 int) {
		w4a8Row4TileSpan(aq, aScales, w4Row4, wScales4, dst, M, K, N, nGroups, bpr, q0, q1)
	})
}

// w4a8Row4TileSpan computes output quads [q0,q1) — 4 weight rows each — for all
// M activation rows, given the already-quantized activations.
//
// Quad-outer, activation-block-inner: each quad's weight bytes are streamed once
// per span and every activation row is applied to them while they are in cache,
// which is the reuse direction that matters — the weights are the large operand
// (N*K/2 bytes) and the activations the small one (M*K, L1-resident at decode
// and verify batch sizes).
func w4a8Row4TileSpan(aq []int8, aScales []float32, w4Row4 []byte, wScales4 []float32, dst []float32, M, K, N, nGroups, bpr, q0, q1 int) {
	var tile [16]float32
	var one [4]float32
	mFull := M &^ 3 // the largest multiple of 4 at or below M
	for q := q0; q < q1; q++ {
		blk := w4Row4[q*4*bpr : q*4*bpr+4*bpr]
		sblk := wScales4[q*4*nGroups : q*4*nGroups+4*nGroups]
		for i := 0; i < mFull; i += 4 {
			dotW4A8Row4Tile4x4(&aq[i*K], K, &blk[0], &sblk[0], &tile[0], nGroups)
			for m := range 4 {
				scatterRow4(dst, (i+m)*N+q*4, tile[m*4:m*4+4], aScales[i+m])
			}
		}
		// M%4 leftover: the shipped four-weight-row kernel, one activation row
		// per call — the exact M=1 path, so the remainder needs no argument of
		// its own for why it matches.
		for i := mFull; i < M; i++ {
			dotW4A8SplitHalf4Row(&aq[i*K], &blk[0], &sblk[0], &one[0], nGroups)
			scatterRow4(dst, i*N+q*4, one[:], aScales[i])
		}
	}
}

// scatterRow4 writes one activation row's four outputs for a quad, applying the
// activation scale. The aScale==0 shortcut mirrors w4a8Span's: a row that
// quantized to all zeros stores a literal 0 rather than a product, so the tile
// path cannot differ from the canonical one even in the sign of a zero.
func scatterRow4(dst []float32, base int, out []float32, aScale float32) {
	if aScale == 0 {
		dst[base] = 0
		dst[base+1] = 0
		dst[base+2] = 0
		dst[base+3] = 0
		return
	}
	dst[base] = out[0] * aScale
	dst[base+1] = out[1] * aScale
	dst[base+2] = out[2] * aScale
	dst[base+3] = out[3] * aScale
}
