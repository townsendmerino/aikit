//go:build arm64

package linalg

import "fmt"

// RepackW4A8Row4 converts a canonical group-int4 packed weight matrix
// (QuantizeGroupsInt4's layout: [N,K] row-major, interleaved even/odd
// nibbles per row) into the split-half + 4-row-interleaved layout
// MatmulBTW4A8Row4Into consumes, for use as a one-time arm64 CPU load-time
// repack (docs/task-w4a8-neon-bandwidth.md's item-3+4 harness result — GO,
// 2026-08-23/24). The canonical packer, the .giw on-disk format, the scalar
// oracle, and amd64 are all untouched by this: this produces a SECOND,
// arm64-only byte array the caller stores alongside (or in place of) the
// canonical one.
//
// N must be a multiple of 4 and K a multiple of group (32, the only group
// size MatmulBTW4A8Row4Into accepts) — both are the interleave's own
// contract, not negotiable per-call. A caller whose tensor doesn't meet
// both should not call this at all; route that tensor through the
// unmodified MatmulBTW4A8Into/canonical packing instead. Output is the same
// total byte count as the input (repacking, not re-quantizing), so this
// costs one extra O(N*K) pass at load time and briefly doubles the
// CPU-resident bytes for the tensor (both copies exist until the caller
// drops one) — a real load-time/RAM cost, not free; the plumbing phase
// measures both.
func RepackW4A8Row4(packed []byte, N, K, group int) []byte {
	if group != 32 {
		panic(fmt.Sprintf("linalg: RepackW4A8Row4 requires group=32, got %d", group))
	}
	if N%4 != 0 {
		panic(fmt.Sprintf("linalg: RepackW4A8Row4 requires N a multiple of 4, got %d", N))
	}
	if K%group != 0 {
		panic(fmt.Sprintf("linalg: RepackW4A8Row4 requires K a multiple of group=%d, got %d", group, K))
	}
	_, bpr := groupsFor(K, group)
	if len(packed) < N*bpr {
		panic(fmt.Sprintf("linalg: RepackW4A8Row4 packed len %d < N*bytesPerRow = %d", len(packed), N*bpr))
	}
	out := make([]byte, N*bpr)
	for q := 0; q < N/4; q++ {
		r0, r1, r2, r3 := q*4*bpr, (q*4+1)*bpr, (q*4+2)*bpr, (q*4+3)*bpr
		blk := repackSplitHalf4RowBlock(packed[r0:r0+bpr], packed[r1:r1+bpr], packed[r2:r2+bpr], packed[r3:r3+bpr], K)
		copy(out[q*4*bpr:(q*4+4)*bpr], blk)
	}
	return out
}

// RepackW4A8Row4Scales is RepackW4A8Row4's counterpart for the per-group f32
// scales array: same interleaving, same N/K/group contract.
func RepackW4A8Row4Scales(scales []float32, N, K, group int) []float32 {
	if group != 32 {
		panic(fmt.Sprintf("linalg: RepackW4A8Row4Scales requires group=32, got %d", group))
	}
	if N%4 != 0 {
		panic(fmt.Sprintf("linalg: RepackW4A8Row4Scales requires N a multiple of 4, got %d", N))
	}
	if K%group != 0 {
		panic(fmt.Sprintf("linalg: RepackW4A8Row4Scales requires K a multiple of group=%d, got %d", group, K))
	}
	nGroups, _ := groupsFor(K, group)
	if len(scales) < N*nGroups {
		panic(fmt.Sprintf("linalg: RepackW4A8Row4Scales scales len %d < N*nGroups = %d", len(scales), N*nGroups))
	}
	out := make([]float32, N*nGroups)
	for q := 0; q < N/4; q++ {
		r0, r1, r2, r3 := q*4*nGroups, (q*4+1)*nGroups, (q*4+2)*nGroups, (q*4+3)*nGroups
		blk := interleaveScales4Row(scales[r0:r0+nGroups], scales[r1:r1+nGroups], scales[r2:r2+nGroups], scales[r3:r3+nGroups], nGroups)
		copy(out[q*4*nGroups:(q*4+4)*nGroups], blk)
	}
	return out
}

// MatmulBTW4A8Row4Into computes dst[N] = a[K]·bᵀ where b is [N,K] stored
// group-int4 in RepackW4A8Row4/RepackW4A8Row4Scales's layout — a drop-in
// PERFORMANCE replacement for MatmulBTW4A8Into at M=1 (bit-identical given
// the same logical weights: docs/task-w4a8-neon-bandwidth.md's
// TestDotW4A8SplitHalf4Row_bitIdenticalToCanonical), not a numerics change.
//
// Hard contract, checked and panicking (caller decides per-tensor whether it
// holds, mirroring how MatmulBTW4A8Into's own ragged-K tail is handled by
// falling back rather than growing hybrid logic here): M must be 1 (this
// kernel fuses one activation row's quantization with 4-row weight
// processing — M>1/batched prefill gets nothing from this optimization and
// should keep routing through MatmulBTW4A8Into, matching the harness
// brief's own "no M>1-specific work" scoping), N a multiple of 4, K a
// multiple of group=32. A tensor that doesn't meet all three should never
// have been repacked with RepackW4A8Row4 in the first place — route its
// whole matmul through MatmulBTW4A8Into with the canonical layout instead
// of mixing layouts within one call.
func MatmulBTW4A8Row4Into(ws *Workspace, a []float32, w4Row4 []byte, wScales4 []float32, dst []float32, M, K, N, group int) {
	checkMatmulW4A8("MatmulBTW4A8Row4", len(a), len(w4Row4), len(wScales4), len(dst), M, K, N, group)
	checkGroupMatmul("MatmulBTW4A8Row4", len(a), w4Row4, wScales4, len(dst), M, K, N, group)
	if M != 1 {
		panic(fmt.Sprintf("linalg: MatmulBTW4A8Row4Into requires M=1, got M=%d — use MatmulBTW4A8Into for M>1", M))
	}
	if group != 32 {
		panic(fmt.Sprintf("linalg: MatmulBTW4A8Row4Into requires group=32, got %d", group))
	}
	if N%4 != 0 {
		panic(fmt.Sprintf("linalg: MatmulBTW4A8Row4Into requires N a multiple of 4, got %d", N))
	}
	if K%group != 0 {
		panic(fmt.Sprintf("linalg: MatmulBTW4A8Row4Into requires K a multiple of group=%d, got %d", group, K))
	}
	// Same K=0 hole as MatmulBTW4A8Row4TileInto: K%group==0 admits it, and the
	// span then indexes an empty block.
	if K < group {
		panic(fmt.Sprintf("linalg: MatmulBTW4A8Row4Into requires K >= group=%d, got %d", group, K))
	}
	nGroups, bpr := groupsFor(K, group)
	aq := ws.int8Buf(K)
	aScale := quantizeRowInt8(a[:K], aq)
	if aScale == 0 {
		for j := range N {
			dst[j] = 0
		}
		return
	}
	nQuads := N / 4
	if M*N*K < ws.thr() || nQuads < 2 {
		w4a8Row4Span(aq, aScale, w4Row4, wScales4, dst, nGroups, bpr, 0, nQuads)
		return
	}
	ws.parallel(nQuads, func(q0, q1 int) {
		w4a8Row4Span(aq, aScale, w4Row4, wScales4, dst, nGroups, bpr, q0, q1)
	})
}

// w4a8Row4Span computes output quads [q0,q1) — 4 real rows each — of
// MatmulBTW4A8Row4Into's dst, given the already-quantized activation row.
func w4a8Row4Span(aq []int8, aScale float32, w4Row4 []byte, wScales4 []float32, dst []float32, nGroups, bpr, q0, q1 int) {
	var out [4]float32
	for q := q0; q < q1; q++ {
		blk := w4Row4[q*4*bpr : q*4*bpr+4*bpr]
		sblk := wScales4[q*4*nGroups : q*4*nGroups+4*nGroups]
		dotW4A8SplitHalf4Row(&aq[0], &blk[0], &sblk[0], &out[0], nGroups)
		dst[q*4] = out[0] * aScale
		dst[q*4+1] = out[1] * aScale
		dst[q*4+2] = out[2] * aScale
		dst[q*4+3] = out[3] * aScale
	}
}
