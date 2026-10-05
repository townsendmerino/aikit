//go:build amd64

package linalg

import "fmt"

// RepackInt4SplitHalf builds the amd64 split-half layout for this int4-resident WeightMat and
// returns whether it did. OPT-IN by design: nothing calls it implicitly, because it allocates a
// SECOND copy of the tensor's packed nibbles (canonical stays authoritative and is never
// dropped — M>1 and every non-AVX2 path still need it). That memory trade is the caller's to
// make, exactly as RepackInt4Row4's is on arm64.
//
// Returns false — not an error, just "this tensor or this core does not qualify" — when w is not
// int4-resident, AVX2 is unavailable, the group size is not 32, or cols is not a multiple of 32.
// MatmulBTW4A8Into below then keeps using the canonical path transparently.
//
// Scales are shared, not repacked: split-half permutes nibbles within a group and never reorders
// groups, so w.q4s serves both layouts (unlike row4, which needs RepackW4A8Row4Scales).
func (w *WeightMat) RepackInt4SplitHalf() bool {
	// !hasAVX512VNNIVL is not caution, it is correctness in both currencies.
	//
	// The canonical W4A8 dot (quant_w4a8_amd64.go) prefers the AVX-512 VNNI
	// tier whenever the host has it and only falls back to AVX2 otherwise,
	// while the split-half kernel exists at the AVX2 tier ONLY. So on a VNNI
	// host, opting into this layout would swap a VNNI kernel for an AVX2 one:
	//
	//   PERFORMANCE — that is a downgrade, not the 1.12x speedup this repack
	//   is for. The lever points the wrong way on exactly the newest hardware.
	//
	//   NUMERICS — the two tiers accumulate differently, so the split-half
	//   result stops being bit-identical to canonical. Measured by CI, which
	//   is how this was found: TestWeightMatSplitHalf_matchesCanonical passed
	//   on a Zen 2 box (AVX2, no VNNI) at every shape and failed on a VNNI
	//   runner at the largest one, rel 1.09e-4. Bit-identity between the AVX2
	//   canonical and AVX2 split-half kernels holds and is still asserted; it
	//   was never a claim about AVX2-vs-VNNI, and this gate is what keeps the
	//   comparison to the pair it was true of.
	//
	// A split-half VNNI kernel would lift this, and is the obvious follow-up
	// if the layout ever earns its memory. Until then, declining here means
	// the repack is a no-op on VNNI hosts rather than a silent pessimization.
	if w.q4 == nil || !hasAVX2 || hasAVX512VNNIVL {
		return false
	}
	if w.group != 32 || w.cols%32 != 0 {
		return false
	}
	w.q4SplitHalf = RepackW4A8SplitHalf(w.q4, w.rows, w.cols, w.group)
	return true
}

// MatmulBTW4A8Into is the WeightMat-method form for an int4-resident w: it uses the split-half
// AVX2 kernels when RepackInt4SplitHalf has populated the layout, and the canonical per-row
// kernel otherwise.
//
// TWO split-half kernels, split at M=1, the same shape as arm64's row4 dispatch. M=1 takes
// matmulBTW4A8SplitHalfInto, the decode kernel. M>1 takes matmulBTW4A8SplitHalfMultiInto (audit
// M-22 follow-up, prerequisite for a split-half-only WeightMat to serve prefill): the AVX2 tile
// over the M&^3 tile-eligible rows, the M%4 remainder through the same per-row kernel M=1 uses.
// Before this, M>1 had no split-half path at all and a split-half-only WeightMat (no canonical
// to fall back to) panicked here — see docs/audit-2026-09-10.md's M-22 entry.
//
// A WeightMat that was never repacked — including every paged tensor, which by construction has
// no load-time repack step because a read-only mmap span cannot be rewritten in place — simply
// always takes the fallback branch. That makes the paged-MoE carve-out automatic rather than a
// call-site special case.
//
// Chooses a KERNEL, never a numeric result: the two layouts hold the same logical weights and
// the kernels do the same arithmetic in the same order
// (TestWeightMatSplitHalf_matchesCanonical, TestWeightMatSplitHalf_repackedOnlyMatchesCanonical).
//
// Output contract: overwrites dst; do not pre-zero. Covers dst[:M*N].
func (w *WeightMat) MatmulBTW4A8Into(ws *Workspace, a, dst []float32, M int) {
	q := quantizeActScratch(ws, w.w4a8ActGroup(ws, M), a, M, w.cols)
	w.matmulBTW4A8Q(ws, &q, dst, M)
}

// w4a8ActGroup is the activation group MatmulBTW4A8Into quantizes with (R-13).
func (w *WeightMat) w4a8ActGroup(ws *Workspace, _ int) int { return w.groupFor(ws) }

// matmulBTW4A8Q is MatmulBTW4A8Into after its activation quantization: the dispatch it and MatmulBTW4A8PreInto
// share (R-13).
func (w *WeightMat) matmulBTW4A8Q(ws *Workspace, q *ActQ, dst []float32, M int) {
	if q.Group > 0 {
		matmulW4A8GroupedQ(ws, q, w.int4Layout(), dst, M, w.rows) // actgroup.go
		return
	}
	if w.q4SplitHalf != nil {
		matmulBTW4A8SplitHalfF16Q(ws, q, w.q4SplitHalf, w.q4s16, dst, M, w.cols, w.rows, w.group)
		return
	}
	// audit M-22: structurally unreachable for a successfully-constructed
	// split-half-only WeightMat — the branch above now catches q4SplitHalf !=
	// nil at every M, so this is reached only by a hand-built WeightMat{} that
	// bypasses RepackInt4SplitHalfInPlace/WrapInt4SplitHalfOnly's validation
	// (no split-half AND no canonical). A canonical-only or "both" WeightMat
	// never reaches this: q4SplitHalf is nil for canonical-only, and w.q4 is
	// non-nil for "both", so the call below is identical to before this check
	// existed for both those cases.
	if w.q4 == nil {
		panic(fmt.Sprintf("linalg: WeightMat.MatmulBTW4A8Into: no split-half and no canonical layout "+
			"(rows=%d cols=%d) — nothing to dispatch to", w.rows, w.cols))
	}
	matmulBTW4A8F16Q(ws, q, w.q4, w.q4s16, dst, M, w.cols, w.rows, w.group)
}

// matmulBTW4A8SplitHalfInto is MatmulBTW4A8Into's M=1 split-half twin. It mirrors that function's
// structure deliberately — same activation quantization, same serial/parallel split on ws.thr(),
// same span shape — so the two stay comparable when either is changed; arm64's
// MatmulBTW4A8Row4Into duplicates the same skeleton for the same reason.
func matmulBTW4A8SplitHalfInto(ws *Workspace, a []float32, w4sh []byte, wScales, dst []float32, K, N, group int) {
	const M = 1
	checkMatmulW4A8("MatmulBTW4A8SplitHalf", len(a), len(w4sh), len(wScales), len(dst), M, K, N, group)
	// Hard guard, not an assumption. w4a8SplitHalfSpan hands K/32 whole groups to the kernel and
	// has NO ragged-tail mop-up, unlike dotW4A8 — so a K that is not a multiple of 32 would
	// silently drop the last partial group and return a slightly wrong dot with no error. The
	// RepackInt4SplitHalf gate already refuses such tensors, but that gate is in another function
	// and a future caller could reach here directly.
	if K%32 != 0 {
		panic("matmulBTW4A8SplitHalfInto: K must be a multiple of 32 (no ragged-tail path)")
	}
	nGroups, bpr := groupsFor(K, group)
	aq := ws.int8Buf(K)
	aScales := ws.f32Buf(1)
	aScales[0] = quantizeRowInt8(a[:K], aq[:K])
	if N*K < ws.thr() || N < 2 {
		w4a8SplitHalfSpan(aq, aScales[0], w4sh, wScales, dst, K, N, nGroups, bpr, 0, N)
		return
	}
	ws.parallel(N, func(j0, j1 int) {
		w4a8SplitHalfSpan(aq, aScales[0], w4sh, wScales, dst, K, N, nGroups, bpr, j0, j1)
	})
}

// w4a8SplitHalfSpan computes output columns [j0,j1) for the single activation row.
func w4a8SplitHalfSpan(aq []int8, aScale float32, w4sh []byte, wScales, dst []float32, K, N, nGroups, bpr, j0, j1 int) {
	if aScale == 0 {
		for j := j0; j < j1; j++ {
			dst[j] = 0
		}
		return
	}
	nFull := K / 32
	for j := j0; j < j1; j++ {
		prow := w4sh[j*bpr : j*bpr+bpr]
		srow := wScales[j*nGroups : j*nGroups+nGroups]
		dst[j] = dotW4A8SplitHalfAVX2(&aq[0], &prow[0], &srow[0], nFull) * aScale
	}
}

// splitHalfUsable reports whether this CPU can safely dispatch the
// split-half AVX2 kernel — the same hasAVX2 && !hasAVX512VNNIVL gate
// RepackInt4SplitHalf's own body applies (see its comment for why a VNNI
// host declines rather than downgrading). Exported via Int4SplitHalfUsable
// (weightmat.go, audit M-22) for a caller building a repacked-only
// WeightMat, which has no canonical to fall back to if this is false.
func splitHalfUsable() bool { return hasAVX2 && !hasAVX512VNNIVL }

// w4a8BatchSplitHalfSpan is w4a8SplitHalfSpan reached from the portable batch
// span (quant.go's w4a8BatchOp), which is compiled on every architecture —
// exists only so that file can name the split-half kernel without an
// amd64-only symbol reference. The non-amd64 twin (weightmat_splithalf_
// arm64.go, weightmat_canonical_other.go) is unreachable behind
// splitHalfUsable(). Audit M-22 follow-up.
func w4a8BatchSplitHalfSpan(aq []int8, aScale float32, splitHalf []byte, scales, dst []float32, K, N, nGroups, bpr, j0, j1 int) {
	w4a8SplitHalfSpan(aq, aScale, splitHalf, scales, dst, K, N, nGroups, bpr, j0, j1)
}

// matmulBTW4A8SplitHalfGroupedInto is matmulBTW4A8SplitHalfInto with per-32 ACTIVATION scales
// (SetActQuantGroup(32)), on dotW4A8SplitHalfScaledAVX2: the split-half kernel with each group's
// scale multiplied by the activation's group scale in-register (two instructions per group), and no
// final per-row activation multiply. The first version formed those combined scales in Go, per row,
// into a scratch array: ~20% extra work on a small matmul, the 1.5B speed-gate FAIL
// (docs/tasks/task-actquant-pergroup-2026-09.md). Matches matmulW4A8GroupedRef to accumulation order
// (TestActGroup_splitHalfKernelMatchesReference). It takes one activation row already quantized per 32 (R-13).
func matmulBTW4A8SplitHalfGroupedQ(ws *Workspace, aq []int8, aS []float32, w4sh []byte, wScales, dst []float32, K, N int) {
	const group = 32
	nGroups, bpr := groupsFor(K, group)
	span := func(j0, j1 int) {
		for j := j0; j < j1; j++ {
			dst[j] = dotW4A8SplitHalfScaledAVX2(&aq[0], &w4sh[j*bpr], &wScales[j*nGroups], &aS[0], nGroups)
		}
	}
	if N*K < ws.thr() || N < 2 {
		span(0, N)
		return
	}
	ws.parallel(N, span)
}

// w4a8GroupedFastRowsQ serves matmulW4A8GroupedQ from the split-half AVX2 kernel, one activation row at
// a time (M>1 loses the tiles' weight reuse but keeps SIMD). false when the layout or CPU cannot.
// It takes the activation already quantized per 32 for all M rows (R-13); the grouped quantizer is row-independent.
func w4a8GroupedFastRowsQ(ws *Workspace, q *ActQ, l int4Layout, dst []float32, M, N int) bool {
	if l.sh == nil || !splitHalfUsable() {
		return false
	}
	K := l.K
	nG := K / 32
	for m := range M {
		aq, aS := q.Q[m*K:(m+1)*K], q.S[m*nG:(m+1)*nG]
		if l.wS != nil {
			matmulBTW4A8SplitHalfGroupedQ(ws, aq, aS, l.sh, l.wS, dst[m*N:(m+1)*N], K, N)
		} else {
			matmulBTW4A8SplitHalfGroupedF16Q(ws, aq, aS, l.sh, l.wS16, dst[m*N:(m+1)*N], K, N)
		}
	}
	return true
}
