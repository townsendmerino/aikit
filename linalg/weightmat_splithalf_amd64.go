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
// AVX2 kernel when RepackInt4SplitHalf has populated the layout and M=1, and the canonical
// per-row kernel otherwise.
//
// M=1 ONLY, matching arm64's row4 dispatch and for the same reason: this is a decode-path
// optimization. Prefill (M>1) amortizes the unpack across many activation rows, so the shuffle
// ops the split-half layout deletes are not the binding cost there, and routing it through an
// untested path would buy nothing for the risk.
//
// A WeightMat that was never repacked — including every paged tensor, which by construction has
// no load-time repack step because a read-only mmap span cannot be rewritten in place — simply
// always takes the fallback branch. That makes the paged-MoE carve-out automatic rather than a
// call-site special case.
//
// Chooses a KERNEL, never a numeric result: the two layouts hold the same logical weights and
// the kernels do the same arithmetic in the same order
// (TestWeightMatSplitHalf_matchesCanonical).
func (w *WeightMat) MatmulBTW4A8Into(ws *Workspace, a, dst []float32, M int) {
	if M == 1 && w.q4SplitHalf != nil {
		matmulBTW4A8SplitHalfInto(ws, a, w.q4SplitHalf, w.q4s, dst, w.cols, w.rows, w.group)
		return
	}
	// audit M-22: reachable at M>1 for a split-half-only WeightMat — amd64 has
	// no register-blocked tile for split-half (unlike arm64's row4, which got
	// one specifically so this branch stays reachable for M>1 too), so there
	// is genuinely no fast path for prefill through this layout yet, and
	// w.q4 == nil here means there is no canonical to fall back to either. A
	// canonical-only or "both" WeightMat never reaches this: q4SplitHalf is
	// nil for canonical-only, and w.q4 is non-nil for "both", so the call
	// below is identical to before this check existed for both those cases.
	if w.q4 == nil {
		panic(fmt.Sprintf("linalg: WeightMat.MatmulBTW4A8Into: split-half-only WeightMat (rows=%d cols=%d) "+
			"has no path for M=%d — the split-half AVX2 kernel is M=1 only and there is no canonical "+
			"fallback; a split-half-only tensor cannot serve M>1 through this method", w.rows, w.cols, M))
	}
	MatmulBTW4A8Into(ws, a, w.q4, w.q4s, dst, M, w.cols, w.rows, w.group)
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
