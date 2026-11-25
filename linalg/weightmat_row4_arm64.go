//go:build arm64

package linalg

import (
	"fmt"

	"github.com/townsendmerino/aikit/internal/cpufeat"

	"github.com/townsendmerino/aikit/linalg/internal/shape"
)

// RepackInt4Row4 attempts to populate w's optional split-half +
// 4-row-interleaved layout (RepackW4A8Row4/RepackW4A8Row4Scales) from its
// canonical int4 storage, for a load-time caller to opt a tensor into the
// faster MatmulBTW4A8Into dispatch below. Explicit and caller-driven, never
// probed for automatically inside a matmul (docs/task-w4a8-neon-bandwidth.md's
// plumbing brief design constraint): a loader decides per-tensor whether to
// call this, e.g. skipping it for tensors expertPager will read directly off
// a read-only mmap span, where there is no load-time repack step at all.
//
// Returns false, leaving w unchanged (canonical q4/q4s stay the only
// representation), if: w isn't int4-resident, DotProd isn't available on
// this core, rows isn't a multiple of 4, or cols isn't a multiple of the
// int4 group size (currently only group=32 is supported — the kernel's own
// fixed contract). A false return is not an error; it means this tensor's
// shape or this core doesn't qualify, and MatmulBTW4A8Into below will keep
// using the canonical path transparently.
func (w *WeightMat) RepackInt4Row4() bool {
	if w.q4 == nil || !cpufeat.HasDotProd {
		return false
	}
	if w.group != 32 || w.rows%4 != 0 || w.cols%w.group != 0 {
		return false
	}
	w.q4Row4 = RepackW4A8Row4(w.q4, w.rows, w.cols, w.group)
	w.q4Row4Scales16 = RepackW4A8Row4ScalesF16(w.q4s16, w.rows, w.cols, w.group)
	return true
}

// MatmulBTW4A8Into is MatmulBTW4A8Into's WeightMat-method form for an
// int4-resident w, uniform for the caller: uses the row4-interleaved layout
// whenever RepackInt4Row4 (or WrapInt4 with LayoutRow4) has populated it, and the
// canonical per-row kernel otherwise.
//
// TWO row4 kernels, split at M=1. M=1 takes dotW4A8SplitHalf4Row, the decode
// kernel: one activation row against a quad's four weight rows. M>1 takes the
// register-blocked tile (matmulBTW4A8Row4TileInto, docs/task-simd-audit.md
// S-01), which reduces FOUR activation rows against those same four weight rows
// per call so the nibble unpack and scale broadcast are paid once per weight row
// per group instead of once per activation row. Measured 2.88x over the
// canonical path at M>=4, K=1536 N=8960, single core (TestW4A8TileVsCanonicalAB).
//
// The original scoping of the row4 work was M=1 only — "batched/prefill gets
// nothing from this optimization" (docs/prompts/w4a8-item3-harness.md). That was
// true of the M=1 KERNEL and is why prefill kept the canonical path; it stopped
// being true of the LAYOUT once a tile existed to exploit it, which is S-01's
// whole point. This is where the
// paged-MoE carve-out becomes automatic rather than a call-site special
// case: a WeightMat that was never repacked (paged tensors, by
// construction, since a read-only mmap span has no load-time repack step)
// simply always takes the fallback branch here.
//
// Bit-identical down all three branches, for the same logical weights — this
// method chooses a kernel, never a numeric result. Held by
// TestDotW4A8SplitHalf4Row_bitIdenticalToCanonical (the M=1 kernel),
// TestMatmulBTW4A8Row4TileInto_bitIdenticalToCanonical (the tile), and
// TestWeightMatW4A8_MConsistentAcrossRow4Dispatch, which is the one that spans
// the M=1/M>1 split HERE: it compares a row computed alone through the decode
// kernel against the same row computed inside a batch through the tile, which is
// exactly the pair speculative verify exercises.
//
// Output contract: overwrites dst; do not pre-zero. Covers dst[:M*N].
func (w *WeightMat) MatmulBTW4A8Into(ws *Workspace, a, dst []float32, M int) {
	q := quantizeActScratch(ws, w.w4a8ActGroup(ws, M), a, M, w.cols)
	w.matmulBTW4A8Q(ws, &q, dst, M)
}

// w4a8ActGroup is the activation group MatmulBTW4A8Into quantizes with: w.groupFor(ws), except that the row4 M=1
// path reads the workspace's own group (matmulBTW4A8Row4F16Into's actGroupFor), as it always has. The two differ only
// when w carries a group the workspace does not.
func (w *WeightMat) w4a8ActGroup(ws *Workspace, M int) int {
	g := w.groupFor(ws)
	if g > 0 && w.q4Row4 != nil && M == 1 {
		return actGroupFor(ws)
	}
	return g
}

// matmulBTW4A8Q is MatmulBTW4A8Into after its activation quantization: the dispatch it and MatmulBTW4A8PreInto
// share (R-13).
func (w *WeightMat) matmulBTW4A8Q(ws *Workspace, q *ActQ, dst []float32, M int) {
	if w.q4Row4 != nil && M == 1 {
		// matmulBTW4A8Row4F16Q takes its per-32 kernel path (or the reference) itself on a grouped q.
		matmulBTW4A8Row4F16Q(ws, q, w.q4Row4, w.q4Row4Scales16, dst, w.cols, w.rows, w.group)
		return
	}
	if q.Group > 0 {
		matmulW4A8GroupedQ(ws, q, w.int4Layout(), dst, M, w.rows) // actgroup.go
		return
	}
	if w.q4Row4 != nil {
		matmulBTW4A8Row4TileF16Q(ws, q, w.q4Row4, w.q4Row4Scales16, dst, M, w.cols, w.rows, w.group)
		return
	}
	// audit M-22: structurally unreachable for a repacked-only WeightMat —
	// the branch above catches q4Row4 != nil at every M (the tile handles
	// M>1, unlike amd64's split-half), so a successfully-constructed
	// repacked-only WeightMat never reaches here. Guarded anyway against a
	// hand-built WeightMat{} that bypasses RepackInt4InPlace(LayoutRow4)/
	// WrapInt4(LayoutRow4)'s validation.
	if w.q4 == nil {
		panic(fmt.Sprintf("linalg: WeightMat.MatmulBTW4A8Into: no canonical and no row4 layout "+
			"(rows=%d cols=%d) — nothing to dispatch to", w.rows, w.cols))
	}
	matmulBTW4A8F16Q(ws, q, w.q4, w.q4s16, dst, M, w.cols, w.rows, w.group)
}

// row4Usable reports whether this CPU can safely dispatch the split-half +
// 4-row-interleaved kernel — the same cpufeat.HasDotProd gate RepackInt4Row4 already
// applies before ever setting q4Row4. WrapInt4 with LayoutRow4 (through
// int4Row4Usable) applies the identical gate to externally-supplied bytes (a
// .giw kind-4 load reading an already-repacked layout off disk, which may have
// been written on a core with DotProd when this one has none).
func row4Usable() bool { return cpufeat.HasDotProd }

// requireRow4DotProd refuses, by name, a call to one of the exported 4-row W4A8 matmuls
// on a CPU without DotProd: their kernels are SDOT throughout, and without the
// instruction they would die with SIGILL. Callers check row4Usable (WeightMat does) or
// use MatmulBTW4A8Into with the canonical layout.
func requireRow4DotProd(fn string) {
	if !row4Usable() {
		panic("linalg: " + fn + " requires DotProd (SDOT); use MatmulBTW4A8Into with the canonical layout")
	}
}

// w4a8BatchRow4Span is w4a8Row4Span reached from the portable batch span. It
// exists only so matmulBTW4A8Batch can name the row4 kernel from quant.go,
// which is compiled on every architecture; the non-arm64 twin is unreachable
// behind row4Usable().
func w4a8BatchRow4Span(aq []int8, aScale float32, row4 []byte, row4Scales, dst []float32, nGroups, bpr, q0, q1 int) {
	// nil corr: the pre-S-05 kernel. The batched q‖k‖v path is opt-in and parked
	// (goinfer R-06); it gets the S-05 fold when it is funded, not silently here.
	w4a8Row4Span(aq, nil, aScale, row4, row4Scales, dst, nGroups, bpr, q0, q1)
}

// repackInt4Row4InPlace is RepackInt4InPlace's LayoutRow4 form. ok=false, input untouched, when
// int4Row4Usable is false, checked before any write.
//
// Peak extra memory is one quad's worth of scratch (4*bpr bytes, one allocation reused across every quad:
// 8 KB at K=4096, independent of rows) for the nibbles, plus one quad of scales, not a second full-tensor
// array. That works because row4 is a fixed permutation local to one quad of four rows
// (RepackInt4Row4Quad's doc): a quad's row4 output depends only on that quad's canonical input, so the quad
// is copied to scratch and the result written back into the same position. A same-buffer call without the
// copy would not be safe: an early group's write can land on a later group's unread source byte.
//
// The input is consumed: q4 and q4s must be writable (never a read-only mapping), and after ok they hold
// row4 order and belong to the result as q4Row4/q4Row4Scales16. A loader that never wants canonical bytes in
// the heap should run RepackInt4Row4Quad per quad as it copies off the mapping and wrap the result with
// WrapInt4(LayoutRow4).
func repackInt4Row4InPlace(q4 []byte, q4s []uint16, rows, cols, group int) (WeightMat, bool) {
	if !int4Row4Usable(rows, cols, group) {
		return WeightMat{}, false
	}
	nGroups, bpr := groupsFor(cols, group)
	shape.RequireExactLen("RepackInt4InPlace(LayoutRow4)", "q4", len(q4), shape.Mul(rows, bpr))
	shape.RequireExactLen("RepackInt4InPlace(LayoutRow4)", "q4s", len(q4s), shape.Mul(rows, nGroups))

	quadBytes := 4 * bpr
	nibScratch := make([]byte, quadBytes) // the ONE allocation the nibble pass makes, reused every quad
	for q := 0; q < rows/4; q++ {
		base := q * quadBytes
		quad := q4[base : base+quadBytes]
		copy(nibScratch, quad)
		RepackInt4Row4Quad(quad, nibScratch, cols)
	}

	quadScales := 4 * nGroups
	scaleScratch := make([]uint16, quadScales) // the scales pass's one allocation, likewise reused
	for q := 0; q < rows/4; q++ {
		base := q * quadScales
		quad := q4s[base : base+quadScales]
		copy(scaleScratch, quad)
		RepackInt4Row4ScalesQuadF16(quad, scaleScratch, nGroups)
	}

	return WeightMat{q4Row4: q4, q4Row4Scales16: q4s, group: group, rows: rows, cols: cols}, true
}
