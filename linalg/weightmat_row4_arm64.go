//go:build arm64

package linalg

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
	if w.q4 == nil || !hasDotProd {
		return false
	}
	if w.group != 32 || w.rows%4 != 0 || w.cols%w.group != 0 {
		return false
	}
	w.q4Row4 = RepackW4A8Row4(w.q4, w.rows, w.cols, w.group)
	w.q4Row4Scales = RepackW4A8Row4Scales(w.q4s, w.rows, w.cols, w.group)
	return true
}

// MatmulBTW4A8Into is MatmulBTW4A8Into's WeightMat-method form for an
// int4-resident w, uniform for the caller: uses the row4-interleaved layout
// whenever RepackInt4Row4 (or WrapInt4Row4) has populated it, and the
// canonical per-row kernel otherwise.
//
// TWO row4 kernels, split at M=1. M=1 takes dotW4A8SplitHalf4Row, the decode
// kernel: one activation row against a quad's four weight rows. M>1 takes the
// register-blocked tile (MatmulBTW4A8Row4TileInto, docs/task-simd-audit.md
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
func (w *WeightMat) MatmulBTW4A8Into(ws *Workspace, a, dst []float32, M int) {
	if w.q4Row4 != nil {
		if M == 1 {
			MatmulBTW4A8Row4Into(ws, a, w.q4Row4, w.q4Row4Scales, dst, M, w.cols, w.rows, w.group)
			return
		}
		MatmulBTW4A8Row4TileInto(ws, a, w.q4Row4, w.q4Row4Scales, dst, M, w.cols, w.rows, w.group)
		return
	}
	MatmulBTW4A8Into(ws, a, w.q4, w.q4s, dst, M, w.cols, w.rows, w.group)
}

// row4Usable reports whether this CPU can safely dispatch the split-half +
// 4-row-interleaved kernel — the same hasDotProd gate RepackInt4Row4 already
// applies before ever setting q4Row4. WrapInt4Row4 (weightmat.go, portable)
// calls this to apply the identical gate to EXTERNALLY-supplied bytes (a
// .giw kind-4 load reading an already-repacked layout off disk) — a case
// RepackInt4Row4's own gate never covered, since before WrapInt4Row4 existed
// the only way q4Row4 got populated was RepackInt4Row4 itself.
func row4Usable() bool { return hasDotProd }

// w4a8BatchRow4Span is w4a8Row4Span reached from the portable batch span. It
// exists only so MatmulBTW4A8Batch can name the row4 kernel from quant.go,
// which is compiled on every architecture; the non-arm64 twin is unreachable
// behind row4Usable().
func w4a8BatchRow4Span(aq []int8, aScale float32, row4 []byte, row4Scales, dst []float32, nGroups, bpr, q0, q1 int) {
	w4a8Row4Span(aq, aScale, row4, row4Scales, dst, nGroups, bpr, q0, q1)
}
