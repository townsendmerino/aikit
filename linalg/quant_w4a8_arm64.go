//go:build arm64

package linalg

import "github.com/townsendmerino/aikit/internal/cpufeat"

// dotW4A8FoldSDOT returns the per-group-scaled f32 dot Σ_g scale[g]·(act·w)_g of
// one int4 weight row against the int8 activation row, via the fused NEON+SDOT
// kernel in dot_w4a8_arm64.s. The f32 weight scales are folded IN-REGISTER (SCVTF
// + FMLA into a 4-lane accumulator, one FADDP reduce at the end) — no per-group
// int32 scratch and no Go-side fold loop. Only safe on DotProd-capable cores
// (gated by cpufeat.HasDotProd, like dotI8SDOT). group is fixed at 32; nGroups = K/32.
// Validated on M1 Pro (quant_w4a8_test.go + BenchmarkQ4vsQ8).
//
//go:noescape
func dotW4A8FoldSDOT(act *int8, packed *byte, scales *float32, nGroups int) float32

// dotW4A8FoldSDOTv2 is dotW4A8FoldSDOT with the centering subtract dropped from
// the main loop (uncentered nibbles + a separate, batched correction pass over
// sumAct — see dot_w4a8_arm64.s and docs/task-w4a8-neon-bandwidth.md Gate 1,
// items 1+2). NOT yet wired into dotW4A8's dispatch — kept side by side with
// the original for correctness/perf comparison before any switch-over.
// sumAct is nGroups long (SumActGroupsInto), the SAME activation row's
// per-group sums dotW4A8FoldSDOT's caller would otherwise not need.
//
//go:noescape
func dotW4A8FoldSDOTv2(act *int8, packed *byte, scales *float32, sumAct *int32, nGroups int) float32

// dotW4A8SplitHalfSDOT is dotW4A8FoldSDOT with the layout changed to
// split-half (repackSplitHalfRow) and signed centering kept — item 3's core
// lever in isolation. NOT wired into dotW4A8's dispatch: packed must be in
// the split-half layout, which dotW4A8's canonical-layout callers do not
// produce. Harness-only until a winning grid cell funds the production
// repack (docs/prompts/w4a8-item3-harness.md).
//
//go:noescape
func dotW4A8SplitHalfSDOT(act *int8, packed *byte, scales *float32, nGroups int) float32

// dotW4A8FoldSDOT2Acc is dotW4A8FoldSDOT with the fold split across two
// independent accumulator chains (canonical layout, signed centering
// unchanged) — a probe for whether the serial VFMLA fold, not instruction
// count, is the kernel's real bottleneck. nGroups must be even. See
// dot_w4a8_arm64.s for the motivating measurement.
//
//go:noescape
func dotW4A8FoldSDOT2Acc(act *int8, packed *byte, scales *float32, nGroups int) float32

// dotW4A8FoldSDOT4Acc extends the 2-accumulator probe to four independent
// chains, matching dotI8SDOT's own four-accumulator design. nGroups must be
// a multiple of 4. See dot_w4a8_arm64.s.
//
//go:noescape
func dotW4A8FoldSDOT4Acc(act *int8, packed *byte, scales *float32, nGroups int) float32

// dotW4A8SplitHalf2Acc combines the two confirmed levers: 2 independent
// accumulator chains (dotW4A8FoldSDOT2Acc) plus the split-half layout
// (dotW4A8SplitHalfSDOT) so each lane's unpack drops VZIP1/VZIP2. packed
// must be split-half layout (repackSplitHalfRow). nGroups must be even.
//
//go:noescape
func dotW4A8SplitHalf2Acc(act *int8, packed *byte, scales *float32, nGroups int) float32

// dotW4A8SplitHalf4Acc extends dotW4A8SplitHalf2Acc to four accumulator
// lanes. packed must be split-half layout. nGroups must be a multiple of 4.
//
//go:noescape
func dotW4A8SplitHalf4Acc(act *int8, packed *byte, scales *float32, nGroups int) float32

// dotW4A8SplitHalf4Row computes 4 REAL output rows in one call (item 4,
// docs/prompts/w4a8-item3-harness.md), given packed4/scales4 in
// repackSplitHalf4RowBlock / interleaveScales4Row's interleaved layout.
// dst must have room for 4 float32s. See dot_w4a8_arm64.s.
//
//go:noescape
func dotW4A8SplitHalf4Row(act *int8, packed4 *byte, scales4 *float32, dst *float32, nGroups int)

// dotW4A8Row4Tile4x4 is the register-blocked M>1 tile (docs/task-simd-audit.md
// S-01): four ACTIVATION rows, actStride bytes apart, against the four weight
// rows of one row4 quad, producing 16 float32 row-major in dst (dst[m*4+r] is
// activation row m against weight row r). packed4/scales4 are the same
// repackSplitHalf4RowBlock / interleaveScales4Row layout dotW4A8SplitHalf4Row
// consumes; each weight row is loaded and unpacked ONCE per group instead of
// once per activation row. Bit-identical to calling dotW4A8SplitHalf4Row per
// activation row — same per-output instruction sequence, see
// dot_w4a8_tile_arm64.s.
//
//go:noescape
func dotW4A8Row4Tile4x4(act *int8, actStride int, packed4 *byte, scales4 *float32, dst *float32, nGroups int)

// dotW4A8SplitHalf4RowPrefetch is dotW4A8SplitHalf4Row plus one PRFM
// PLDL1KEEP per outer iteration, issued prefetchDistance BYTES ahead of the
// current group's packed4 pointer (docs/task-w4a8-neon-bandwidth.md's
// cold-fix harness pass, PRFM remedy). Pure hint, bit-identical to
// dotW4A8SplitHalf4Row by construction. See dot_w4a8_arm64.s.
//
//go:noescape
func dotW4A8SplitHalf4RowPrefetch(act *int8, packed4 *byte, scales4 *float32, dst *float32, nGroups int, prefetchDistance int)

// dotW4A8SplitHalf4RowDeshared is dotW4A8SplitHalf4Row with the 4 rows' bytes
// kept in 4 separate slices (repackSplitHalfRow's plain per-row layout, no
// interleave) instead of one contiguous interleaved block —
// docs/task-w4a8-neon-bandwidth.md's cold-fix harness pass, chain/line
// de-sharing remedy. The activation is still loaded once per group and
// shared across all 4 SDOTs; only the weight/scale memory is de-shared.
// Bit-identical to dotW4A8SplitHalf4Row by construction (same per-row math,
// different source pointers). See dot_w4a8_arm64.s.
//
//go:noescape
func dotW4A8SplitHalf4RowDeshared(act *int8, packed0, packed1, packed2, packed3 *byte, scales0, scales1, scales2, scales3 *float32, dst *float32, nGroups int)

// dotW4A8 computes one W4A8 output (before the activation scale). The DotProd
// path folds the per-group weight scales inside the kernel and returns the f32
// dot directly; only a ragged final group (K % 32 ≠ 0) is mopped up in Go.
// Everything off the fast path falls back to the reference.
func dotW4A8(act []int8, packed []byte, scales []float32, group, K int) float32 {
	if cpufeat.HasDotProd && group == 32 && K >= 32 {
		nFull := K / 32
		total := dotW4A8FoldSDOT(&act[0], &packed[0], &scales[0], nFull)
		if done := nFull * 32; done < K {
			// Ragged final group (K not a multiple of 32): scalar, scales[nFull].
			var acc int32
			for k := done; k < K; k++ {
				b := packed[k>>1]
				nib := b & 0x0F
				if k&1 == 1 {
					nib = b >> 4
				}
				acc += int32(act[k]) * int32(int(nib)-8)
			}
			total += float32(acc) * scales[nFull]
		}
		return total
	}
	return dotW4A8Scalar(act, packed, scales, group, K)
}
