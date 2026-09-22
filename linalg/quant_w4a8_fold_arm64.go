//go:build arm64

package linalg

// S-05 (docs/task-simd-audit.md): the M=1 row4 decode kernel with the -8
// centering folded into the SDOT accumulators' initial value. See
// dot_w4a8_fold_arm64.s for the identity and the µop count.

// w4a8RowFold selects dotW4A8SplitHalf4RowFold for MatmulBTW4A8Row4Into's M=1
// path (true) or the pre-S-05 dotW4A8SplitHalf4Row (false). Default on: the
// two are bit-identical for every input (TestDotW4A8SplitHalf4RowFold_*), so
// this is a performance switch kept so an in-process A/B can flip it —
// SetW4A8RowFold. Not goroutine-safe to flip mid-matmul; a benchmark sets it
// between calls.
var w4a8RowFold = true

// SetW4A8RowFold switches the M=1 row4 W4A8 path between the S-05 fold kernel
// (true, the default) and the pre-S-05 kernel (false). Numerically inert —
// both kernels produce identical bits — so this exists only for same-process
// A/B measurement (docs/internal/measuring-performance.md rule 3).
func SetW4A8RowFold(on bool) { w4a8RowFold = on }

// W4A8RowFold reports whether the S-05 fold kernel is selected (see
// SetW4A8RowFold).
func W4A8RowFold() bool { return w4a8RowFold }

// w4a8LaneCorrNeg8 writes corr[4g+l] = -8·(Σ act[32g+4l..32g+4l+3] +
// Σ act[32g+16+4l..32g+16+4l+3]) for g < nGroups, l < 4 — the per-lane
// activation sums in the row kernel's own SDOT lane mapping, negated and
// scaled by the centering constant. corr must hold 4*nGroups int32; act
// 32*nGroups int8. DotProd required (SDOT), like every kernel in this family.
//
//go:noescape
func w4a8LaneCorrNeg8(act *int8, corr *int32, nGroups int)

// dotW4A8SplitHalf4RowFold is dotW4A8SplitHalf4Row (same layout, same dst,
// same nGroups contract) with corr — w4a8LaneCorrNeg8's output for this act —
// loaded as each row's SDOT starting value in place of MOVI #0, and the two
// per-row VSUBs gone. Bit-identical to dotW4A8SplitHalf4Row for every input.
//
//go:noescape
func dotW4A8SplitHalf4RowFold(act *int8, corr *int32, packed4 *byte, scales4 *float32, dst *float32, nGroups int)

// w4a8LaneCorrNeg8Scalar is the pure-Go reference for w4a8LaneCorrNeg8 (the
// gate's oracle, and the definition of the lane mapping in one place).
func w4a8LaneCorrNeg8Scalar(act []int8, corr []int32, nGroups int) {
	for g := 0; g < nGroups; g++ {
		base := 32 * g
		for l := 0; l < 4; l++ {
			var s int32
			for i := 0; i < 4; i++ {
				s += int32(act[base+4*l+i]) + int32(act[base+16+4*l+i])
			}
			corr[4*g+l] = -8 * s
		}
	}
}
