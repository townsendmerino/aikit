//go:build arm64

package linalg

import "github.com/townsendmerino/aikit/internal/cpufeat"

// dotPartials16SDOT is the DotProd (SDOT) implementation in kquant_dp_arm64.s; only safe on a
// DotProd-capable core (gated by cpufeat.HasDotProd).
//
//go:noescape
func dotPartials16SDOT(codes *int8, qs *int8, nsub int, out *int32)

// dotPartials16 fills out[j] = Σ_{i<16} codes[j*16+i]·qs[j*16+i] for each of the len(out) sub-blocks.
// SDOT on hardware that has it (Apple Silicon always), else the scalar reference. Integer-exact
// either way. len(codes) == len(qs) == 16·len(out) is the caller's contract.
func dotPartials16(codes, qs []int8, out []int32) {
	if cpufeat.HasDotProd && len(out) > 0 {
		dotPartials16SDOT(&codes[0], &qs[0], len(out), &out[0])
		return
	}
	dotPartials16Scalar(codes, qs, out)
}
