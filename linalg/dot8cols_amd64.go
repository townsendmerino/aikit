//go:build amd64

package linalg

import "github.com/townsendmerino/aikit/internal/cpufeat"

// dot8ColsInto computes the eight dot products of a against b0..b7 straight into
// out, bypassing Dot8x4's [32]float32 partial-sum interface.
//
// WHY THIS EXISTS. That interface is arm64-shaped: there, dotNEON8x4 leaves four
// live 4-lane accumulators per column and the caller folds them. On amd64 dotFMA8
// already finishes the horizontal reduction in-register (a VEXTRACTF128 + VADDPS +
// two VHADDPS per column) and writes eight final scalars. The shim then zeroed 128
// bytes, scattered those eight finals to sums[0], sums[4] … sums[28], and had the
// caller re-read all 32 and add — where 24 of the 32 adds were `x + 0.0`.
//
// Measured cost of that round trip at K=768: 156.1 ns → 144.6 ns per 8-column call,
// ~7.4% of the kernel.
//
// NUMERICS. `x + 0.0` is exact in round-to-nearest for every x except -0.0, where it
// yields +0.0. So removing the adds is bit-identical unless a column's dot product is
// exactly -0.0 — which needs every one of its lane sums to be -0.0, hence every
// product a[k]·b[k] to be a negatively-signed zero. The existing whole-corpus parity
// gates cover the claim; this note records the one input class where the two forms
// could differ, and that the new form is the more faithful of the two (it preserves
// the sign the kernel actually produced).
func dot8ColsInto(a, b0, b1, b2, b3, b4, b5, b6, b7 *float32, n4 int, out *[8]float32) {
	if !cpufeat.HasAVX2 {
		dot8ColsGeneric(a, b0, b1, b2, b3, b4, b5, b6, b7, n4, out)
		return
	}
	dotFMA8(a, b0, b1, b2, b3, b4, b5, b6, b7, n4*4, out)
}
