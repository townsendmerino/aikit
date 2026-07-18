package linalg

import "fmt"

// DotI8 returns the int32 dot product of two equal-length int8 vectors via the
// platform SIMD kernel (AVX2 VPMADDWD on amd64, SDOT on ARMv8.2, scalar fallback
// elsewhere). It is the node-node integer similarity primitive for the int8 ann
// indexes (rescale by the two per-vector scales to recover cosine).
// Experimental, like the rest of linalg. (QuantizeRowInt8, the single-row
// f32→int8 quantizer for a query vector, is already exported in quant.go.)
//
// Panics if len(a) != len(b): the SIMD dispatch reads n=len(a) elements of both,
// so a short b would read past its allocation (silent garbage or a fault) on the
// vector path and bounds-panic on the scalar tail — an arch-dependent failure
// mode. The check makes it uniform and cheap (matches dotF32).
func DotI8(a, b []int8) int32 {
	if len(a) != len(b) {
		panic(fmt.Sprintf("linalg: DotI8 length mismatch (%d != %d)", len(a), len(b)))
	}
	return dotI8(a, b)
}
