//go:build amd64

package linalg

import "github.com/townsendmerino/aikit/internal/cpufeat"

// dequantI8AVX2 widens n int8 values (n a multiple of 8) into dst, scaling each
// by scale: dst[i] = float32(q[i]) * scale. VPMOVSXBD → VCVTDQ2PS → VMULPS, 32
// elements per main-loop iteration. Implemented in dequant_i8_amd64.s.
//
//go:noescape
func dequantI8AVX2(q *int8, dst *float32, n int, scale float32)

// dequantRowInt8 widens one row. The AVX2 kernel handles the 8-aligned prefix
// and the Go tail finishes the remainder, so no assembly tail logic is needed
// (every real weight row here is a multiple of 8 anyway — K is 768/2048/3072).
func dequantRowInt8(dst []float32, q []int8, scale float32) {
	if !cpufeat.HasAVX2 || len(q) < 8 {
		dequantRowInt8Scalar(dst, q, scale)
		return
	}
	n := len(q) &^ 7
	dequantI8AVX2(&q[0], &dst[0], n, scale)
	if n < len(q) {
		dequantRowInt8Scalar(dst[n:], q[n:], scale)
	}
}
