//go:build amd64

package linalg

import "github.com/townsendmerino/aikit/internal/cpufeat"

// dotI8 is the SIMD-dispatched int8×int8→int32 inner product used by the W8A8
// matmul. Three tiers, each peeling its own multiple off the front and
// handing the remainder to the next: AVX-512 VNNI (64-multiple,
// dotI8AVX512VNNI in dot_i8_avx512vnni_amd64.s) when cpufeat.HasAVX512VNNI, then AVX2
// (16-multiple, dotI8AVX2 in dot_amd64.s) when cpufeat.HasAVX2, then the scalar
// reference for whatever's left below 16. Without either extension it is all
// scalar. Output is identical to dotI8Scalar (integer arithmetic — exact, no
// reassociation) regardless of which tiers a given n exercises.
func dotI8(a, b []int8) int32 {
	n := len(a)
	i := 0
	var s int32
	if cpufeat.HasAVX512VNNI {
		if n64 := n &^ 63; n64 >= 64 {
			s = dotI8AVX512VNNI(&a[0], &b[0], n64)
			i = n64
		}
	}
	if cpufeat.HasAVX2 {
		if rem := n - i; rem >= 16 {
			n16 := rem &^ 15
			s += dotI8AVX2(&a[i], &b[i], n16)
			i += n16
		}
	}
	if i < n {
		s += dotI8Scalar(a[i:], b[i:])
	}
	return s
}
