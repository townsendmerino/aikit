//go:build !arm64

package linalg

import "unsafe"

// This file holds the architecture-neutral scalar kernels. It builds
// on every non-arm64 target (amd64 included), so the amd64 AVX2 path
// in dot_amd64.go can fall back to these when the running CPU lacks
// AVX2/FMA, and the catch-all path in dot_other.go can alias straight
// to them. arm64 never sees this file — it has its own NEON asm.
//
// The per-row CONTRACT matches the arm64 asm — each row's dot lives in
// its own 4-lane block of sums, which the caller horizontal-sums — but
// the per-lane layout WITHIN a block differs: this generic path stores
// the full dot in lane 0 and zeros the other three, whereas the arm64
// kernel leaves four raw partial sums. The two agree only AFTER the
// caller's 4-lane reduction; don't read individual lanes.

// dotGeneric returns the sum over the first n4*4 elements of a and b.
func dotGeneric(a *float32, b *float32, n4 int) float32 {
	n := n4 * 4
	aSlice := unsafe.Slice(a, n)
	bSlice := unsafe.Slice(b, n)
	var sum float32
	for i := 0; i < n; i++ {
		sum += aSlice[i] * bSlice[i]
	}
	return sum
}

// dot4x4Generic computes 4 dot products (a·b0..a·b3) over the first
// n4*4 elements, storing each full sum in the first lane of its block.
func dot4x4Generic(a, b0, b1, b2, b3 *float32, n4 int, sums *[16]float32) {
	n := n4 * 4
	aS := unsafe.Slice(a, n)
	b0S := unsafe.Slice(b0, n)
	b1S := unsafe.Slice(b1, n)
	b2S := unsafe.Slice(b2, n)
	b3S := unsafe.Slice(b3, n)
	var s0, s1, s2, s3 float32
	for i := 0; i < n; i++ {
		s0 += aS[i] * b0S[i]
		s1 += aS[i] * b1S[i]
		s2 += aS[i] * b2S[i]
		s3 += aS[i] * b3S[i]
	}
	*sums = [16]float32{}
	sums[0] = s0
	sums[4] = s1
	sums[8] = s2
	sums[12] = s3
}

// dot8x4Generic computes 8 dot products (a·b0..a·b7) over the first
// n4*4 elements, storing each full sum in the first lane of its block.
func dot8x4Generic(a, b0, b1, b2, b3, b4, b5, b6, b7 *float32, n4 int, sums *[32]float32) {
	n := n4 * 4
	aS := unsafe.Slice(a, n)
	bS := [8][]float32{
		unsafe.Slice(b0, n),
		unsafe.Slice(b1, n),
		unsafe.Slice(b2, n),
		unsafe.Slice(b3, n),
		unsafe.Slice(b4, n),
		unsafe.Slice(b5, n),
		unsafe.Slice(b6, n),
		unsafe.Slice(b7, n),
	}
	var s [8]float32
	for i := 0; i < n; i++ {
		av := aS[i]
		for r := 0; r < 8; r++ {
			s[r] += av * bS[r][i]
		}
	}
	*sums = [32]float32{}
	for r := 0; r < 8; r++ {
		sums[r*4] = s[r]
	}
}
