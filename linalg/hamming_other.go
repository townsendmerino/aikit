//go:build !amd64 && !arm64

package linalg

// hammingRows on every non-amd64, non-arm64 arch is the portable kernel.
//
// arm64 has its own file (hamming_arm64.go) with a NEON kernel for the two
// real production shapes (words==4 dim 256, words==12 dim 768) — measured ~3x
// over this same generic implementation, because NEON's VCNT processes 16
// bytes per instruction where Go's (already-intrinsified, VCNT+VADDV)
// OnesCount64 handles 8. Every other word count still falls back to this
// generic implementation from hamming_arm64.go directly, so it has two
// callers on arm64 despite the build tag excluding it there.
func hammingRows(q, codes []uint64, words, n int, dst []uint16) {
	hammingRowsGeneric(q, codes, words, n, dst)
}
