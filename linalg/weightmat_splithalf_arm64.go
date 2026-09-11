//go:build arm64

package linalg

// RepackInt4SplitHalf is a no-op on arm64. The split-half-only layout exists for the AVX2
// kernel; arm64's own split-half work ships as the 4-row-interleaved variant instead
// (RepackInt4Row4), and dotW4A8SplitHalfSDOT measured a FLAT 1.000x on its own there because
// that kernel is latency-bound rather than port-bound. Always returns false.
func (w *WeightMat) RepackInt4SplitHalf() bool { return false }

// splitHalfUsable is always false on arm64 — the split-half AVX2 kernel is
// amd64-only; arm64's own split-half work ships as row4 instead. See
// row4Usable (weightmat_row4_arm64.go) for that gate. Audit M-22.
func splitHalfUsable() bool { return false }
