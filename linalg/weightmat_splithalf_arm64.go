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

// w4a8BatchSplitHalfSpan is unreachable off amd64: quant.go's w4a8BatchOp
// only calls it behind splitHalfUsable(), which is a constant false here.
// Audit M-22 follow-up. See weightmat_splithalf_amd64.go for the real body.
func w4a8BatchSplitHalfSpan(aq []int8, aScale float32, splitHalf []byte, scales, dst []float32, K, N, nGroups, bpr, j0, j1 int) {
	panic("linalg: w4a8BatchSplitHalfSpan reached off amd64 — splitHalfUsable() should have gated this")
}
