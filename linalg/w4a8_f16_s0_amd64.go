//go:build amd64

package linalg

// S0 prototype, see w4a8_f16_s0.go.

//go:noescape
func dotW4A8FoldF16AVX2(act *int8, packed *byte, scales *uint16, nGroups int) float32

var hasW4A8F16AVX2 = hasAVX2 && detectF16C()

// dotW4A8F16 is dotW4A8 over f16 scales: the AVX2+F16C kernel when it applies (the same host and
// shape conditions under which dotW4A8 takes dotW4A8FoldAVX2), else the Go fallback, which widens the
// scales and calls dotW4A8 itself — bit-identical either way.
func dotW4A8F16(act []int8, packed []byte, scales []uint16, group, K int) float32 {
	if group == 32 && K >= 32 && K%32 == 0 && hasW4A8F16AVX2 && !hasAVX512VNNIVL {
		return dotW4A8FoldF16AVX2(&act[0], &packed[0], &scales[0], K/32)
	}
	return dotW4A8F16Go(act, packed, scales, group, K)
}

// (*WeightMat).MatmulBTW4A8Into routes here first (weightmat_splithalf_amd64.go).
func (w *WeightMat) matmulBTW4A8F16S0(ws *Workspace, a, dst []float32, M int) bool {
	if !int4F16Scales || w.q4sF16 == nil || w.q4 == nil || w.q4SplitHalf != nil || M >= 4 {
		return false
	}
	MatmulBTW4A8F16Into(ws, a, w.q4, w.q4sF16, dst, M, w.cols, w.rows, w.group)
	return true
}
