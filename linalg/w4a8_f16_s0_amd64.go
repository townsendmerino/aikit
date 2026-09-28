//go:build amd64

package linalg

// S0 prototype, see w4a8_f16_s0.go.

//go:noescape
func dotW4A8FoldF16AVX2(act *int8, packed *byte, scales *uint16, nGroups int) float32

var hasW4A8F16AVX2 = hasAVX2 && detectF16C()

//go:noescape
func cvtF16ToF32x8F16C(dst *float32, src *uint16, n8 int)

//go:noescape
func dotW4A8FoldF16RowAVX2(act *int8, packed *byte, scales *uint16, nGroups int, buf *float32) float32

// dotW4A8F16Row is the row-widening variant: one asm call per row, buf holds >= nGroups floats.
func dotW4A8F16Row(act []int8, packed []byte, scales []uint16, group, K int, buf []float32) float32 {
	if group == 32 && K%32 == 0 && K/32 >= 8 && K/32 <= len(buf) && hasW4A8F16AVX2 && !hasAVX512VNNIVL {
		return dotW4A8FoldF16RowAVX2(&act[0], &packed[0], &scales[0], K/32, &buf[0])
	}
	return dotW4A8F16(act, packed, scales, group, K)
}

// f16S0PerGroup selects the per-group-conversion kernel (dotW4A8FoldF16AVX2); false (the default)
// converts a weight row's scales once, eight per VCVTPH2PS, into an L1 buffer and runs the f32 kernel.
var f16S0PerGroup = false

// SetInt4F16PerGroupS0 picks the S0 kernel variant (benchmarks and the goinfer A/B).
func SetInt4F16PerGroupS0(on bool) { f16S0PerGroup = on }

// widenScalesF16 fills dst[:len(src)] with the exact f32 values of src.
func widenScalesF16(dst []float32, src []uint16) {
	n8 := len(src) / 8
	if n8 > 0 && hasW4A8F16AVX2 {
		cvtF16ToF32x8F16C(&dst[0], &src[0], n8)
	} else {
		n8 = 0
	}
	for i := n8 * 8; i < len(src); i++ {
		dst[i] = f16ToF32(src[i])
	}
}

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
