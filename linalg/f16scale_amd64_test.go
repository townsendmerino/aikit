//go:build amd64

package linalg

func f16FusedKernelActive() bool { return hasAVX2 && hasF16C && !hasAVX512VNNIVL }

// layoutRefF32 is the f32-scale reference for a repacked WeightMat's own layout (split-half here), following
// the amd64 WeightMat.MatmulBTW4A8Into dispatch; nil when wm has no repacked layout.
func layoutRefF32(ws *Workspace, a []float32, wm *WeightMat, q4 []byte, s32 []float32, M, K, N int) []float32 {
	if wm.q4SplitHalf == nil {
		return nil
	}
	out := make([]float32, M*N)
	if g := wm.groupFor(ws); g > 0 {
		matmulW4A8Grouped(ws, g, a, int4Layout{w4: q4, sh: wm.q4SplitHalf, wS: s32, group: 32, K: K}, out, M, N)
		return out
	}
	if M == 1 {
		matmulBTW4A8SplitHalfInto(ws, a, wm.q4SplitHalf, s32, out, K, N, 32)
	} else {
		matmulBTW4A8SplitHalfMultiInto(ws, a, wm.q4SplitHalf, s32, out, M, K, N, 32)
	}
	return out
}

// batchRow4RefF32 is unused on amd64 (no row4 layout).
func batchRow4RefF32(ws *Workspace, a []float32, q4 []byte, s32 []float32, wm *WeightMat, M, K, N int) []float32 {
	return nil
}
