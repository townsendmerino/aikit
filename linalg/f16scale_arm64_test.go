//go:build arm64

package linalg

// layoutRefF32 is the f32-scale reference for a repacked WeightMat's own layout (row4 here), following the
// arm64 WeightMat.MatmulBTW4A8Into dispatch; nil when wm has no repacked layout.
func layoutRefF32(ws *Workspace, a []float32, wm *WeightMat, q4 []byte, s32 []float32, M, K, N int) []float32 {
	if wm.q4Row4 == nil {
		return nil
	}
	s4 := RepackW4A8Row4Scales(s32, N, K, 32)
	out := make([]float32, M*N)
	if g := wm.groupFor(ws); g > 0 {
		if M == 1 {
			MatmulBTW4A8Row4Into(ws, a, wm.q4Row4, s4, out, M, K, N, 32)
			return out
		}
		matmulW4A8Grouped(ws, g, a, int4Layout{w4: q4, r4: wm.q4Row4, wS: s32, r4S: s4, group: 32, K: K}, out, M, N)
		return out
	}
	if M == 1 {
		MatmulBTW4A8Row4Into(ws, a, wm.q4Row4, s4, out, M, K, N, 32)
	} else {
		MatmulBTW4A8Row4TileInto(ws, a, wm.q4Row4, s4, out, M, K, N, 32)
	}
	return out
}

// batchRow4RefF32 is MatmulBTW4A8Batch over the same row4 op with f32 interleaved scales.
func batchRow4RefF32(ws *Workspace, a []float32, q4 []byte, s32 []float32, wm *WeightMat, M, K, N int) []float32 {
	out := make([]float32, M*N)
	MatmulBTW4A8Batch(ws, a, M, K, 32, []W4A8Op{{W4: q4, Scales: s32, Row4: wm.q4Row4, Row4Scales: RepackW4A8Row4Scales(s32, N, K, 32), Dst: out, N: N}})
	return out
}
