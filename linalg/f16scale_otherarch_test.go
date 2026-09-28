//go:build !amd64 && !arm64

package linalg

func layoutRefF32(ws *Workspace, a []float32, wm *WeightMat, q4 []byte, s32 []float32, M, K, N int) []float32 {
	return nil
}

func batchRow4RefF32(ws *Workspace, a []float32, q4 []byte, s32 []float32, wm *WeightMat, M, K, N int) []float32 {
	return nil
}
