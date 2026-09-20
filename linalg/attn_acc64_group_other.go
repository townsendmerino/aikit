//go:build !arm64

package linalg

// No vector port of the grouped acc64 attention kernels outside arm64: the Go
// kernels (matmul_av_acc64_group.go, matmul_qk_acc64_group.go) run every dim,
// key and query. arm64 has the G=6 NEON port (attn_acc64_group_arm64.s).

func avAcc64GroupBlocks(scores, vals, dst []float32, G, nKeys, hd, headOff, rowStride int) int {
	return 0
}

func qkAcc64GroupKeys(a, bMat, dst []float32, G, K, N, bOff, rowStride int) int {
	return 0
}
