//go:build !arm64 && !amd64

package linalg

// No vector port of the acc64 attention kernels on this arch: the Go kernels
// run every dim and every key. arm64 has the NEON ports (attn_acc64_arm64.s)
// and amd64 the AVX2 ones (attn_acc64_amd64.s, audit M-11).

func avAcc64Blocks(srow, vals, drow []float32, nKeys, hd, headOff, rowStride int) int { return 0 }

func qkAcc64Keys(arow, bMat, drow []float32, K, N, bOff, rowStride int) int { return 0 }
