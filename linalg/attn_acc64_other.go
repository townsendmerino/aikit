//go:build !arm64

package linalg

// No vector port of the acc64 attention kernels on this arch: the Go kernels
// run every dim and every key. (The lane-per-output argument in
// attn_acc64_arm64.s holds for AVX2 too — the products are exact in f64 there
// as well — but no amd64 kernel has been written; that is task-simd-audit.md
// S-04's open amd64 half.)

func avAcc64Blocks(srow, vals, drow []float32, nKeys, hd, headOff, rowStride int) int { return 0 }

func qkAcc64Keys(arow, bMat, drow []float32, K, N, bOff, rowStride int) int { return 0 }
