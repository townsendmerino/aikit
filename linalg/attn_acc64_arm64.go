//go:build arm64

package linalg

// NEON f64 lane-per-output ports of the acc64 attention kernels
// (task-simd-audit.md S-04, step 2). The identity argument and the instruction
// accounting are in attn_acc64_arm64.s; the Go kernels in matmul_av_acc64.go and
// matmul_qk_acc64.go stay as the definition, the fallback for the shapes the
// kernels do not cover, and the oracle the tests compare against.

// avAcc64NEON32 folds nKeys keys of one 32-dim block of V into 16 f64 lane-pair
// accumulators, key-ascending, and stores the 32 sums narrowed to f32 at dst.
// vals points at the block's first dim of key 0; rows are rowStrideBytes apart.
//
//go:noescape
func avAcc64NEON32(scores *float32, vals *float32, nKeys int, rowStrideBytes int, dst *float32)

// qkAcc64NEON16 computes nBlocks×16 keys' dots against q (K = 4·k4 dims, taken in
// ascending d), one key per f64 lane, and stores them narrowed to f32 at dst.
// rows points at key 0's first dim; rows are rowStrideBytes apart.
//
//go:noescape
func qkAcc64NEON16(q *float32, rows *float32, rowStrideBytes int, k4 int, nBlocks int, dst *float32)

// avAcc64Blocks runs one query row's scores·V through the NEON 32-dim kernel for
// as many whole 32-dim blocks as hd holds and returns the first dim it did NOT
// compute; the caller finishes from there with the Go 16-block and tail code,
// which is the same arithmetic in the same order. With no keys there is nothing
// to point the kernel at (vals may be empty), so the Go path writes the zeros.
func avAcc64Blocks(srow, vals, drow []float32, nKeys, hd, headOff, rowStride int) int {
	if nKeys == 0 {
		return 0
	}
	d0 := 0
	for ; d0+32 <= hd; d0 += 32 {
		avAcc64NEON32(&srow[0], &vals[headOff+d0], nKeys, rowStride*4, &drow[d0])
	}
	return d0
}

// qkAcc64Keys runs one query row's q·Kᵀ through the NEON 16-key kernel for as
// many whole 16-key blocks as N holds and returns the number of keys it wrote;
// the caller finishes the N%16 tail with the Go 8-chain loop. The kernel walks d
// four at a time, so a K that is not a multiple of 4 (no production head dim is)
// takes the Go path entirely.
func qkAcc64Keys(arow, bMat, drow []float32, K, N, bOff, rowStride int) int {
	if K == 0 || K%4 != 0 {
		return 0
	}
	nBlocks := N / 16
	if nBlocks == 0 {
		return 0
	}
	qkAcc64NEON16(&arow[0], &bMat[bOff], rowStride*4, K/4, nBlocks, &drow[0])
	return nBlocks * 16
}
