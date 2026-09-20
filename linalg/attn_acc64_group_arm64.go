//go:build arm64

package linalg

// avAcc64NEON8G6 folds nKeys keys of one 8-dim block of V into 24 f64
// lane-pair accumulators (G=6 queries × 4 dim-pairs), key-ascending, and
// stores the 6×8 sums narrowed to f32 at dst. vals points at the block's
// first dim of key 0; V rows are rowStrideBytes apart. scores points at
// query 0's score row; score rows are scoresRowStrideBytes apart. dst points
// at query 0's block; dst rows are dstRowStrideBytes apart.
//
//go:noescape
func avAcc64NEON8G6(scores *float32, scoresRowStrideBytes int, vals *float32, nKeys int, rowStrideBytes int, dst *float32, dstRowStrideBytes int)

// avAcc64GroupBlocks runs G=6 queries' scores·V through the NEON 8-dim block
// kernel for as many whole 8-dim blocks as hd holds and returns the first dim
// it did NOT compute; the caller finishes from there with the Go per-key
// loop, same arithmetic, same key-ascending order. Only G=6 (the 1.5B's real
// GQA group size, this brief's registered decision shape) has a NEON port;
// every other G returns 0 and MatmulAVAcc64Group's own Go loop does
// everything, matching the fallback discipline attn_acc64_arm64.go already
// established for the per-head kernels.
func avAcc64GroupBlocks(scores, vals, dst []float32, G, nKeys, hd, headOff, rowStride int) int {
	if G != 6 || nKeys == 0 {
		return 0
	}
	d0 := 0
	for ; d0+8 <= hd; d0 += 8 {
		avAcc64NEON8G6(&scores[0], nKeys*4, &vals[headOff+d0], nKeys, rowStride*4, &dst[d0], hd*4)
	}
	return d0
}

// qkAcc64NEON8G6 computes nBlocks×8 keys' dots against G=6 queries (K = 4·k4
// dims, taken in ascending d), storing them narrowed to f32 at dst — dst is
// [G,N] row-major (dstRowStrideBytes between query rows), unlike
// avAcc64NEON8G6's [G,hd]. q points at query 0's row; query rows are
// qStrideBytes apart. rows points at key 0's first dim of the first block;
// K rows are rowStrideBytes apart.
//
//go:noescape
func qkAcc64NEON8G6(q *float32, qStrideBytes int, rows *float32, rowStrideBytes int, k4 int, nBlocks int, dst *float32, dstRowStrideBytes int)

// qkAcc64GroupKeys runs G=6 queries' q·Kᵀ through the NEON 8-key block kernel
// for as many whole 8-key blocks as N holds and returns the number of keys it
// wrote; the caller finishes the N%8 tail with the Go per-head chain. The
// kernel walks d four at a time (like qkAcc64Keys), so a K not a multiple of
// 4 takes the Go path entirely. Only G=6 has a NEON port; every other G
// returns 0, matching avAcc64GroupBlocks's own fallback discipline.
func qkAcc64GroupKeys(a, bMat, dst []float32, G, K, N, bOff, rowStride int) int {
	if G != 6 || K == 0 || K%4 != 0 {
		return 0
	}
	nBlocks := N / 8
	if nBlocks == 0 {
		return 0
	}
	qkAcc64NEON8G6(&a[0], K*4, &bMat[bOff], rowStride*4, K/4, nBlocks, &dst[0], N*4)
	return nBlocks * 8
}
