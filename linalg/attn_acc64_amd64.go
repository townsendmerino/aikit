//go:build amd64

package linalg

// AVX2 f64 lane-per-output ports of the acc64 attention kernels (audit M-11,
// docs/task-simd-audit.md S-04's open amd64 half). The identity argument — and
// in particular why FMA is bit-identical to the Go reference's separate
// multiply and add here — is in attn_acc64_amd64.s. The Go kernels in
// matmul_av_acc64.go and matmul_qk_acc64.go stay as the definition, the
// fallback for shapes the kernels do not cover, and the oracle the tests
// compare against.

// avAcc64AVX32 folds nKeys keys of one 32-dim block of V into 8 YMM f64
// accumulators (4 lanes each), key-ascending, and stores the 32 sums narrowed
// to f32 at dst. vals points at the block's first dim of key 0; rows are
// rowStrideBytes apart. Only safe when hasAVX2.
//
//go:noescape
func avAcc64AVX32(scores *float32, vals *float32, nKeys int, rowStrideBytes int, dst *float32)

// qkAcc64AVX4 computes nBlocks×4 keys' dots against q (K = 4·k4 dims, ascending
// d), one key per f64 lane, and stores them narrowed to f32 at dst. rows points
// at key 0's first dim; rows are rowStrideBytes apart. Only safe when hasAVX2.
//
//go:noescape
func qkAcc64AVX4(q *float32, rows *float32, rowStrideBytes int, k4 int, nBlocks int, dst *float32)

// qkAcc64AVX16 is qkAcc64AVX4 over n16 groups of sixteen keys with four accumulators in flight; the same sums, bit for bit
// (attn_acc64_amd64.s). Only safe when hasAVX2.
//
//go:noescape
func qkAcc64AVX16(q *float32, rows *float32, rowStrideBytes int, k4 int, n16 int, dst *float32)

// avAcc64Blocks runs one query row's scores·V through the AVX2 32-dim kernel
// for as many whole 32-dim blocks as hd holds and returns the first dim it did
// NOT compute; the caller finishes from there with the Go 16-block and tail
// code, which is the same arithmetic in the same order. With no keys there is
// nothing to point the kernel at (vals may be empty), so the Go path writes the
// zeros — the same shape the arm64 sibling takes.
func avAcc64Blocks(srow, vals, drow []float32, nKeys, hd, headOff, rowStride int) int {
	if !hasAVX2 || nKeys == 0 {
		return 0
	}
	d0 := 0
	for ; d0+32 <= hd; d0 += 32 {
		avAcc64AVX32(&srow[0], &vals[headOff+d0], nKeys, rowStride*4, &drow[d0])
	}
	return d0
}

// qkAcc64Keys runs one query row's q·Kᵀ through the AVX2 16-key kernel, then the 4-key kernel, for as
// many whole 4-key blocks as N holds and returns the number of keys it wrote;
// the caller finishes the N%4 tail with the Go 8-chain loop. The kernel walks d
// one at a time but requires K%4==0 for parity with the arm64 sibling's
// contract and with the Go tail's expectations (no production head dim is not a
// multiple of 4).
func qkAcc64Keys(arow, bMat, drow []float32, K, N, bOff, rowStride int) int {
	if !hasAVX2 || K == 0 || K%4 != 0 {
		return 0
	}
	j := qkAcc64Keys16(arow, bMat, drow, K, N, bOff, rowStride)
	nBlocks := (N - j) / 4
	if nBlocks == 0 {
		return j
	}
	qkAcc64AVX4(&arow[0], &bMat[bOff+j*rowStride], rowStride*4, K/4, nBlocks, &drow[j])
	return j + nBlocks*4
}

// qkAcc64Keys16 runs whole sixteen-key groups through qkAcc64AVX16 and returns the keys it wrote (a multiple of 16; 0 where the
// kernel declines). qkAcc64Keys finishes the rest four keys at a time.
func qkAcc64Keys16(arow, bMat, drow []float32, K, N, bOff, rowStride int) int {
	if !hasAVX2 || K == 0 || K%4 != 0 {
		return 0
	}
	n16 := N / 16
	if n16 == 0 {
		return 0
	}
	qkAcc64AVX16(&arow[0], &bMat[bOff], rowStride*4, K/4, n16, &drow[0])
	return n16 * 16
}

// qkAcc64AVX16F16 is qkAcc64AVX16 over a key cache of IEEE binary16 bits: each half widened exactly with VCVTPH2PS, then the same
// four accumulators and d-ascending chains (attn_acc64_amd64.s). Only safe when hasAVX2 && hasF16C.
//
//go:noescape
func qkAcc64AVX16F16(q *float32, rows *uint16, rowStrideBytes int, k4 int, n16 int, dst *float32)

// avAcc64AVX32F16 is avAcc64AVX32 over a value cache of IEEE binary16 bits: four halves a VCVTPH2PS, widened exactly, then the same
// eight accumulators and key-ascending folds. Only safe when hasAVX2 && hasF16C.
//
//go:noescape
func avAcc64AVX32F16(scores *float32, vals *uint16, nKeys int, rowStrideBytes int, dst *float32)

// qkAcc64KeysF16 runs whole sixteen-key groups of an f16 key cache through qkAcc64AVX16F16 and returns the keys it wrote;
// MatmulQKAcc64F16's Go loop finishes the rest.
func qkAcc64KeysF16(arow []float32, bMat []uint16, drow []float32, K, N, bOff, rowStride int) int {
	if !hasAVX2 || !hasF16C || K == 0 || K%4 != 0 || N < 16 {
		return 0
	}
	n16 := N / 16
	qkAcc64AVX16F16(&arow[0], &bMat[bOff], rowStride*2, K/4, n16, &drow[0])
	return n16 * 16
}

// avAcc64BlocksF16 runs one query row's scores·V over an f16 value cache through avAcc64AVX32F16 for as many whole 32-dim blocks as hd
// holds and returns the first dim it did not compute; MatmulAVAcc64F16's Go blocks finish from there.
func avAcc64BlocksF16(srow []float32, vals []uint16, drow []float32, nKeys, hd, headOff, rowStride int) int {
	if !hasAVX2 || !hasF16C || nKeys == 0 {
		return 0
	}
	d0 := 0
	for ; d0+32 <= hd; d0 += 32 {
		avAcc64AVX32F16(&srow[0], &vals[headOff+d0], nKeys, rowStride*2, &drow[d0])
	}
	return d0
}
