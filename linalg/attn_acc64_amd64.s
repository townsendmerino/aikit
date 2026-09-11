//go:build amd64

#include "textflag.h"

// AVX2 f64 lane-per-output ports of the acc64 attention kernels
// (docs/task-simd-audit.md S-04's open amd64 half, audit M-11). The arm64
// siblings are in attn_acc64_arm64.s and the Go kernels in matmul_av_acc64.go /
// matmul_qk_acc64.go remain the definition and the oracle.
//
// WHY FMA IS BIT-IDENTICAL HERE, which is the load-bearing part. The Go
// reference computes `acc += w * float64(v[d])` — on amd64 that is a separate
// MULSD and ADDSD, two roundings, because Go does not fuse below GOAMD64=v3.
// VFMADD231PD rounds once. Those differ IN GENERAL — but not here: w and v[d]
// both come from float32, so each has a 24-bit mantissa and their product needs
// at most 48, against float64's 53. THE PRODUCT IS EXACT, so the rounding FMA
// skips is a rounding that never happens, and fused and unfused agree bit for
// bit. This is what "exact f64 products" means in S-04's identity argument, and
// it is the reason these ports are allowed to use FMA at all.
//
// The other half of the argument is unchanged from arm64: one accumulator per
// OUTPUT, never shared, fed in the same key-ascending (resp. dim-ascending)
// order as the Go loop. Floating-point addition depends on the order of adds
// into one accumulator, not on what other accumulators do in between.

// func avAcc64AVX32(scores *float32, vals *float32, nKeys int, rowStrideBytes int, dst *float32)
//
// Folds nKeys keys of ONE 32-dim block of V into 8 YMM f64 accumulators
// (4 lanes each = 32 outputs), key-ascending, and stores the 32 sums narrowed
// to f32 at dst. vals points at the block's first dim of key 0; rows are
// rowStrideBytes apart.
//
// Per key: one broadcast of the score widened to f64, then 8×(VCVTPS2PD +
// VFMADD231PD) — 32 MACs for ~18 instructions, against the Go block's
// load+convert+multiply+add per element.
TEXT ·avAcc64AVX32(SB), NOSPLIT, $0-40
	MOVQ scores+0(FP), SI
	MOVQ vals+8(FP), DI
	MOVQ nKeys+16(FP), CX
	MOVQ rowStrideBytes+24(FP), R8
	MOVQ dst+32(FP), DX

	VXORPD Y0, Y0, Y0
	VXORPD Y1, Y1, Y1
	VXORPD Y2, Y2, Y2
	VXORPD Y3, Y3, Y3
	VXORPD Y4, Y4, Y4
	VXORPD Y5, Y5, Y5
	VXORPD Y6, Y6, Y6
	VXORPD Y7, Y7, Y7

	TESTQ CX, CX
	JZ    avstore

avloop:
	// w = float64(scores[s]) in all four lanes. Broadcast as f32 first, then
	// widen: VCVTPS2PD's 4-lane form reads 128 bits, so the four copies become
	// four identical f64.
	VBROADCASTSS (SI), X9
	VCVTPS2PD    X9, Y8

	VCVTPS2PD   0(DI), Y10
	VFMADD231PD Y10, Y8, Y0
	VCVTPS2PD   16(DI), Y11
	VFMADD231PD Y11, Y8, Y1
	VCVTPS2PD   32(DI), Y10
	VFMADD231PD Y10, Y8, Y2
	VCVTPS2PD   48(DI), Y11
	VFMADD231PD Y11, Y8, Y3
	VCVTPS2PD   64(DI), Y10
	VFMADD231PD Y10, Y8, Y4
	VCVTPS2PD   80(DI), Y11
	VFMADD231PD Y11, Y8, Y5
	VCVTPS2PD   96(DI), Y10
	VFMADD231PD Y10, Y8, Y6
	VCVTPS2PD   112(DI), Y11
	VFMADD231PD Y11, Y8, Y7

	ADDQ $4, SI
	ADDQ R8, DI
	DECQ CX
	JNZ  avloop

avstore:
	VCVTPD2PSY Y0, X9
	VMOVUPS   X9, 0(DX)
	VCVTPD2PSY Y1, X9
	VMOVUPS   X9, 16(DX)
	VCVTPD2PSY Y2, X9
	VMOVUPS   X9, 32(DX)
	VCVTPD2PSY Y3, X9
	VMOVUPS   X9, 48(DX)
	VCVTPD2PSY Y4, X9
	VMOVUPS   X9, 64(DX)
	VCVTPD2PSY Y5, X9
	VMOVUPS   X9, 80(DX)
	VCVTPD2PSY Y6, X9
	VMOVUPS   X9, 96(DX)
	VCVTPD2PSY Y7, X9
	VMOVUPS   X9, 112(DX)
	VZEROUPPER
	RET

// func qkAcc64AVX4(q *float32, rows *float32, rowStrideBytes int, k4 int, nBlocks int, dst *float32)
//
// Computes nBlocks×4 keys' dots against q (K = 4·k4 dims, taken in ascending d),
// ONE KEY PER f64 LANE, and stores them narrowed to f32 at dst. rows points at
// key 0's first dim; rows are rowStrideBytes apart.
//
// Four keys per block rather than arm64's sixteen: a YMM holds 4 f64, and this
// keeps one accumulator register per block with three temporaries, which is
// comfortable on amd64's 16 YMM. The lane-per-KEY layout means the four keys'
// dims must be gathered a lane at a time — there is no strided load — so each
// d-step is 4 scalar widens and one FMA. That still beats the Go chain, which
// pays a convert and a separate multiply and add per key per dim.
TEXT ·qkAcc64AVX4(SB), NOSPLIT, $0-48
	MOVQ q+0(FP), SI
	MOVQ rows+8(FP), DI
	MOVQ rowStrideBytes+16(FP), R8
	MOVQ k4+24(FP), R9
	MOVQ nBlocks+32(FP), R10
	MOVQ dst+40(FP), DX

	TESTQ R10, R10
	JZ    qkdone

qkblock:
	VXORPD Y0, Y0, Y0 // 4 key accumulators, one per lane

	// Row pointers for the four keys in this block.
	MOVQ DI, R11
	MOVQ DI, R12
	ADDQ R8, R12
	MOVQ R12, R13
	ADDQ R8, R13
	MOVQ R13, R14
	ADDQ R8, R14

	MOVQ SI, R15 // q cursor
	MOVQ R9, CX  // k4 groups of 4 dims
	SHLQ $2, CX  // -> dim count K = 4*k4
	TESTQ CX, CX
	JZ    qkstore

qkdim:
	// qv = float64(q[d]) in all four lanes.
	VBROADCASTSS (R15), X5
	VCVTPS2PD    X5, Y1

	// Gather the four keys' d-th element into the four lanes of Y2, in KEY
	// order (lane i = key i) so each lane accumulates exactly one key's dot.
	VMOVSS       (R11), X2
	VINSERTPS    $0x10, (R12), X2, X2
	VINSERTPS    $0x20, (R13), X2, X2
	VINSERTPS    $0x30, (R14), X2, X2
	VCVTPS2PD    X2, Y2

	VFMADD231PD Y2, Y1, Y0

	ADDQ $4, R15
	ADDQ $4, R11
	ADDQ $4, R12
	ADDQ $4, R13
	ADDQ $4, R14
	DECQ CX
	JNZ  qkdim

qkstore:
	VCVTPD2PSY Y0, X3
	VMOVUPS   X3, 0(DX)
	ADDQ      $16, DX

	// Advance to the next block of four keys.
	MOVQ R8, R11
	SHLQ $2, R11
	ADDQ R11, DI

	DECQ R10
	JNZ  qkblock

qkdone:
	VZEROUPPER
	RET
