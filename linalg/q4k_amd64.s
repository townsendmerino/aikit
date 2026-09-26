#include "textflag.h"

DATA q4kMask0F<>+0(SB)/8, $0x0f0f0f0f0f0f0f0f
DATA q4kMask0F<>+8(SB)/8, $0x0f0f0f0f0f0f0f0f
DATA q4kMask0F<>+16(SB)/8, $0x0f0f0f0f0f0f0f0f
DATA q4kMask0F<>+24(SB)/8, $0x0f0f0f0f0f0f0f0f
GLOBL q4kMask0F<>(SB), RODATA|NOPTR, $32

DATA q4kOnes16<>+0(SB)/8, $0x0001000100010001
DATA q4kOnes16<>+8(SB)/8, $0x0001000100010001
DATA q4kOnes16<>+16(SB)/8, $0x0001000100010001
DATA q4kOnes16<>+24(SB)/8, $0x0001000100010001
GLOBL q4kOnes16<>(SB), RODATA|NOPTR, $32

// func dotQ4KAVX2(row *byte, aq *int8, aS *float32, asumf *float32, nSB int) float32
//
// One row of Q4_K weights (nSB super-blocks of 144 bytes) against one per-32 int8 activation row
// (q4k.go). Per super-block: the eight 32-weight groups' integer dots, each VPMADDUBSW (unsigned
// codes × signed activations, pairwise ≤ 2·15·127, no saturation) then VPMADDWD by ones to 8 int32
// lanes; a VPHADDD tree plus one VPERM2I128 folds the eight lane-vectors into one vector of the eight
// group sums; then in f32
//
//	acc += (d · sc_g · aS_g) · idot_g − dmin · m_g · asumf_g,   asumf_g = aS_g · Σ aq_g,
//
// eight groups per FMA. The 6-bit scales and mins are unpacked with ggml's utmp masks (the
// get_scale_min_k4 layout) and widened with VPMOVZXBD; d and dmin with VCVTPH2PS (F16C).
TEXT ·dotQ4KAVX2(SB), NOSPLIT, $0-44
	MOVQ row+0(FP), SI
	MOVQ aq+8(FP), DI
	MOVQ aS+16(FP), DX
	MOVQ asumf+24(FP), BX
	MOVQ nSB+32(FP), CX
	VXORPS  Y0, Y0, Y0
	VMOVDQU q4kMask0F<>(SB), Y15
	VMOVDQU q4kOnes16<>(SB), Y14
	TESTQ   CX, CX
	JLE     q4k_done

q4k_sb:
	// groups 0 (low nibbles) and 1 (high nibbles): codes qs[0:32], activations aq[0:32], aq[32:64]
	VMOVDQU    16(SI), Y12
	VPSRLW     $4, Y12, Y13
	VPAND      Y15, Y12, Y12
	VPAND      Y15, Y13, Y13
	VPMADDUBSW 0(DI), Y12, Y1
	VPMADDWD   Y14, Y1, Y1
	VPMADDUBSW 32(DI), Y13, Y2
	VPMADDWD   Y14, Y2, Y2

	// groups 2, 3
	VMOVDQU    48(SI), Y12
	VPSRLW     $4, Y12, Y13
	VPAND      Y15, Y12, Y12
	VPAND      Y15, Y13, Y13
	VPMADDUBSW 64(DI), Y12, Y3
	VPMADDWD   Y14, Y3, Y3
	VPMADDUBSW 96(DI), Y13, Y4
	VPMADDWD   Y14, Y4, Y4

	// groups 4, 5
	VMOVDQU    80(SI), Y12
	VPSRLW     $4, Y12, Y13
	VPAND      Y15, Y12, Y12
	VPAND      Y15, Y13, Y13
	VPMADDUBSW 128(DI), Y12, Y5
	VPMADDWD   Y14, Y5, Y5
	VPMADDUBSW 160(DI), Y13, Y6
	VPMADDWD   Y14, Y6, Y6

	// groups 6, 7
	VMOVDQU    112(SI), Y12
	VPSRLW     $4, Y12, Y13
	VPAND      Y15, Y12, Y12
	VPAND      Y15, Y13, Y13
	VPMADDUBSW 192(DI), Y12, Y7
	VPMADDWD   Y14, Y7, Y7
	VPMADDUBSW 224(DI), Y13, Y8
	VPMADDWD   Y14, Y8, Y8

	// Fold: after two VPHADDD levels, Y1 = [g0..g3 lanes 0-3 | g0..g3 lanes 4-7] and Y5 likewise
	// for g4..g7; swapping 128-bit halves and adding leaves Y9 = [g0 .. g7] full sums.
	VPHADDD    Y2, Y1, Y1
	VPHADDD    Y4, Y3, Y3
	VPHADDD    Y3, Y1, Y1
	VPHADDD    Y6, Y5, Y5
	VPHADDD    Y8, Y7, Y7
	VPHADDD    Y7, Y5, Y5
	VPERM2I128 $0x20, Y5, Y1, Y9
	VPERM2I128 $0x31, Y5, Y1, Y10
	VPADDD     Y10, Y9, Y9
	VCVTDQ2PS  Y9, Y9

	// d (lane 0) and dmin (lane 1)
	VMOVD        0(SI), X1
	VCVTPH2PS    X1, X1
	VBROADCASTSS X1, Y2
	VMOVSHDUP    X1, X3
	VBROADCASTSS X3, Y3

	// 6-bit scales and mins: u0, u1, u2 = the 12 scale bytes as three uint32
	MOVL 4(SI), R8
	MOVL 8(SI), R9
	MOVL 12(SI), R10
	MOVL R8, R11
	ANDL $0x3f3f3f3f, R11 // scales 0..3
	MOVL R10, R12
	ANDL $0x0f0f0f0f, R12
	MOVL R8, AX
	SHRL $6, AX
	ANDL $0x03030303, AX
	SHLL $4, AX
	ORL  AX, R12          // scales 4..7
	MOVL R9, R13
	ANDL $0x3f3f3f3f, R13 // mins 0..3
	MOVL R10, AX
	SHRL $4, AX
	ANDL $0x0f0f0f0f, AX
	MOVL R9, R8
	SHRL $6, R8
	ANDL $0x03030303, R8
	SHLL $4, R8
	ORL  R8, AX           // mins 4..7
	SHLQ $32, R12
	ORQ  R11, R12
	SHLQ $32, AX
	ORQ  R13, AX
	// VMOVQ, NOT MOVQ: the Go assembler encodes `MOVQ reg, X` as legacy SSE, and a legacy-SSE write
	// to an XMM register while the upper YMM halves are dirty cost this loop ~16x on a Ryzen 7 3700X
	// (Zen 2): 1900 ns vs 115 ns per 14-super-block row (BenchmarkQ4KDotRow, 2026-09-26). Inside an
	// AVX loop, every XMM write must be VEX-encoded.
	VMOVQ R12, X4
	VPMOVZXBD X4, Y4
	VCVTDQ2PS Y4, Y4
	VMOVQ AX, X5
	VPMOVZXBD X5, Y5
	VCVTDQ2PS Y5, Y5

	// acc += d·sc·aS ⊙ idot ; acc -= dmin · m ⊙ asumf
	VMULPS       0(DX), Y4, Y4
	VMULPS       Y2, Y4, Y4
	VFMADD231PS  Y9, Y4, Y0
	VMULPS       0(BX), Y5, Y5
	VFNMADD231PS Y5, Y3, Y0

	ADDQ $144, SI
	ADDQ $256, DI
	ADDQ $32, DX
	ADDQ $32, BX
	DECQ CX
	JNZ  q4k_sb

q4k_done:
	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	VMOVSS       X0, ret+40(FP)
	VZEROUPPER
	RET
