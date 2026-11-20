#include "textflag.h"

DATA q56kMask0F<>+0(SB)/8, $0x0f0f0f0f0f0f0f0f
DATA q56kMask0F<>+8(SB)/8, $0x0f0f0f0f0f0f0f0f
DATA q56kMask0F<>+16(SB)/8, $0x0f0f0f0f0f0f0f0f
DATA q56kMask0F<>+24(SB)/8, $0x0f0f0f0f0f0f0f0f
GLOBL q56kMask0F<>(SB), RODATA|NOPTR, $32

DATA q56kOnes16<>+0(SB)/8, $0x0001000100010001
DATA q56kOnes16<>+8(SB)/8, $0x0001000100010001
DATA q56kOnes16<>+16(SB)/8, $0x0001000100010001
DATA q56kOnes16<>+24(SB)/8, $0x0001000100010001
GLOBL q56kOnes16<>(SB), RODATA|NOPTR, $32

DATA q5kMask01<>+0(SB)/8, $0x0101010101010101
DATA q5kMask01<>+8(SB)/8, $0x0101010101010101
DATA q5kMask01<>+16(SB)/8, $0x0101010101010101
DATA q5kMask01<>+24(SB)/8, $0x0101010101010101
GLOBL q5kMask01<>(SB), RODATA|NOPTR, $32

DATA q5kMask02<>+0(SB)/8, $0x0202020202020202
DATA q5kMask02<>+8(SB)/8, $0x0202020202020202
DATA q5kMask02<>+16(SB)/8, $0x0202020202020202
DATA q5kMask02<>+24(SB)/8, $0x0202020202020202
GLOBL q5kMask02<>(SB), RODATA|NOPTR, $32

DATA q6kMask03<>+0(SB)/8, $0x0303030303030303
DATA q6kMask03<>+8(SB)/8, $0x0303030303030303
DATA q6kMask03<>+16(SB)/8, $0x0303030303030303
DATA q6kMask03<>+24(SB)/8, $0x0303030303030303
GLOBL q6kMask03<>(SB), RODATA|NOPTR, $32

DATA q6kBias32<>+0(SB)/8, $0x2020202020202020
DATA q6kBias32<>+8(SB)/8, $0x2020202020202020
DATA q6kBias32<>+16(SB)/8, $0x2020202020202020
DATA q6kBias32<>+24(SB)/8, $0x2020202020202020
GLOBL q6kBias32<>(SB), RODATA|NOPTR, $32

DATA q6kScalePerm<>+0(SB)/8, $0x0000000200000000
DATA q6kScalePerm<>+8(SB)/8, $0x0000000600000004
DATA q6kScalePerm<>+16(SB)/8, $0x0000000300000001
DATA q6kScalePerm<>+24(SB)/8, $0x0000000700000005
GLOBL q6kScalePerm<>(SB), RODATA|NOPTR, $32

// func dotQ5KAVX2(row *byte, aq *int8, aS *float32, asumf *float32, nSB int) float32
//
// One row of Q5_K weights (nSB super-blocks of 176 bytes) against one per-32 int8 activation row
// (q56k.go). dotQ4KAVX2 with a fifth code bit: per 64-element chunk j, bits 2j and 2j+1 of qh[l]
// become bit 4 of the low- and high-nibble codes (qh is shifted right by 2 after each chunk; VPSRLW
// leaks the upper byte's bits only into bits 2-7, above the two read). Codes are ≤ 31, so VPMADDUBSW
// pairs are ≤ 2·31·127 and do not saturate. The scale, minimum and f32 steps are dotQ4KAVX2's.
TEXT ·dotQ5KAVX2(SB), NOSPLIT, $0-44
	MOVQ row+0(FP), SI
	MOVQ aq+8(FP), DI
	MOVQ aS+16(FP), DX
	MOVQ asumf+24(FP), BX
	MOVQ nSB+32(FP), CX
	VXORPS  Y0, Y0, Y0
	VMOVDQU q56kMask0F<>(SB), Y15
	VMOVDQU q56kOnes16<>(SB), Y14
	TESTQ   CX, CX
	JLE     q5k_done

q5k_sb:
	VMOVDQU 16(SI), Y11 // qh

	// groups 0 (low nibbles, qh bit 0) and 1 (high nibbles, qh bit 1)
	VMOVDQU    48(SI), Y12
	VPSRLW     $4, Y12, Y13
	VPAND      Y15, Y12, Y12
	VPAND      Y15, Y13, Y13
	VPAND      q5kMask01<>(SB), Y11, Y1
	VPSLLW     $4, Y1, Y1
	VPOR       Y1, Y12, Y12
	VPAND      q5kMask02<>(SB), Y11, Y2
	VPSLLW     $3, Y2, Y2
	VPOR       Y2, Y13, Y13
	VPMADDUBSW 0(DI), Y12, Y1
	VPMADDWD   Y14, Y1, Y1
	VPMADDUBSW 32(DI), Y13, Y2
	VPMADDWD   Y14, Y2, Y2
	VPSRLW     $2, Y11, Y11

	// groups 2 (low nibbles, qh bit 2) and 3 (high nibbles, qh bit 3)
	VMOVDQU    80(SI), Y12
	VPSRLW     $4, Y12, Y13
	VPAND      Y15, Y12, Y12
	VPAND      Y15, Y13, Y13
	VPAND      q5kMask01<>(SB), Y11, Y3
	VPSLLW     $4, Y3, Y3
	VPOR       Y3, Y12, Y12
	VPAND      q5kMask02<>(SB), Y11, Y4
	VPSLLW     $3, Y4, Y4
	VPOR       Y4, Y13, Y13
	VPMADDUBSW 64(DI), Y12, Y3
	VPMADDWD   Y14, Y3, Y3
	VPMADDUBSW 96(DI), Y13, Y4
	VPMADDWD   Y14, Y4, Y4
	VPSRLW     $2, Y11, Y11

	// groups 4 (low nibbles, qh bit 4) and 5 (high nibbles, qh bit 5)
	VMOVDQU    112(SI), Y12
	VPSRLW     $4, Y12, Y13
	VPAND      Y15, Y12, Y12
	VPAND      Y15, Y13, Y13
	VPAND      q5kMask01<>(SB), Y11, Y5
	VPSLLW     $4, Y5, Y5
	VPOR       Y5, Y12, Y12
	VPAND      q5kMask02<>(SB), Y11, Y6
	VPSLLW     $3, Y6, Y6
	VPOR       Y6, Y13, Y13
	VPMADDUBSW 128(DI), Y12, Y5
	VPMADDWD   Y14, Y5, Y5
	VPMADDUBSW 160(DI), Y13, Y6
	VPMADDWD   Y14, Y6, Y6
	VPSRLW     $2, Y11, Y11

	// groups 6 (low nibbles, qh bit 6) and 7 (high nibbles, qh bit 7)
	VMOVDQU    144(SI), Y12
	VPSRLW     $4, Y12, Y13
	VPAND      Y15, Y12, Y12
	VPAND      Y15, Y13, Y13
	VPAND      q5kMask01<>(SB), Y11, Y7
	VPSLLW     $4, Y7, Y7
	VPOR       Y7, Y12, Y12
	VPAND      q5kMask02<>(SB), Y11, Y8
	VPSLLW     $3, Y8, Y8
	VPOR       Y8, Y13, Y13
	VPMADDUBSW 192(DI), Y12, Y7
	VPMADDWD   Y14, Y7, Y7
	VPMADDUBSW 224(DI), Y13, Y8
	VPMADDWD   Y14, Y8, Y8

	// Fold (as dotQ4KAVX2): after two VPHADDD levels, Y1 = [g0..g3 lanes 0-3 | g0..g3 lanes 4-7] and Y5
	// likewise for g4..g7.
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

	// 6-bit scales and mins, unpacked as dotQ4KAVX2 does
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
	// VMOVQ, not MOVQ: a legacy-SSE XMM write with dirty upper YMM halves stalls the loop (dotQ4KAVX2).
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

	ADDQ $176, SI
	ADDQ $256, DI
	ADDQ $32, DX
	ADDQ $32, BX
	DECQ CX
	JNZ  q5k_sb

q5k_done:
	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	VMOVSS       X0, ret+40(FP)
	VZEROUPPER
	RET

// func dotQ6KAVX2(row *byte, aq *int8, aS *float32, nSB int) float32
//
// One row of Q6_K weights (nSB super-blocks of 210 bytes) against one per-32 int8 activation row
// (q56k.go). Per 32-element group the unsigned code q+32 ∈ [0, 63] is built from ql's nibble and two
// qh bits; VPMADDUBSW(q+32, aq) − VPMADDUBSW(32, aq) is Σ(q−32)·aq per byte pair (each term ≤ 2·63·127,
// no saturation), and VPMADDWD by ones gives 8 int32 lanes: lanes 0-3 the first 16-element sub-block,
// 4-7 the second. dotQ4KAVX2's VPHADDD fold leaves Y1 = [g0a..g3a | g0b..g3b] and Y5 the same for
// g4..g7; VPMULLD by the int8 sub-scales arranged to match (VPMOVSXBD + VPERMD) and the half-swap add
// give each group's exact Σ_s sc_s·Σ(q−32)·aq; then acc += (d·aS) ⊙ that, eight groups per FMA.
TEXT ·dotQ6KAVX2(SB), NOSPLIT, $0-36
	MOVQ row+0(FP), SI
	MOVQ aq+8(FP), DI
	MOVQ aS+16(FP), DX
	MOVQ nSB+24(FP), CX
	VXORPS  Y0, Y0, Y0
	VMOVDQU q56kMask0F<>(SB), Y15
	VMOVDQU q56kOnes16<>(SB), Y14
	VMOVDQU q6kBias32<>(SB), Y13
	TESTQ   CX, CX
	JLE     q6k_done

q6k_sb:

	// half 0: groups 0..3; L0 = ql[0:32], L1 = ql[32:64], H = qh[0:32]
	VMOVDQU 0(SI), Y9
	VMOVDQU 32(SI), Y10
	VMOVDQU 128(SI), Y11
	// group 0: L0 low nibbles, H bits 0-1
	VPAND      Y15, Y9, Y12
	VPAND      q6kMask03<>(SB), Y11, Y1
	VPSLLW     $4, Y1, Y1
	VPOR       Y1, Y12, Y12
	VPMADDUBSW 0(DI), Y12, Y12
	VPMADDUBSW 0(DI), Y13, Y1
	VPSUBW     Y1, Y12, Y12
	VPMADDWD   Y14, Y12, Y1
	// group 1: L1 low nibbles, H bits 2-3
	VPAND      Y15, Y10, Y12
	VPSRLW     $2, Y11, Y2
	VPAND      q6kMask03<>(SB), Y2, Y2
	VPSLLW     $4, Y2, Y2
	VPOR       Y2, Y12, Y12
	VPMADDUBSW 32(DI), Y12, Y12
	VPMADDUBSW 32(DI), Y13, Y2
	VPSUBW     Y2, Y12, Y12
	VPMADDWD   Y14, Y12, Y2
	// group 2: L0 high nibbles, H bits 4-5
	VPSRLW     $4, Y9, Y12
	VPAND      Y15, Y12, Y12
	VPSRLW     $4, Y11, Y3
	VPAND      q6kMask03<>(SB), Y3, Y3
	VPSLLW     $4, Y3, Y3
	VPOR       Y3, Y12, Y12
	VPMADDUBSW 64(DI), Y12, Y12
	VPMADDUBSW 64(DI), Y13, Y3
	VPSUBW     Y3, Y12, Y12
	VPMADDWD   Y14, Y12, Y3
	// group 3: L1 high nibbles, H bits 6-7
	VPSRLW     $4, Y10, Y12
	VPAND      Y15, Y12, Y12
	VPSRLW     $6, Y11, Y4
	VPAND      q6kMask03<>(SB), Y4, Y4
	VPSLLW     $4, Y4, Y4
	VPOR       Y4, Y12, Y12
	VPMADDUBSW 96(DI), Y12, Y12
	VPMADDUBSW 96(DI), Y13, Y4
	VPSUBW     Y4, Y12, Y12
	VPMADDWD   Y14, Y12, Y4

	// half 1: groups 4..7; L0 = ql[64:96], L1 = ql[96:128], H = qh[32:64]
	VMOVDQU 64(SI), Y9
	VMOVDQU 96(SI), Y10
	VMOVDQU 160(SI), Y11
	// group 4: L0 low nibbles, H bits 0-1
	VPAND      Y15, Y9, Y12
	VPAND      q6kMask03<>(SB), Y11, Y5
	VPSLLW     $4, Y5, Y5
	VPOR       Y5, Y12, Y12
	VPMADDUBSW 128(DI), Y12, Y12
	VPMADDUBSW 128(DI), Y13, Y5
	VPSUBW     Y5, Y12, Y12
	VPMADDWD   Y14, Y12, Y5
	// group 5: L1 low nibbles, H bits 2-3
	VPAND      Y15, Y10, Y12
	VPSRLW     $2, Y11, Y6
	VPAND      q6kMask03<>(SB), Y6, Y6
	VPSLLW     $4, Y6, Y6
	VPOR       Y6, Y12, Y12
	VPMADDUBSW 160(DI), Y12, Y12
	VPMADDUBSW 160(DI), Y13, Y6
	VPSUBW     Y6, Y12, Y12
	VPMADDWD   Y14, Y12, Y6
	// group 6: L0 high nibbles, H bits 4-5
	VPSRLW     $4, Y9, Y12
	VPAND      Y15, Y12, Y12
	VPSRLW     $4, Y11, Y7
	VPAND      q6kMask03<>(SB), Y7, Y7
	VPSLLW     $4, Y7, Y7
	VPOR       Y7, Y12, Y12
	VPMADDUBSW 192(DI), Y12, Y12
	VPMADDUBSW 192(DI), Y13, Y7
	VPSUBW     Y7, Y12, Y12
	VPMADDWD   Y14, Y12, Y7
	// group 7: L1 high nibbles, H bits 6-7
	VPSRLW     $4, Y10, Y12
	VPAND      Y15, Y12, Y12
	VPSRLW     $6, Y11, Y8
	VPAND      q6kMask03<>(SB), Y8, Y8
	VPSLLW     $4, Y8, Y8
	VPOR       Y8, Y12, Y12
	VPMADDUBSW 224(DI), Y12, Y12
	VPMADDUBSW 224(DI), Y13, Y8
	VPSUBW     Y8, Y12, Y12
	VPMADDWD   Y14, Y12, Y8

	// Fold (as dotQ4KAVX2): after two VPHADDD levels, Y1 = [g0..g3 lanes 0-3 | g0..g3 lanes 4-7] and Y5
	// likewise for g4..g7.
	VPHADDD    Y2, Y1, Y1
	VPHADDD    Y4, Y3, Y3
	VPHADDD    Y3, Y1, Y1
	VPHADDD    Y6, Y5, Y5
	VPHADDD    Y8, Y7, Y7
	VPHADDD    Y7, Y5, Y5

	// sub-scales: Y9 = [s0 s2 s4 s6 | s1 s3 s5 s7], Y10 = [s8 s10 s12 s14 | s9 s11 s13 s15]
	VMOVDQU   q6kScalePerm<>(SB), Y11
	VPMOVSXBD 192(SI), Y9
	VPERMD    Y9, Y11, Y9
	VPMOVSXBD 200(SI), Y10
	VPERMD    Y10, Y11, Y10
	VPMULLD   Y9, Y1, Y1
	VPMULLD   Y10, Y5, Y5
	VPERM2I128 $0x20, Y5, Y1, Y9
	VPERM2I128 $0x31, Y5, Y1, Y10
	VPADDD     Y10, Y9, Y9
	VCVTDQ2PS  Y9, Y9

	// d is the block's last two bytes: a 16-bit load, so the last block of a row is not over-read.
	MOVWLZX      208(SI), AX
	VMOVD        AX, X1
	VCVTPH2PS    X1, X1
	VBROADCASTSS X1, Y2

	// acc += (d·aS) ⊙ isum
	VMULPS      0(DX), Y2, Y2
	VFMADD231PS Y9, Y2, Y0

	ADDQ $210, SI
	ADDQ $256, DI
	ADDQ $32, DX
	DECQ CX
	JNZ  q6k_sb

q6k_done:
	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	VMOVSS       X0, ret+32(FP)
	VZEROUPPER
	RET
