//go:build arm64

#include "textflag.h"

// S-05 (docs/task-simd-audit.md): the -8 centering folded into the SDOT
// accumulator's INITIAL VALUE instead of being subtracted from every nibble.
//
// dotW4A8SplitHalf4Row (dot_w4a8_arm64.s) computes, per row per 32-k group and
// per SDOT lane l:
//
//	acc[l] = Σ_{i<4} (nibLo[4l+i] - 8)·act[4l+i] + Σ_{i<4} (nibHi[4l+i] - 8)·act[16+4l+i]
//
// with two VSUB.16B per row to form (nib - 8). In exact integer arithmetic
//
//	Σ (nib - 8)·act  =  Σ nib·act  -  8·Σ act
//
// and the right-hand term depends only on the ACTIVATION — the same for every
// weight row of the matmul — so it is computed once per token per group by
// w4a8LaneCorrNeg8 below (corr[g][l] = -8·Σ act over lane l's eight k's) and
// loaded as the accumulator's starting value. Both VSUBs vanish and MOVI #0
// becomes a register copy: 9 → 7 SIMD µops per row-group. The int32 that
// reaches SCVTF is the same integer (|Σ| < 2^20, no overflow), so the f32 fold,
// the FADDP tree and every bit of the output are unchanged — held by
// TestDotW4A8SplitHalf4RowFold_bitIdenticalToSplitHalf4Row with exact ==.
//
// Every SDOT/SCVTF/FADDP WORD below was computed from the encoding formula
// (0x4E809400 | Rm<<16 | Rn<<5 | Rd; 0x4E21D800 | Rn<<5 | Rd), self-checked
// against the four words dot_w4a8_arm64.s already carries, and cross-checked
// word for word against clang's assembler (14/14 equal, 2026-09-22).

// func w4a8LaneCorrNeg8(act *int8, corr *int32, nGroups int)
//
// corr[4g+l] = -8 · ( Σ_{i<4} act[32g+4l+i] + Σ_{i<4} act[32g+16+4l+i] ), the
// exact lane mapping of the two SDOTs in the row kernel (low half V6 against
// lanes 4l..4l+3, high half V7 against 16+4l..16+4l+3). Two SDOTs against a
// vector of int8 -8 per group; K/32 groups; written once per activation row and
// shared by every quad of the matmul.
TEXT ·w4a8LaneCorrNeg8(SB), NOSPLIT, $0-24
	MOVD act+0(FP), R0
	MOVD corr+8(FP), R1
	MOVD nGroups+16(FP), R3

	VMOVI $0xF8, V31.B16 // int8 -8 in every lane (SDOT is signed × signed)

corrloop:
	VLD1.P 32(R0), [V6.B16, V7.B16]
	VMOVI  $0, V16.B16
	WORD   $0x4E9F94D0 // SDOT V16.4S, V6.16B, V31.16B
	WORD   $0x4E9F94F0 // SDOT V16.4S, V7.16B, V31.16B
	VST1.P [V16.S4], 16(R1)
	SUBS   $1, R3, R3
	BNE    corrloop
	RET

// func dotW4A8SplitHalf4RowFold(act *int8, corr *int32, packed4 *byte, scales4 *float32, dst *float32, nGroups int)
//
// dotW4A8SplitHalf4Row with the centering folded: same packed4/scales4 layout
// (repackSplitHalf4RowBlock / interleaveScales4Row), same dst (4 float32s, one
// per row), same nGroups contract (any count >= 1). corr is w4a8LaneCorrNeg8's
// output for THIS act — 4 int32 per group, nGroups groups. Per group the
// activation chunk (32 bytes) and the correction (16 bytes) are loaded once
// and shared by the four rows.
TEXT ·dotW4A8SplitHalf4RowFold(SB), NOSPLIT, $0-48
	MOVD act+0(FP), R0
	MOVD corr+8(FP), R5
	MOVD packed4+16(FP), R1
	MOVD scales4+24(FP), R2
	MOVD dst+32(FP), R4
	MOVD nGroups+40(FP), R3

	VMOVI $0x0F, V30.B16
	VEOR  V20.B16, V20.B16, V20.B16 // row0 acc
	VEOR  V21.B16, V21.B16, V21.B16 // row1 acc
	VEOR  V27.B16, V27.B16, V27.B16 // row2 acc
	VEOR  V10.B16, V10.B16, V10.B16 // row3 acc

foldloop:
	VLD1.P 32(R0), [V6.B16, V7.B16] // activation for this group — loaded ONCE
	VLD1.P 16(R5), [V5.S4]          // corr_g — the four rows' SDOT starting value

	// row0 → V20
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16 // low nibbles, uncentered (k 0..15)
	VUSHR  $4, V0.B16, V2.B16      // high nibbles, uncentered (k 16..31)
	VORR   V5.B16, V5.B16, V16.B16 // acc := corr_g (was MOVI #0 after two VSUBs)
	WORD   $0x4E8194D0 // SDOT V16.4S, V6.16B, V1.16B
	WORD   $0x4E8294F0 // SDOT V16.4S, V7.16B, V2.16B
	WORD   $0x4E21DA12 // SCVTF V18.4S, V16.4S
	VLD1R  (R2), [V19.S4]
	VFMLA  V19.S4, V18.S4, V20.S4
	ADD    $4, R2, R2

	// row1 → V21
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VORR   V5.B16, V5.B16, V17.B16
	WORD   $0x4E8194D1 // SDOT V17.4S, V6.16B, V1.16B
	WORD   $0x4E8294F1 // SDOT V17.4S, V7.16B, V2.16B
	WORD   $0x4E21DA36 // SCVTF V22.4S, V17.4S
	VLD1R  (R2), [V23.S4]
	VFMLA  V23.S4, V22.S4, V21.S4
	ADD    $4, R2, R2

	// row2 → V27
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VORR   V5.B16, V5.B16, V24.B16
	WORD   $0x4E8194D8 // SDOT V24.4S, V6.16B, V1.16B
	WORD   $0x4E8294F8 // SDOT V24.4S, V7.16B, V2.16B
	WORD   $0x4E21DB19 // SCVTF V25.4S, V24.4S
	VLD1R  (R2), [V26.S4]
	VFMLA  V26.S4, V25.S4, V27.S4
	ADD    $4, R2, R2

	// row3 → V10
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VORR   V5.B16, V5.B16, V28.B16
	WORD   $0x4E8194DC // SDOT V28.4S, V6.16B, V1.16B
	WORD   $0x4E8294FC // SDOT V28.4S, V7.16B, V2.16B
	WORD   $0x4E21DB88 // SCVTF V8.4S, V28.4S
	VLD1R  (R2), [V9.S4]
	VFMLA  V9.S4, V8.S4, V10.S4
	ADD    $4, R2, R2

	SUBS $1, R3, R3
	BNE  foldloop

	WORD  $0x6E34D694 // FADDP V20 → lane0 = row0 sum
	WORD  $0x6E34D694
	FMOVS F20, (R4)
	WORD  $0x6E35D6B5 // FADDP V21 → lane0 = row1 sum
	WORD  $0x6E35D6B5
	FMOVS F21, 4(R4)
	WORD  $0x6E3BD77B // FADDP V27 → lane0 = row2 sum
	WORD  $0x6E3BD77B
	FMOVS F27, 8(R4)
	WORD  $0x6E2AD54A // FADDP V10 → lane0 = row3 sum
	WORD  $0x6E2AD54A
	FMOVS F10, 12(R4)
	RET
