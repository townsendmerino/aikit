//go:build arm64

#include "textflag.h"

// func dotW4A8SplitHalf4RowFoldF16(act *int8, corr *int32, packed4 *byte, scales4 *uint16, dst *float32, nGroups int)
//
// dotW4A8SplitHalf4RowFold (dot_w4a8_fold_arm64.s) over binary16 scales, widened in-kernel: the L1 design's
// "NEON FCVTL twin" of the amd64 fused kernel (goinfer docs/tasks/task-cpu-decode-peer-gap-2026-09.md). Same
// act / corr / packed4 / dst / nGroups contract; scales4 is the same interleaved layout (group g's four row
// scales at [4g, 4g+4)) holding binary16 bit patterns.
//
// Per group, the f32 kernel does one VLD1R broadcast per row (four 4-byte loads). This one does a single
// 8-byte load and one FCVTL for the quad's four scales, then FMLA by element (Vd += Vn · V29.S[r]) in place
// of the broadcast-then-FMLA. Each lane still computes one fused multiply-add of the same SCVTF output with
// the same f32 scale, since FCVTL widens exactly, so the output equals dotW4A8SplitHalf4RowFold on the
// widened scales bit for bit (TestDotW4A8SplitHalf4RowFoldF16_matchesF32).
//
// New WORDs, cross-checked against clang's assembler (2026-09-28):
//   FCVTL V29.4S, V29.4H          = 0x0E217BBD
//   FMLA  V20.4S, V18.4S, V29.S[0] = 0x4F9D1254
//   FMLA  V21.4S, V22.4S, V29.S[1] = 0x4FBD12D5
//   FMLA  V27.4S, V25.4S, V29.S[2] = 0x4F9D1B3B
//   FMLA  V10.4S, V8.4S,  V29.S[3] = 0x4FBD190A
// Every other instruction is dotW4A8SplitHalf4RowFold's own.
TEXT ·dotW4A8SplitHalf4RowFoldF16(SB), NOSPLIT, $0-48
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

foldf16loop:
	VLD1.P 32(R0), [V6.B16, V7.B16] // activation for this group — loaded ONCE
	VLD1.P 16(R5), [V5.S4]          // corr_g — the four rows' SDOT starting value
	VLD1.P 8(R2), [V29.H4]          // the quad's four binary16 scales for this group
	WORD   $0x0E217BBD              // FCVTL V29.4S, V29.4H — widened exactly

	// row0 → V20
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16 // low nibbles, uncentered (k 0..15)
	VUSHR  $4, V0.B16, V2.B16      // high nibbles, uncentered (k 16..31)
	VORR   V5.B16, V5.B16, V16.B16 // acc := corr_g
	WORD   $0x4E8194D0 // SDOT V16.4S, V6.16B, V1.16B
	WORD   $0x4E8294F0 // SDOT V16.4S, V7.16B, V2.16B
	WORD   $0x4E21DA12 // SCVTF V18.4S, V16.4S
	WORD   $0x4F9D1254 // FMLA V20.4S, V18.4S, V29.S[0]

	// row1 → V21
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VORR   V5.B16, V5.B16, V17.B16
	WORD   $0x4E8194D1 // SDOT V17.4S, V6.16B, V1.16B
	WORD   $0x4E8294F1 // SDOT V17.4S, V7.16B, V2.16B
	WORD   $0x4E21DA36 // SCVTF V22.4S, V17.4S
	WORD   $0x4FBD12D5 // FMLA V21.4S, V22.4S, V29.S[1]

	// row2 → V27
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VORR   V5.B16, V5.B16, V24.B16
	WORD   $0x4E8194D8 // SDOT V24.4S, V6.16B, V1.16B
	WORD   $0x4E8294F8 // SDOT V24.4S, V7.16B, V2.16B
	WORD   $0x4E21DB19 // SCVTF V25.4S, V24.4S
	WORD   $0x4F9D1B3B // FMLA V27.4S, V25.4S, V29.S[2]

	// row3 → V10
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VORR   V5.B16, V5.B16, V28.B16
	WORD   $0x4E8194DC // SDOT V28.4S, V6.16B, V1.16B
	WORD   $0x4E8294FC // SDOT V28.4S, V7.16B, V2.16B
	WORD   $0x4E21DB88 // SCVTF V8.4S, V28.4S
	WORD   $0x4FBD190A // FMLA V10.4S, V8.4S, V29.S[3]

	SUBS $1, R3, R3
	BNE  foldf16loop

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
