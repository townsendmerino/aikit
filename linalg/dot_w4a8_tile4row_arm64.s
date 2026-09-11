//go:build arm64

#include "textflag.h"

// func dotW4A8Tile4RowSDOT(act *int8, actStride int, packed *byte, scales *float32, dst *float32, nGroups int)
//
// Four W4A8 outputs in one call — four activation rows, actStride bytes apart,
// against ONE weight row — written to dst[0..3] BEFORE the activation scale,
// exactly as dotW4A8FoldSDOT returns it. The arm64 twin of
// dotW4A8Tile4RowAVX2 (audit M-02, docs/task-simd-audit.md S-01's arm64 half on
// the CANONICAL layout, as opposed to the row4 tile that needs a repack).
//
// THE WIN IS THE UNPACK, not the loads. dotW4A8FoldSDOT unpacks a weight group
// — AND / USHR / ZIP1 / ZIP2 / SUB / SUB — on every call, so four activation
// rows against one weight row unpacked it four times. Here the group is
// unpacked ONCE into V3/V4 and all four rows' SDOT pairs run against it, and the
// group's f32 scale is broadcast once into V19 rather than four times.
//
// BIT-IDENTICAL TO dotW4A8FoldSDOT BY CONSTRUCTION, which is the whole design
// constraint: each row's arithmetic is the identical instruction sequence in the
// identical order — zero a fresh int32 accumulator per group, the two SDOTs low
// half then high half, SCVTF, then FMLA into that row's own f32 accumulator with
// the group scale, and two FADDPs at the end. Nothing is shared between rows
// except the unpacked weights and the scale broadcast, neither of which enters a
// reduction. That matters beyond tidiness: TestMatmulBTW4A8_MConsistent forbids
// the result depending on M, and goinfer's speculative verify relies on it.
//
// Registers: V3/V4 unpacked weights, V19 group scale, V20-V23 the four f32
// accumulators, V6/V7 + V16 + V18 reused per row, V30/V31 the 0x0F and 8
// constants. SDOT, SCVTF and FADDP have no Go mnemonics, so they are raw WORDs
// with the encoding arithmetic shown — the same ones dot_w4a8_arm64.s uses.
TEXT ·dotW4A8Tile4RowSDOT(SB), NOSPLIT, $0-48
	MOVD act+0(FP), R0
	MOVD actStride+8(FP), R1
	MOVD packed+16(FP), R2
	MOVD scales+24(FP), R3
	MOVD dst+32(FP), R4
	MOVD nGroups+40(FP), R5

	// Four activation cursors, actStride bytes apart, each advanced 32 bytes
	// per group by its own post-indexed load.
	MOVD R0, R6
	ADD  R1, R6, R7
	ADD  R1, R7, R8
	ADD  R1, R8, R9

	VMOVI $0x0F, V30.B16
	VMOVI $8, V31.B16
	VEOR  V20.B16, V20.B16, V20.B16
	VEOR  V21.B16, V21.B16, V21.B16
	VEOR  V22.B16, V22.B16, V22.B16
	VEOR  V23.B16, V23.B16, V23.B16

	CBZ R5, tile4store

tile4loop:
	// Unpack one weight group ONCE — shared by all four rows.
	VLD1.P 16(R2), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VZIP1  V2.B16, V1.B16, V3.B16
	VZIP2  V2.B16, V1.B16, V4.B16
	VSUB   V31.B16, V3.B16, V3.B16
	VSUB   V31.B16, V4.B16, V4.B16
	VLD1R  (R3), [V19.S4]            // broadcast scale[g], once for four rows
	ADD    $4, R3, R3

	// Row 0.
	VLD1.P 32(R6), [V6.B16, V7.B16]
	VMOVI  $0, V16.B16
	WORD   $0x4E8394D0               // SDOT V16.4S, V6.16B, V3.16B
	WORD   $0x4E8494F0               // SDOT V16.4S, V7.16B, V4.16B
	WORD   $0x4E21DA12               // SCVTF V18.4S, V16.4S
	VFMLA  V19.S4, V18.S4, V20.S4

	// Row 1.
	VLD1.P 32(R7), [V6.B16, V7.B16]
	VMOVI  $0, V16.B16
	WORD   $0x4E8394D0
	WORD   $0x4E8494F0
	WORD   $0x4E21DA12
	VFMLA  V19.S4, V18.S4, V21.S4

	// Row 2.
	VLD1.P 32(R8), [V6.B16, V7.B16]
	VMOVI  $0, V16.B16
	WORD   $0x4E8394D0
	WORD   $0x4E8494F0
	WORD   $0x4E21DA12
	VFMLA  V19.S4, V18.S4, V22.S4

	// Row 3.
	VLD1.P 32(R9), [V6.B16, V7.B16]
	VMOVI  $0, V16.B16
	WORD   $0x4E8394D0
	WORD   $0x4E8494F0
	WORD   $0x4E21DA12
	VFMLA  V19.S4, V18.S4, V23.S4

	SUBS $1, R5, R5
	BNE  tile4loop

tile4store:
	// Same two pairwise f32 reduces per row that dotW4A8FoldSDOT ends with.
	// FADDP Vd.4S,Vn.4S,Vm.4S = 0x6E20D400 | (Rm<<16) | (Rn<<5) | Rd.
	WORD  $0x6E34D694                // FADDP V20.4S, V20.4S, V20.4S
	WORD  $0x6E34D694
	FMOVS F20, (R4)
	WORD  $0x6E35D6B5                // FADDP V21.4S, V21.4S, V21.4S
	WORD  $0x6E35D6B5
	FMOVS F21, 4(R4)
	WORD  $0x6E36D6D6                // FADDP V22.4S, V22.4S, V22.4S
	WORD  $0x6E36D6D6
	FMOVS F22, 8(R4)
	WORD  $0x6E37D6F7                // FADDP V23.4S, V23.4S, V23.4S
	WORD  $0x6E37D6F7
	FMOVS F23, 12(R4)
	RET
