// docs/task-simd-audit.md S-01.
//
// Hand-written. Every raw WORD below is verifiable by hand from the field
// decomposition, which is the point of writing them out here rather than citing a
// generator that is not in this tree:
//
//   SDOT  Vd.4S, Vn.16B, Vm.16B = 0x4E809400 | (m<<16) | (n<<5) | d
//   SCVTF Vd.4S, Vn.4S          = 0x4E21D800 | (n<<5) | d
//   FADDP Vd.4S, Vn.4S, Vm.4S   = 0x6E20D400 | (m<<16) | (n<<5) | d
//
// Each was cross-checked against an encoding already in the tree before use —
// dot_i8dp_arm64.s documents the same SDOT formula and carries 0x4E819404, and
// dot_w4a8_arm64.s carries 0x4E8394D0 (SDOT V16,V6,V3), 0x4E21DA12 (SCVTF V18,V16)
// and 0x6E34D694 (FADDP V20) — so the formula is anchored to instructions this
// package already executes, not merely to a manual.

#include "textflag.h"

// func dotW4A8Row4Tile4x4(act *int8, actStride int, packed4 *byte, scales4 *float32, dst *float32, nGroups int)
//
// task-simd-audit.md S-01: the register-blocked W4A8 tile. Four ACTIVATION rows
// against the four WEIGHT rows of one row4 quad, 16 outputs per call, in one pass
// over the weights.
//
// The point is what does NOT happen four times. dotW4A8SplitHalf4Row already shares
// one activation load across a quad's four weight rows, but at M>1 the caller invokes
// it once per activation row, so the 16-byte weight load, the {AND, USHR, SUB, SUB}
// unpack and the scale broadcast are all repeated per activation row. Here each weight
// row is loaded and unpacked ONCE per group and then consumed by all four activation
// rows: 96 SIMD uops per group for 512 MACs (0.19/MAC) against 0.34 for the canonical
// per-(row,column) GEMV, and issue-bound rather than latency-bound because 16 live
// accumulators leave no dependency stall to hide.
//
// BIT-IDENTICAL BY CONSTRUCTION, and deliberately so: for every one of the 16 outputs
// the per-group sequence below is instruction-for-instruction the sequence
// dotW4A8SplitHalf4Row runs -- VMOVI zero, SDOT lo, SDOT hi, SCVTF, VFMLA by the
// broadcast group scale into a persistent 4-lane f32 accumulator, in ascending group --
// and the epilogue is the same FADDP-FADDP tree. Nothing is reassociated; the only
// change is which loads and unpacks are shared. TestMatmulBTW4A8_MConsistent and
// TestWeightMatW4A8_MConsistentAcrossRow4Dispatch are the gates.
//
// Layout contract: packed4/scales4 are repackSplitHalf4RowBlock / interleaveScales4Row
// interleaved (per group, four rows' 16 split-half bytes back to back, then four f32
// scales). act points at the first of four activation rows actStride BYTES apart. dst
// receives 16 float32 row-major: dst[m*4+r] is activation row m against weight row r.
//
// Register budget is exactly 32 and that is why the loop is shaped m-inner:
//   V0-V7   four activation rows, lo/hi halves, loaded once per group
//   V8-V23  the 16 f32 accumulators, live for the whole call
//   V24/V25 the current weight row unpacked      V26/V27 SDOT int32 / its SCVTF
//   V28     the current weight row's group scale V29 the raw 16 packed bytes
//   V30/V31 the 0x0F mask and the 8 the nibbles are centred by
TEXT ·dotW4A8Row4Tile4x4(SB), NOSPLIT, $0-48
	MOVD act+0(FP), R0
	MOVD actStride+8(FP), R8
	MOVD packed4+16(FP), R1
	MOVD scales4+24(FP), R2
	MOVD dst+32(FP), R4
	MOVD nGroups+40(FP), R3

	// The other three activation rows, actStride bytes apart. Each of the four
	// advances by 32 per group independently (VLD1.P), so no re-derivation.
	ADD R0, R8, R5
	ADD R5, R8, R6
	ADD R6, R8, R7

	VMOVI $0x0F, V30.B16
	VMOVI $8, V31.B16
	VEOR V8.B16, V8.B16, V8.B16 // acc[act 0][wrow 0]
	VEOR V9.B16, V9.B16, V9.B16 // acc[act 0][wrow 1]
	VEOR V10.B16, V10.B16, V10.B16 // acc[act 0][wrow 2]
	VEOR V11.B16, V11.B16, V11.B16 // acc[act 0][wrow 3]
	VEOR V12.B16, V12.B16, V12.B16 // acc[act 1][wrow 0]
	VEOR V13.B16, V13.B16, V13.B16 // acc[act 1][wrow 1]
	VEOR V14.B16, V14.B16, V14.B16 // acc[act 1][wrow 2]
	VEOR V15.B16, V15.B16, V15.B16 // acc[act 1][wrow 3]
	VEOR V16.B16, V16.B16, V16.B16 // acc[act 2][wrow 0]
	VEOR V17.B16, V17.B16, V17.B16 // acc[act 2][wrow 1]
	VEOR V18.B16, V18.B16, V18.B16 // acc[act 2][wrow 2]
	VEOR V19.B16, V19.B16, V19.B16 // acc[act 2][wrow 3]
	VEOR V20.B16, V20.B16, V20.B16 // acc[act 3][wrow 0]
	VEOR V21.B16, V21.B16, V21.B16 // acc[act 3][wrow 1]
	VEOR V22.B16, V22.B16, V22.B16 // acc[act 3][wrow 2]
	VEOR V23.B16, V23.B16, V23.B16 // acc[act 3][wrow 3]

tile4x4loop:
	// One 32-wide K group. Four activation chunks, then four weight rows over them.
	VLD1.P 32(R0), [V0.B16, V1.B16] // activation row 0
	VLD1.P 32(R5), [V2.B16, V3.B16] // activation row 1
	VLD1.P 32(R6), [V4.B16, V5.B16] // activation row 2
	VLD1.P 32(R7), [V6.B16, V7.B16] // activation row 3

	// ---- weight row 0: loaded and unpacked once, used by all four activation rows ----
	VLD1.P 16(R1), [V29.B16]
	VAND   V30.B16, V29.B16, V24.B16
	VUSHR  $4, V29.B16, V25.B16
	VSUB   V31.B16, V24.B16, V24.B16
	VSUB   V31.B16, V25.B16, V25.B16
	VLD1R  (R2), [V28.S4]
	ADD    $4, R2, R2
	VMOVI $0, V26.B16
	WORD  $0x4E98941A // SDOT V26.4S, V0.16B, V24.16B
	WORD  $0x4E99943A // SDOT V26.4S, V1.16B, V25.16B
	WORD  $0x4E21DB5B // SCVTF V27.4S, V26.4S
	VFMLA V28.S4, V27.S4, V8.S4 // acc[act 0][wrow 0]
	VMOVI $0, V26.B16
	WORD  $0x4E98945A // SDOT V26.4S, V2.16B, V24.16B
	WORD  $0x4E99947A // SDOT V26.4S, V3.16B, V25.16B
	WORD  $0x4E21DB5B // SCVTF V27.4S, V26.4S
	VFMLA V28.S4, V27.S4, V12.S4 // acc[act 1][wrow 0]
	VMOVI $0, V26.B16
	WORD  $0x4E98949A // SDOT V26.4S, V4.16B, V24.16B
	WORD  $0x4E9994BA // SDOT V26.4S, V5.16B, V25.16B
	WORD  $0x4E21DB5B // SCVTF V27.4S, V26.4S
	VFMLA V28.S4, V27.S4, V16.S4 // acc[act 2][wrow 0]
	VMOVI $0, V26.B16
	WORD  $0x4E9894DA // SDOT V26.4S, V6.16B, V24.16B
	WORD  $0x4E9994FA // SDOT V26.4S, V7.16B, V25.16B
	WORD  $0x4E21DB5B // SCVTF V27.4S, V26.4S
	VFMLA V28.S4, V27.S4, V20.S4 // acc[act 3][wrow 0]

	// ---- weight row 1: loaded and unpacked once, used by all four activation rows ----
	VLD1.P 16(R1), [V29.B16]
	VAND   V30.B16, V29.B16, V24.B16
	VUSHR  $4, V29.B16, V25.B16
	VSUB   V31.B16, V24.B16, V24.B16
	VSUB   V31.B16, V25.B16, V25.B16
	VLD1R  (R2), [V28.S4]
	ADD    $4, R2, R2
	VMOVI $0, V26.B16
	WORD  $0x4E98941A // SDOT V26.4S, V0.16B, V24.16B
	WORD  $0x4E99943A // SDOT V26.4S, V1.16B, V25.16B
	WORD  $0x4E21DB5B // SCVTF V27.4S, V26.4S
	VFMLA V28.S4, V27.S4, V9.S4 // acc[act 0][wrow 1]
	VMOVI $0, V26.B16
	WORD  $0x4E98945A // SDOT V26.4S, V2.16B, V24.16B
	WORD  $0x4E99947A // SDOT V26.4S, V3.16B, V25.16B
	WORD  $0x4E21DB5B // SCVTF V27.4S, V26.4S
	VFMLA V28.S4, V27.S4, V13.S4 // acc[act 1][wrow 1]
	VMOVI $0, V26.B16
	WORD  $0x4E98949A // SDOT V26.4S, V4.16B, V24.16B
	WORD  $0x4E9994BA // SDOT V26.4S, V5.16B, V25.16B
	WORD  $0x4E21DB5B // SCVTF V27.4S, V26.4S
	VFMLA V28.S4, V27.S4, V17.S4 // acc[act 2][wrow 1]
	VMOVI $0, V26.B16
	WORD  $0x4E9894DA // SDOT V26.4S, V6.16B, V24.16B
	WORD  $0x4E9994FA // SDOT V26.4S, V7.16B, V25.16B
	WORD  $0x4E21DB5B // SCVTF V27.4S, V26.4S
	VFMLA V28.S4, V27.S4, V21.S4 // acc[act 3][wrow 1]

	// ---- weight row 2: loaded and unpacked once, used by all four activation rows ----
	VLD1.P 16(R1), [V29.B16]
	VAND   V30.B16, V29.B16, V24.B16
	VUSHR  $4, V29.B16, V25.B16
	VSUB   V31.B16, V24.B16, V24.B16
	VSUB   V31.B16, V25.B16, V25.B16
	VLD1R  (R2), [V28.S4]
	ADD    $4, R2, R2
	VMOVI $0, V26.B16
	WORD  $0x4E98941A // SDOT V26.4S, V0.16B, V24.16B
	WORD  $0x4E99943A // SDOT V26.4S, V1.16B, V25.16B
	WORD  $0x4E21DB5B // SCVTF V27.4S, V26.4S
	VFMLA V28.S4, V27.S4, V10.S4 // acc[act 0][wrow 2]
	VMOVI $0, V26.B16
	WORD  $0x4E98945A // SDOT V26.4S, V2.16B, V24.16B
	WORD  $0x4E99947A // SDOT V26.4S, V3.16B, V25.16B
	WORD  $0x4E21DB5B // SCVTF V27.4S, V26.4S
	VFMLA V28.S4, V27.S4, V14.S4 // acc[act 1][wrow 2]
	VMOVI $0, V26.B16
	WORD  $0x4E98949A // SDOT V26.4S, V4.16B, V24.16B
	WORD  $0x4E9994BA // SDOT V26.4S, V5.16B, V25.16B
	WORD  $0x4E21DB5B // SCVTF V27.4S, V26.4S
	VFMLA V28.S4, V27.S4, V18.S4 // acc[act 2][wrow 2]
	VMOVI $0, V26.B16
	WORD  $0x4E9894DA // SDOT V26.4S, V6.16B, V24.16B
	WORD  $0x4E9994FA // SDOT V26.4S, V7.16B, V25.16B
	WORD  $0x4E21DB5B // SCVTF V27.4S, V26.4S
	VFMLA V28.S4, V27.S4, V22.S4 // acc[act 3][wrow 2]

	// ---- weight row 3: loaded and unpacked once, used by all four activation rows ----
	VLD1.P 16(R1), [V29.B16]
	VAND   V30.B16, V29.B16, V24.B16
	VUSHR  $4, V29.B16, V25.B16
	VSUB   V31.B16, V24.B16, V24.B16
	VSUB   V31.B16, V25.B16, V25.B16
	VLD1R  (R2), [V28.S4]
	ADD    $4, R2, R2
	VMOVI $0, V26.B16
	WORD  $0x4E98941A // SDOT V26.4S, V0.16B, V24.16B
	WORD  $0x4E99943A // SDOT V26.4S, V1.16B, V25.16B
	WORD  $0x4E21DB5B // SCVTF V27.4S, V26.4S
	VFMLA V28.S4, V27.S4, V11.S4 // acc[act 0][wrow 3]
	VMOVI $0, V26.B16
	WORD  $0x4E98945A // SDOT V26.4S, V2.16B, V24.16B
	WORD  $0x4E99947A // SDOT V26.4S, V3.16B, V25.16B
	WORD  $0x4E21DB5B // SCVTF V27.4S, V26.4S
	VFMLA V28.S4, V27.S4, V15.S4 // acc[act 1][wrow 3]
	VMOVI $0, V26.B16
	WORD  $0x4E98949A // SDOT V26.4S, V4.16B, V24.16B
	WORD  $0x4E9994BA // SDOT V26.4S, V5.16B, V25.16B
	WORD  $0x4E21DB5B // SCVTF V27.4S, V26.4S
	VFMLA V28.S4, V27.S4, V19.S4 // acc[act 2][wrow 3]
	VMOVI $0, V26.B16
	WORD  $0x4E9894DA // SDOT V26.4S, V6.16B, V24.16B
	WORD  $0x4E9994FA // SDOT V26.4S, V7.16B, V25.16B
	WORD  $0x4E21DB5B // SCVTF V27.4S, V26.4S
	VFMLA V28.S4, V27.S4, V23.S4 // acc[act 3][wrow 3]

	SUBS $1, R3, R3
	BNE  tile4x4loop

	// Same FADDP-FADDP horizontal fold as dotW4A8SplitHalf4Row, 16 times.
	WORD  $0x6E28D508 // FADDP V8.4S, V8.4S, V8.4S
	WORD  $0x6E28D508
	FMOVS F8, (R4) // dst[act 0][wrow 0]
	WORD  $0x6E29D529 // FADDP V9.4S, V9.4S, V9.4S
	WORD  $0x6E29D529
	FMOVS F9, 4(R4) // dst[act 0][wrow 1]
	WORD  $0x6E2AD54A // FADDP V10.4S, V10.4S, V10.4S
	WORD  $0x6E2AD54A
	FMOVS F10, 8(R4) // dst[act 0][wrow 2]
	WORD  $0x6E2BD56B // FADDP V11.4S, V11.4S, V11.4S
	WORD  $0x6E2BD56B
	FMOVS F11, 12(R4) // dst[act 0][wrow 3]
	WORD  $0x6E2CD58C // FADDP V12.4S, V12.4S, V12.4S
	WORD  $0x6E2CD58C
	FMOVS F12, 16(R4) // dst[act 1][wrow 0]
	WORD  $0x6E2DD5AD // FADDP V13.4S, V13.4S, V13.4S
	WORD  $0x6E2DD5AD
	FMOVS F13, 20(R4) // dst[act 1][wrow 1]
	WORD  $0x6E2ED5CE // FADDP V14.4S, V14.4S, V14.4S
	WORD  $0x6E2ED5CE
	FMOVS F14, 24(R4) // dst[act 1][wrow 2]
	WORD  $0x6E2FD5EF // FADDP V15.4S, V15.4S, V15.4S
	WORD  $0x6E2FD5EF
	FMOVS F15, 28(R4) // dst[act 1][wrow 3]
	WORD  $0x6E30D610 // FADDP V16.4S, V16.4S, V16.4S
	WORD  $0x6E30D610
	FMOVS F16, 32(R4) // dst[act 2][wrow 0]
	WORD  $0x6E31D631 // FADDP V17.4S, V17.4S, V17.4S
	WORD  $0x6E31D631
	FMOVS F17, 36(R4) // dst[act 2][wrow 1]
	WORD  $0x6E32D652 // FADDP V18.4S, V18.4S, V18.4S
	WORD  $0x6E32D652
	FMOVS F18, 40(R4) // dst[act 2][wrow 2]
	WORD  $0x6E33D673 // FADDP V19.4S, V19.4S, V19.4S
	WORD  $0x6E33D673
	FMOVS F19, 44(R4) // dst[act 2][wrow 3]
	WORD  $0x6E34D694 // FADDP V20.4S, V20.4S, V20.4S
	WORD  $0x6E34D694
	FMOVS F20, 48(R4) // dst[act 3][wrow 0]
	WORD  $0x6E35D6B5 // FADDP V21.4S, V21.4S, V21.4S
	WORD  $0x6E35D6B5
	FMOVS F21, 52(R4) // dst[act 3][wrow 1]
	WORD  $0x6E36D6D6 // FADDP V22.4S, V22.4S, V22.4S
	WORD  $0x6E36D6D6
	FMOVS F22, 56(R4) // dst[act 3][wrow 2]
	WORD  $0x6E37D6F7 // FADDP V23.4S, V23.4S, V23.4S
	WORD  $0x6E37D6F7
	FMOVS F23, 60(R4) // dst[act 3][wrow 3]
	RET
