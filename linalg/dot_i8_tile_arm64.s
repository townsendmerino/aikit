// docs/task-simd-audit.md S-01b.
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

// func dotI8Tile4x4(act *int8, actStride int, w *int8, wStride int, dst *int32, n int)
//
// task-simd-audit.md S-01b: the register-blocked W8A8 tile. Four activation rows
// against four weight rows, 16 int32 dots per call, n a multiple of 16 (the Go
// caller mops up the remainder, exactly as dotI8 does for dotI8SDOT).
//
// WHY THIS IS FASTER, and it is not the same reason as the W4A8 tile. There is no
// nibble unpack to share here -- int8 weights feed SDOT directly. The binding
// constraint on dotI8SDOT is instead a pair of near-coincident limits: four int32
// accumulators at ~4-cycle SDOT latency cap it at 1 SDOT/cycle, and its 2 loads per
// SDOT need 2 of the 3 load slots to sustain even that. Its measured 49.2 GMAC/s is
// ~0.95 SDOT/cycle, i.e. sitting on both.
//
// The tile breaks both at once. Sixteen accumulators leave far more in flight than
// the latency needs, and sharing each load across the other dimension takes the
// ratio from 2 loads per SDOT to 8 loads per 16 SDOTs -- 0.5. At 3 load slots per
// cycle that would feed 6 SDOTs/cycle, so the load port stops being the constraint
// and SDOT issue becomes it.
//
// BIT-IDENTICAL FOR FREE, unlike the W4A8 tile which had to preserve an f32
// reduction order. Every partial here is int32 and integer addition is exact and
// associative, so any grouping of the same products yields the same integer --
// which is the property dotI8 already documents ("identical to dotI8Scalar
// regardless of the path taken"). The int32 overflow envelope is unchanged too:
// |sum| <= n*127*127 as before, so this inherits dotI8SDOT's contract rather than
// narrowing it.
//
//   V0-V3   the four activation rows' current 16 bytes
//   V4-V7   the four weight rows' current 16 bytes
//   V8-V23  the 16 int32 accumulators, acc(m,r) = V(8+m*4+r)
//   V24     VADDV scratch in the reduction
TEXT ·dotI8Tile4x4(SB), NOSPLIT, $0-48
	MOVD act+0(FP), R0
	MOVD actStride+8(FP), R10
	MOVD w+16(FP), R4
	MOVD wStride+24(FP), R11
	MOVD dst+32(FP), R8
	MOVD n+40(FP), R9

	// Four activation rows actStride apart, four weight rows wStride apart. Each
	// of the eight pointers post-increments by 16 per chunk, so no index math runs
	// in the loop.
	ADD R0, R10, R1
	ADD R1, R10, R2
	ADD R2, R10, R3
	ADD R4, R11, R5
	ADD R5, R11, R6
	ADD R6, R11, R7

	LSR $4, R9, R9 // n / 16 chunks
	VMOVI $0, V8.B16 // acc[act 0][wrow 0]
	VMOVI $0, V9.B16 // acc[act 0][wrow 1]
	VMOVI $0, V10.B16 // acc[act 0][wrow 2]
	VMOVI $0, V11.B16 // acc[act 0][wrow 3]
	VMOVI $0, V12.B16 // acc[act 1][wrow 0]
	VMOVI $0, V13.B16 // acc[act 1][wrow 1]
	VMOVI $0, V14.B16 // acc[act 1][wrow 2]
	VMOVI $0, V15.B16 // acc[act 1][wrow 3]
	VMOVI $0, V16.B16 // acc[act 2][wrow 0]
	VMOVI $0, V17.B16 // acc[act 2][wrow 1]
	VMOVI $0, V18.B16 // acc[act 2][wrow 2]
	VMOVI $0, V19.B16 // acc[act 2][wrow 3]
	VMOVI $0, V20.B16 // acc[act 3][wrow 0]
	VMOVI $0, V21.B16 // acc[act 3][wrow 1]
	VMOVI $0, V22.B16 // acc[act 3][wrow 2]
	VMOVI $0, V23.B16 // acc[act 3][wrow 3]

i8tile4x4loop:
	VLD1.P 16(R0), [V0.B16] // activation row 0
	VLD1.P 16(R1), [V1.B16] // activation row 1
	VLD1.P 16(R2), [V2.B16] // activation row 2
	VLD1.P 16(R3), [V3.B16] // activation row 3
	VLD1.P 16(R4), [V4.B16] // weight row 0
	VLD1.P 16(R5), [V5.B16] // weight row 1
	VLD1.P 16(R6), [V6.B16] // weight row 2
	VLD1.P 16(R7), [V7.B16] // weight row 3

	WORD $0x4E849408 // SDOT V8.4S, V0.16B, V4.16B  acc[act 0][wrow 0]
	WORD $0x4E859409 // SDOT V9.4S, V0.16B, V5.16B  acc[act 0][wrow 1]
	WORD $0x4E86940A // SDOT V10.4S, V0.16B, V6.16B  acc[act 0][wrow 2]
	WORD $0x4E87940B // SDOT V11.4S, V0.16B, V7.16B  acc[act 0][wrow 3]
	WORD $0x4E84942C // SDOT V12.4S, V1.16B, V4.16B  acc[act 1][wrow 0]
	WORD $0x4E85942D // SDOT V13.4S, V1.16B, V5.16B  acc[act 1][wrow 1]
	WORD $0x4E86942E // SDOT V14.4S, V1.16B, V6.16B  acc[act 1][wrow 2]
	WORD $0x4E87942F // SDOT V15.4S, V1.16B, V7.16B  acc[act 1][wrow 3]
	WORD $0x4E849450 // SDOT V16.4S, V2.16B, V4.16B  acc[act 2][wrow 0]
	WORD $0x4E859451 // SDOT V17.4S, V2.16B, V5.16B  acc[act 2][wrow 1]
	WORD $0x4E869452 // SDOT V18.4S, V2.16B, V6.16B  acc[act 2][wrow 2]
	WORD $0x4E879453 // SDOT V19.4S, V2.16B, V7.16B  acc[act 2][wrow 3]
	WORD $0x4E849474 // SDOT V20.4S, V3.16B, V4.16B  acc[act 3][wrow 0]
	WORD $0x4E859475 // SDOT V21.4S, V3.16B, V5.16B  acc[act 3][wrow 1]
	WORD $0x4E869476 // SDOT V22.4S, V3.16B, V6.16B  acc[act 3][wrow 2]
	WORD $0x4E879477 // SDOT V23.4S, V3.16B, V7.16B  acc[act 3][wrow 3]

	SUBS $1, R9, R9
	BNE  i8tile4x4loop

	// Horizontal-add each accumulator to its int32 output. Lane order is
	// irrelevant: exact integer addition.
	VADDV V8.S4, V24
	VMOV  V24.S[0], R10
	MOVW  R10, 0(R8) // dst[act 0][wrow 0]
	VADDV V9.S4, V24
	VMOV  V24.S[0], R10
	MOVW  R10, 4(R8) // dst[act 0][wrow 1]
	VADDV V10.S4, V24
	VMOV  V24.S[0], R10
	MOVW  R10, 8(R8) // dst[act 0][wrow 2]
	VADDV V11.S4, V24
	VMOV  V24.S[0], R10
	MOVW  R10, 12(R8) // dst[act 0][wrow 3]
	VADDV V12.S4, V24
	VMOV  V24.S[0], R10
	MOVW  R10, 16(R8) // dst[act 1][wrow 0]
	VADDV V13.S4, V24
	VMOV  V24.S[0], R10
	MOVW  R10, 20(R8) // dst[act 1][wrow 1]
	VADDV V14.S4, V24
	VMOV  V24.S[0], R10
	MOVW  R10, 24(R8) // dst[act 1][wrow 2]
	VADDV V15.S4, V24
	VMOV  V24.S[0], R10
	MOVW  R10, 28(R8) // dst[act 1][wrow 3]
	VADDV V16.S4, V24
	VMOV  V24.S[0], R10
	MOVW  R10, 32(R8) // dst[act 2][wrow 0]
	VADDV V17.S4, V24
	VMOV  V24.S[0], R10
	MOVW  R10, 36(R8) // dst[act 2][wrow 1]
	VADDV V18.S4, V24
	VMOV  V24.S[0], R10
	MOVW  R10, 40(R8) // dst[act 2][wrow 2]
	VADDV V19.S4, V24
	VMOV  V24.S[0], R10
	MOVW  R10, 44(R8) // dst[act 2][wrow 3]
	VADDV V20.S4, V24
	VMOV  V24.S[0], R10
	MOVW  R10, 48(R8) // dst[act 3][wrow 0]
	VADDV V21.S4, V24
	VMOV  V24.S[0], R10
	MOVW  R10, 52(R8) // dst[act 3][wrow 1]
	VADDV V22.S4, V24
	VMOV  V24.S[0], R10
	MOVW  R10, 56(R8) // dst[act 3][wrow 2]
	VADDV V23.S4, V24
	VMOV  V24.S[0], R10
	MOVW  R10, 60(R8) // dst[act 3][wrow 3]
	RET
