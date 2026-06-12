#include "textflag.h"

// func dotNEON2x8(a0, a1, b0, b1, b2, b3, b4, b5, b6, b7 *float32, n4 int, sums *[64]float32)
//
// The MR×NR (2×8) register microkernel: 2 a-rows × 8 b-rows, 16 live 4-lane
// accumulators (V0..V15) held across the whole K loop. Each k-step loads 2 a-vectors
// + 8 b-vectors into distinct registers (no load→FMLA serialisation) and issues 16
// INDEPENDENT FMLAs — so every b-load feeds 2 FMLAs (vs 1 in the 1×8 Dot8x4) and 16
// accumulators hide the ~4-cycle FMA latency across 4 pipes. Caller horizontal-reduces
// the 16 partial-dot vectors and adds the scalar K-tail.
//
// sums layout (row-major, 16 groups of 4): [a0·b0 | a0·b1 | … | a0·b7 | a1·b0 | … | a1·b7].
TEXT ·dotNEON2x8(SB), NOSPLIT, $0-96
	MOVD	a0+0(FP), R0
	MOVD	a1+8(FP), R1
	MOVD	b0+16(FP), R2
	MOVD	b1+24(FP), R3
	MOVD	b2+32(FP), R4
	MOVD	b3+40(FP), R5
	MOVD	b4+48(FP), R6
	MOVD	b5+56(FP), R7
	MOVD	b6+64(FP), R8
	MOVD	b7+72(FP), R9
	MOVD	n4+80(FP), R10
	MOVD	sums+88(FP), R11

	VMOVI	$0, V0.B16
	VMOVI	$0, V1.B16
	VMOVI	$0, V2.B16
	VMOVI	$0, V3.B16
	VMOVI	$0, V4.B16
	VMOVI	$0, V5.B16
	VMOVI	$0, V6.B16
	VMOVI	$0, V7.B16
	VMOVI	$0, V8.B16
	VMOVI	$0, V9.B16
	VMOVI	$0, V10.B16
	VMOVI	$0, V11.B16
	VMOVI	$0, V12.B16
	VMOVI	$0, V13.B16
	VMOVI	$0, V14.B16
	VMOVI	$0, V15.B16

	CBZ	R10, store

loop:
	VLD1.P	16(R0), [V16.S4]     // a0[k..k+4]
	VLD1.P	16(R1), [V17.S4]     // a1[k..k+4]
	VLD1.P	16(R2), [V18.S4]     // b0
	VLD1.P	16(R3), [V19.S4]     // b1
	VLD1.P	16(R4), [V20.S4]     // b2
	VLD1.P	16(R5), [V21.S4]     // b3
	VLD1.P	16(R6), [V22.S4]     // b4
	VLD1.P	16(R7), [V23.S4]     // b5
	VLD1.P	16(R8), [V24.S4]     // b6
	VLD1.P	16(R9), [V25.S4]     // b7

	VFMLA	V16.S4, V18.S4, V0.S4    // a0·b0
	VFMLA	V16.S4, V19.S4, V1.S4    // a0·b1
	VFMLA	V16.S4, V20.S4, V2.S4    // a0·b2
	VFMLA	V16.S4, V21.S4, V3.S4    // a0·b3
	VFMLA	V16.S4, V22.S4, V4.S4    // a0·b4
	VFMLA	V16.S4, V23.S4, V5.S4    // a0·b5
	VFMLA	V16.S4, V24.S4, V6.S4    // a0·b6
	VFMLA	V16.S4, V25.S4, V7.S4    // a0·b7
	VFMLA	V17.S4, V18.S4, V8.S4    // a1·b0
	VFMLA	V17.S4, V19.S4, V9.S4    // a1·b1
	VFMLA	V17.S4, V20.S4, V10.S4   // a1·b2
	VFMLA	V17.S4, V21.S4, V11.S4   // a1·b3
	VFMLA	V17.S4, V22.S4, V12.S4   // a1·b4
	VFMLA	V17.S4, V23.S4, V13.S4   // a1·b5
	VFMLA	V17.S4, V24.S4, V14.S4   // a1·b6
	VFMLA	V17.S4, V25.S4, V15.S4   // a1·b7

	SUBS	$1, R10, R10
	BNE	loop

store:
	VST1.P	[V0.S4], 16(R11)
	VST1.P	[V1.S4], 16(R11)
	VST1.P	[V2.S4], 16(R11)
	VST1.P	[V3.S4], 16(R11)
	VST1.P	[V4.S4], 16(R11)
	VST1.P	[V5.S4], 16(R11)
	VST1.P	[V6.S4], 16(R11)
	VST1.P	[V7.S4], 16(R11)
	VST1.P	[V8.S4], 16(R11)
	VST1.P	[V9.S4], 16(R11)
	VST1.P	[V10.S4], 16(R11)
	VST1.P	[V11.S4], 16(R11)
	VST1.P	[V12.S4], 16(R11)
	VST1.P	[V13.S4], 16(R11)
	VST1.P	[V14.S4], 16(R11)
	VST1	[V15.S4], (R11)
	RET
