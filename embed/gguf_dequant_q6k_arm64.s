// NEON body of the Q6_K dequantizer for arm64 (gguf_dequant_q6k_arm64.go).
//
// A super-block is two chunks of 128 weights. A chunk reads 64 bytes of ql (low four bits) and 32 of qh (high two
// bits, four quants to a byte). For l in 0..31, with a = ql[l], b = ql[l+32], h = qh[l], the chunk's four runs of 32
// weights take the quants
//
//	(a & 15) | (h<<4 & 48)      (b & 15) | (h<<2 & 48)      (a >> 4) | (h & 48)      (b >> 4) | (h>>2 & 48)
//
// each less 32. Sixteen l at a time share their four multipliers, so the loop is four steps of 16: build the four
// quant vectors as bytes, subtract 32, sign-extend to 32 bits, SCVTF (exact, the values are -32..31), FMUL by the
// multiplier, store 16 weights into each run. One float32 multiply per weight, as the scalar.
//
// All base ARMv8-A NEON, no feature detection.

#include "textflag.h"

// func dequantQ6KBodyNEON(q *byte, out *float32, ds *[16]float32)
TEXT ·dequantQ6KBodyNEON(SB), NOSPLIT, $0-24
	MOVD q+0(FP), R0          // ql
	MOVD out+8(FP), R1
	MOVD ds+16(FP), R2
	ADD  $128, R0, R3         // qh
	MOVD $15, R4
	VDUP R4, V30.B16
	MOVD $48, R4
	VDUP R4, V31.B16
	MOVD $32, R4
	VDUP R4, V29.B16
	MOVD $2, R5               // two chunks

chunk:
	MOVD $2, R6               // two steps of 16 l

step:
	VLD1   (R0), [V0.B16]     // a
	ADD    $32, R0, R7
	VLD1   (R7), [V1.B16]     // b
	VLD1.P 16(R3), [V2.B16]   // h
	ADD    $16, R0

	MOVWU  (R2), R8
	VDUP   R8, V20.S4
	MOVWU  4(R2), R8
	VDUP   R8, V21.S4
	MOVWU  8(R2), R8
	VDUP   R8, V22.S4
	MOVWU  12(R2), R8
	VDUP   R8, V23.S4
	ADD    $16, R2

	VAND   V30.B16, V0.B16, V3.B16    // a & 15
	VSHL   $4, V2.B16, V16.B16
	VAND   V31.B16, V16.B16, V16.B16
	VORR   V16.B16, V3.B16, V3.B16
	VSUB   V29.B16, V3.B16, V3.B16    // run 0's quants - 32

	VAND   V30.B16, V1.B16, V4.B16    // b & 15
	VSHL   $2, V2.B16, V16.B16
	VAND   V31.B16, V16.B16, V16.B16
	VORR   V16.B16, V4.B16, V4.B16
	VSUB   V29.B16, V4.B16, V4.B16    // run 1

	VUSHR  $4, V0.B16, V5.B16         // a >> 4
	VAND   V31.B16, V2.B16, V16.B16
	VORR   V16.B16, V5.B16, V5.B16
	VSUB   V29.B16, V5.B16, V5.B16    // run 2

	VUSHR  $4, V1.B16, V6.B16         // b >> 4
	VUSHR  $2, V2.B16, V16.B16
	VAND   V31.B16, V16.B16, V16.B16
	VORR   V16.B16, V6.B16, V6.B16
	VSUB   V29.B16, V6.B16, V6.B16    // run 3

	VSXTL  V3.B8, V8.H8
	VSXTL2 V3.B16, V9.H8
	VSXTL  V8.H4, V10.S4
	VSXTL2 V8.H8, V11.S4
	VSXTL  V9.H4, V12.S4
	VSXTL2 V9.H8, V13.S4
	VSCVTF V10.S4, V10.S4
	VSCVTF V11.S4, V11.S4
	VSCVTF V12.S4, V12.S4
	VSCVTF V13.S4, V13.S4
	VFMUL  V20.S4, V10.S4, V10.S4
	VFMUL  V20.S4, V11.S4, V11.S4
	VFMUL  V20.S4, V12.S4, V12.S4
	VFMUL  V20.S4, V13.S4, V13.S4
	VST1   [V10.S4, V11.S4, V12.S4, V13.S4], (R1)

	VSXTL  V4.B8, V8.H8
	VSXTL2 V4.B16, V9.H8
	VSXTL  V8.H4, V10.S4
	VSXTL2 V8.H8, V11.S4
	VSXTL  V9.H4, V12.S4
	VSXTL2 V9.H8, V13.S4
	VSCVTF V10.S4, V10.S4
	VSCVTF V11.S4, V11.S4
	VSCVTF V12.S4, V12.S4
	VSCVTF V13.S4, V13.S4
	VFMUL  V21.S4, V10.S4, V10.S4
	VFMUL  V21.S4, V11.S4, V11.S4
	VFMUL  V21.S4, V12.S4, V12.S4
	VFMUL  V21.S4, V13.S4, V13.S4
	ADD    $128, R1, R9
	VST1   [V10.S4, V11.S4, V12.S4, V13.S4], (R9)

	VSXTL  V5.B8, V8.H8
	VSXTL2 V5.B16, V9.H8
	VSXTL  V8.H4, V10.S4
	VSXTL2 V8.H8, V11.S4
	VSXTL  V9.H4, V12.S4
	VSXTL2 V9.H8, V13.S4
	VSCVTF V10.S4, V10.S4
	VSCVTF V11.S4, V11.S4
	VSCVTF V12.S4, V12.S4
	VSCVTF V13.S4, V13.S4
	VFMUL  V22.S4, V10.S4, V10.S4
	VFMUL  V22.S4, V11.S4, V11.S4
	VFMUL  V22.S4, V12.S4, V12.S4
	VFMUL  V22.S4, V13.S4, V13.S4
	ADD    $256, R1, R9
	VST1   [V10.S4, V11.S4, V12.S4, V13.S4], (R9)

	VSXTL  V6.B8, V8.H8
	VSXTL2 V6.B16, V9.H8
	VSXTL  V8.H4, V10.S4
	VSXTL2 V8.H8, V11.S4
	VSXTL  V9.H4, V12.S4
	VSXTL2 V9.H8, V13.S4
	VSCVTF V10.S4, V10.S4
	VSCVTF V11.S4, V11.S4
	VSCVTF V12.S4, V12.S4
	VSCVTF V13.S4, V13.S4
	VFMUL  V23.S4, V10.S4, V10.S4
	VFMUL  V23.S4, V11.S4, V11.S4
	VFMUL  V23.S4, V12.S4, V12.S4
	VFMUL  V23.S4, V13.S4, V13.S4
	ADD    $384, R1, R9
	VST1   [V10.S4, V11.S4, V12.S4, V13.S4], (R9)

	ADD  $64, R1              // 16 weights on in every run
	SUBS $1, R6, R6
	BNE  step

	ADD  $32, R0              // past the chunk's second 32 bytes of ql, read as b
	ADD  $384, R1             // past the chunk's runs 1 to 3
	SUBS $1, R5, R5
	BNE  chunk

	RET
