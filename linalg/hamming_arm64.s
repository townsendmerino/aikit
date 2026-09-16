//go:build arm64

#include "textflag.h"

// func hammingNEON(q *uint64, codes *uint64, words int, n int, dst *uint16)
//
// dst[i] = Σ_j popcount(codes[i*words+j] ^ q[j]).
//
// Arguments:
//   q:     R0
//   codes: R1
//   words: R2
//   n:     R3
//   dst:   R4
TEXT ·hammingNEON(SB), NOSPLIT, $0-40
	MOVD	q+0(FP), R0
	MOVD	codes+8(FP), R1
	MOVD	words+16(FP), R2
	MOVD	n+24(FP), R3
	MOVD	dst+32(FP), R4

	CBZ	R3, done
	CBZ	R2, done

	// Check for specialized word sizes
	CMP	$4, R2
	BEQ	words4_entry

	CMP	$12, R2
	BEQ	words12_entry

general_entry:
	// NOT CURRENTLY DISPATCHED — hamming_arm64.go only calls into this file
	// for words==4 or words==12. This path's 8-bit-per-lane VADD accumulator
	// overflows for words>=64 (dim>~4032; confirmed via a differential test
	// against hammingRowsGeneric: words=64, maximally-differing input, wraps
	// 4096 to 0). Fix the accumulator (periodic widen, e.g. every ~30
	// iterations) and re-verify before wiring this back in.
	//
	// General loop for arbitrary words
	// Pre-compute row increment in bytes: R7 = words * 8
	LSL	$3, R2, R7

row_loop_gen:
	MOVD	R0, R8       // R8 = q
	MOVD	R1, R9       // R9 = current row code pointer
	ADD	R7, R1, R1   // R1 = next row code pointer

	VMOVI	$0, V0.B16   // V0 = 16-byte popcount accumulator

	LSR	$1, R2, R10  // R10 = words / 2 (number of 16-byte pairs)
	CBZ	R10, gen_tail

gen_loop16:
	VLD1.P	16(R8), [V1.B16]
	VLD1.P	16(R9), [V2.B16]
	VEOR	V1.B16, V2.B16, V3.B16
	VCNT	V3.B16, V3.B16
	VADD	V3.B16, V0.B16, V0.B16
	SUBS	$1, R10, R10
	BNE	gen_loop16

gen_tail:
	VUADDLV	V0.B16, V4
	VMOV	V4.H[0], R11

	AND	$1, R2, R10  // R10 = words % 2
	CBZ	R10, gen_store

	// Odd word: load 8 bytes via GPRs and add to R11
	MOVD	(R8), R12
	MOVD	(R9), R13
	EOR	R12, R13, R12
	FMOVD	R12, F1
	VCNT	V1.B8, V1.B8
	VUADDLV	V1.B8, V2
	VMOV	V2.H[0], R14
	ADD	R14, R11, R11

gen_store:
	MOVH	R11, 0(R4)
	ADD	$2, R4, R4

	SUBS	$1, R3, R3
	BNE	row_loop_gen
	RET

// -------------------------------------------------------------
// Specialized path for words == 4 (dim 256, Model2Vec)
// 32 bytes per row. Query is kept in V1, V2 across all rows.
// -------------------------------------------------------------
words4_entry:
	VLD1	(R0), [V1.B16, V2.B16]

row_loop_w4:
	VLD1.P	32(R1), [V3.B16, V4.B16]
	VEOR	V1.B16, V3.B16, V5.B16
	VEOR	V2.B16, V4.B16, V6.B16
	VCNT	V5.B16, V5.B16
	VCNT	V6.B16, V6.B16
	VADD	V5.B16, V6.B16, V5.B16
	VUADDLV	V5.B16, V0
	VMOV	V0.H[0], R8
	MOVH	R8, 0(R4)
	ADD	$2, R4, R4

	SUBS	$1, R3, R3
	BNE	row_loop_w4
	RET

// -------------------------------------------------------------
// Specialized path for words == 12 (dim 768, CodeRankEmbed / BERT)
// 96 bytes per row. Query is kept in V1..V6 across all rows.
// -------------------------------------------------------------
words12_entry:
	VLD1.P	64(R0), [V1.B16, V2.B16, V3.B16, V4.B16]
	VLD1	(R0), [V5.B16, V6.B16]

row_loop_w12:
	VLD1.P	64(R1), [V7.B16, V8.B16, V9.B16, V10.B16]
	VLD1.P	32(R1), [V11.B16, V12.B16]
	VEOR	V1.B16, V7.B16, V13.B16
	VEOR	V2.B16, V8.B16, V14.B16
	VEOR	V3.B16, V9.B16, V15.B16
	VEOR	V4.B16, V10.B16, V16.B16
	VEOR	V5.B16, V11.B16, V17.B16
	VEOR	V6.B16, V12.B16, V18.B16
	VCNT	V13.B16, V13.B16
	VCNT	V14.B16, V14.B16
	VCNT	V15.B16, V15.B16
	VCNT	V16.B16, V16.B16
	VCNT	V17.B16, V17.B16
	VCNT	V18.B16, V18.B16
	VADD	V13.B16, V14.B16, V13.B16
	VADD	V15.B16, V16.B16, V15.B16
	VADD	V17.B16, V18.B16, V17.B16
	VADD	V13.B16, V15.B16, V13.B16
	VADD	V13.B16, V17.B16, V13.B16
	VUADDLV	V13.B16, V0
	VMOV	V0.H[0], R8
	MOVH	R8, 0(R4)
	ADD	$2, R4, R4

	SUBS	$1, R3, R3
	BNE	row_loop_w12
	RET

done:
	RET

