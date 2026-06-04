// NEON int8×int8→int32 dot product using the ARMv8.2 DotProd extension (SDOT),
// the faster sibling of the base-ISA SMULL/SADALP kernel in dot_i8_arm64.s. A
// single SDOT consumes 16 int8 pairs and accumulates their products straight
// into a 4-lane int32 register — replacing the base kernel's SMULL + SMULL2 +
// SADALP + SADALP (four instructions) with one. Four independent accumulators
// (V4..V7) hide SDOT latency; a 64-wide main loop feeds them, a 16-wide tail
// mops up the rest (the Go caller guarantees n is a multiple of 16 and handles
// the sub-16 remainder).
//
// SDOT has no Go arm64 assembler mnemonic, so it is emitted as a raw WORD
// (verified bit-exact vs the scalar reference under qemu-aarch64 with a
// DotProd-capable CPU — see TestDotI8SDOT_matchesScalar). dotI8 only calls this
// when detectDotProd() found HWCAP_ASIMDDP, so the SDOT opcode never reaches a
// core that would trap on it.
//
// SDOT Vd.4S, Vn.16B, Vm.16B encodes as 0x4E809400 | (Rm<<16) | (Rn<<5) | Rd.
// Here Rn=V0, Rm=V1, so the four accumulators V4..V7 are 0x4E819404..07.

#include "textflag.h"

// func dotI8SDOT(a *int8, b *int8, n int) int32
TEXT ·dotI8SDOT(SB), NOSPLIT, $0-28
	MOVD a+0(FP), R0   // &a[0]
	MOVD b+8(FP), R1   // &b[0]
	MOVD n+16(FP), R2  // n (multiple of 16)

	VMOVI $0, V4.B16   // four int32 accumulators (4 lanes each), zeroed
	VMOVI $0, V5.B16
	VMOVI $0, V6.B16
	VMOVI $0, V7.B16

	LSR $6, R2, R3     // R3 = n / 64 (blocks of 4×16)
	CBZ R3, tail_setup

block:
	VLD1.P 16(R0), [V0.B16]
	VLD1.P 16(R1), [V1.B16]
	WORD   $0x4E819404         // SDOT V4.4S, V0.16B, V1.16B
	VLD1.P 16(R0), [V0.B16]
	VLD1.P 16(R1), [V1.B16]
	WORD   $0x4E819405         // SDOT V5.4S, V0.16B, V1.16B
	VLD1.P 16(R0), [V0.B16]
	VLD1.P 16(R1), [V1.B16]
	WORD   $0x4E819406         // SDOT V6.4S, V0.16B, V1.16B
	VLD1.P 16(R0), [V0.B16]
	VLD1.P 16(R1), [V1.B16]
	WORD   $0x4E819407         // SDOT V7.4S, V0.16B, V1.16B
	SUBS   $1, R3, R3
	BNE    block

tail_setup:
	AND $63, R2, R4    // R4 = n % 64
	LSR $4, R4, R4     // R4 = (n%64) / 16 leftover chunks
	CBZ R4, reduce

tail:
	VLD1.P 16(R0), [V0.B16]
	VLD1.P 16(R1), [V1.B16]
	WORD   $0x4E819404         // SDOT V4.4S, V0.16B, V1.16B
	SUBS   $1, R4, R4
	BNE    tail

reduce:
	VADDV V4.S4, V8    // horizontal-add each accumulator, sum in a GP register
	VMOV  V8.S[0], R5
	VADDV V5.S4, V8
	VMOV  V8.S[0], R6
	ADD   R6, R5, R5
	VADDV V6.S4, V8
	VMOV  V8.S[0], R6
	ADD   R6, R5, R5
	VADDV V7.S4, V8
	VMOV  V8.S[0], R6
	ADD   R6, R5, R5
	MOVW  R5, ret+24(FP)
	RET
