// NEON int8×int8→int32 dot product for arm64 (base ARMv8.0 — SMULL/SADALP, no
// DotProd/SDOT extension, so it runs on every arm64 incl. older cores). Each
// iteration multiplies 16 int8 pairs to int16 (SMULL low + SMULL2 high) and
// pairwise-accumulates them into a 4-lane int32 accumulator (SADALP). The 4
// lanes are horizontally summed (ADDV) at the end. The Go caller handles the
// n%16 tail.
//
// Go's arm64 assembler has no mnemonic for the signed integer widening multiply
// (SMULL) or signed pairwise-accumulate-long (SADALP), so those three encodings
// are emitted as raw WORDs (verified bit-exact vs the scalar reference under
// qemu-aarch64 — see TestDotI8_matchesScalar). int8×int8 ≤ 127² = 16129 fits
// int16; the int32 accumulator holds Σ over the transformer's K (≤ ~16k) easily.

#include "textflag.h"

// func dotI8NEON(a *int8, b *int8, n int) int32
TEXT ·dotI8NEON(SB), NOSPLIT, $0-28
	MOVD a+0(FP), R0   // &a[0]
	MOVD b+8(FP), R1   // &b[0]
	MOVD n+16(FP), R2  // n (multiple of 16)

	VMOVI $0, V4.B16   // int32 accumulator (4 lanes), zeroed
	LSR   $4, R2, R3   // R3 = n / 16 iterations
	CBZ   R3, reduce

loop:
	VLD1.P 16(R0), [V0.B16]    // 16 int8 from a
	VLD1.P 16(R1), [V1.B16]    // 16 int8 from b
	WORD   $0x0E21C002         // SMULL  V2.8H, V0.8B,  V1.8B   (low 8: a*b → int16)
	WORD   $0x4E21C003         // SMULL2 V3.8H, V0.16B, V1.16B  (high 8: a*b → int16)
	WORD   $0x4E606844         // SADALP V4.4S, V2.8H           (V4 += pairwise(V2))
	WORD   $0x4E606864         // SADALP V4.4S, V3.8H           (V4 += pairwise(V3))
	SUBS   $1, R3, R3
	BNE    loop

reduce:
	VADDV V4.S4, V5    // horizontal add the 4 int32 lanes → V5[0]
	VMOV  V5.S[0], R4  // move the scalar int32 to a GP register
	MOVW  R4, ret+24(FP)
	RET
