// Per-16 int8×int8→int32 partial dot products via the ARMv8.2 DotProd extension (SDOT), for the
// native K-quant kernel (kquant.go). A K-quant superblock's 16 sub-blocks each carry a distinct
// integer sub-scale, so their dot products cannot be summed by one wide SDOT — each 16-wide
// sub-block must reduce to its own int32 before the Go side weights it. This routine emits those
// nsub partials in one call (nsub SDOTs), so the hot path pays one Go→asm crossing per superblock
// rather than one per sub-block.
//
// SDOT has no Go arm64 mnemonic; it is the same raw WORD used in dot_i8dp_arm64.s —
// SDOT V4.4S, V0.16B, V1.16B = 0x4E819404 — reached only when cpufeat.DetectDotProd() is true, so the
// opcode never hits a core that would trap. Output is pure integer, so it is bit-identical to the
// scalar dotPartials16Scalar reference (TestKQuantDotPartials_matchesScalar).

#include "textflag.h"

// func dotPartials16SDOT(codes *int8, qs *int8, nsub int, out *int32)
// out[j] = Σ_{i=0..16} codes[j*16+i] * qs[j*16+i], for j in 0..nsub.
TEXT ·dotPartials16SDOT(SB), NOSPLIT, $0-32
	MOVD codes+0(FP), R0
	MOVD qs+8(FP), R1
	MOVD nsub+16(FP), R2
	MOVD out+24(FP), R3
	CBZ  R2, done

loop:
	VLD1.P 16(R0), [V0.B16]   // 16 weight codes
	VLD1.P 16(R1), [V1.B16]   // 16 activations
	VMOVI  $0, V4.B16          // zero the accumulator (SDOT accumulates into it)
	WORD   $0x4E819404         // SDOT V4.4S, V0.16B, V1.16B
	VADDV  V4.S4, V8           // horizontal-add the 4 lanes
	VMOV   V8.S[0], R5
	MOVW   R5, (R3)            // out[j] = partial
	ADD    $4, R3, R3
	SUBS   $1, R2, R2
	BNE    loop

done:
	RET
