// docs/task-simd-audit.md Appendix B.
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

// func sdotIssuePeak(iters int) int32
//
// Measures how many SDOTs a single P-core retires per cycle, with NO memory
// traffic and NO dependency chain short enough to bind: 32 SDOTs per iteration
// spread over 16 independent accumulators, both source registers loop-invariant.
// The only non-SDOT work is the counter and the branch.
//
// This exists to settle an item docs/task-simd-audit.md Appendix B lists as
// unconfirmed -- "whether SDOT issues on four pipes or two", with the note that
// two would halve S-01's counted gain. It is measured here rather than inferred
// from a kernel, because a kernel's rate is a floor on issue width and can be
// held down by loads or latency, which is exactly how such a constant gets
// recorded wrong. At the M1 Pro P-core's 3.2 GHz the ceilings are 2 pipes =
// 6.4 G SDOT/s, 3 = 9.6, 4 = 12.8; see TestSDOTIssuePeak for the reading and the
// ceiling assert that keeps a compiler-folded result from being believed.
TEXT ·sdotIssuePeak(SB), NOSPLIT, $0-12
	MOVD iters+0(FP), R0
	VMOVI $1, V0.B16
	VMOVI $2, V1.B16
	VMOVI $0, V8.B16
	VMOVI $0, V9.B16
	VMOVI $0, V10.B16
	VMOVI $0, V11.B16
	VMOVI $0, V12.B16
	VMOVI $0, V13.B16
	VMOVI $0, V14.B16
	VMOVI $0, V15.B16
	VMOVI $0, V16.B16
	VMOVI $0, V17.B16
	VMOVI $0, V18.B16
	VMOVI $0, V19.B16
	VMOVI $0, V20.B16
	VMOVI $0, V21.B16
	VMOVI $0, V22.B16
	VMOVI $0, V23.B16

sdotpeakloop:
	WORD $0x4E819408 // SDOT V8.4S, V0.16B, V1.16B
	WORD $0x4E819409 // SDOT V9.4S, V0.16B, V1.16B
	WORD $0x4E81940A // SDOT V10.4S, V0.16B, V1.16B
	WORD $0x4E81940B // SDOT V11.4S, V0.16B, V1.16B
	WORD $0x4E81940C // SDOT V12.4S, V0.16B, V1.16B
	WORD $0x4E81940D // SDOT V13.4S, V0.16B, V1.16B
	WORD $0x4E81940E // SDOT V14.4S, V0.16B, V1.16B
	WORD $0x4E81940F // SDOT V15.4S, V0.16B, V1.16B
	WORD $0x4E819410 // SDOT V16.4S, V0.16B, V1.16B
	WORD $0x4E819411 // SDOT V17.4S, V0.16B, V1.16B
	WORD $0x4E819412 // SDOT V18.4S, V0.16B, V1.16B
	WORD $0x4E819413 // SDOT V19.4S, V0.16B, V1.16B
	WORD $0x4E819414 // SDOT V20.4S, V0.16B, V1.16B
	WORD $0x4E819415 // SDOT V21.4S, V0.16B, V1.16B
	WORD $0x4E819416 // SDOT V22.4S, V0.16B, V1.16B
	WORD $0x4E819417 // SDOT V23.4S, V0.16B, V1.16B
	WORD $0x4E819408 // SDOT V8.4S, V0.16B, V1.16B
	WORD $0x4E819409 // SDOT V9.4S, V0.16B, V1.16B
	WORD $0x4E81940A // SDOT V10.4S, V0.16B, V1.16B
	WORD $0x4E81940B // SDOT V11.4S, V0.16B, V1.16B
	WORD $0x4E81940C // SDOT V12.4S, V0.16B, V1.16B
	WORD $0x4E81940D // SDOT V13.4S, V0.16B, V1.16B
	WORD $0x4E81940E // SDOT V14.4S, V0.16B, V1.16B
	WORD $0x4E81940F // SDOT V15.4S, V0.16B, V1.16B
	WORD $0x4E819410 // SDOT V16.4S, V0.16B, V1.16B
	WORD $0x4E819411 // SDOT V17.4S, V0.16B, V1.16B
	WORD $0x4E819412 // SDOT V18.4S, V0.16B, V1.16B
	WORD $0x4E819413 // SDOT V19.4S, V0.16B, V1.16B
	WORD $0x4E819414 // SDOT V20.4S, V0.16B, V1.16B
	WORD $0x4E819415 // SDOT V21.4S, V0.16B, V1.16B
	WORD $0x4E819416 // SDOT V22.4S, V0.16B, V1.16B
	WORD $0x4E819417 // SDOT V23.4S, V0.16B, V1.16B
	SUBS $1, R0, R0
	BNE  sdotpeakloop

	// Consume one accumulator so nothing above can be treated as dead.
	VADDV V8.S4, V24
	VMOV  V24.S[0], R1
	MOVW  R1, ret+8(FP)
	RET
