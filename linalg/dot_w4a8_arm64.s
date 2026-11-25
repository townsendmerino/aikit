// Fused int4(weight)×int8(activation) GEMV dot using the ARMv8.2 DotProd
// extension (SDOT) — the arm64 W4A8 decode kernel's hot loop, counterpart of
// dot_w4a8_amd64.s. For ONE output it streams a whole K-wide weight row and
// returns the per-group-scaled f32 dot Σ_g scale[g]·(act·w)_g (the activation
// scale is applied by the Go caller). The nibble-unpack prologue (16 packed
// bytes → 32 centered int8 weights: low nibble = even k, high = odd k, −8) feeds
// the proven dot_i8dp SDOT body; the f32 weight scale is then folded IN-REGISTER
// (see below), so there is no per-group horizontal reduce and no Go fold loop.
//
// Looping the groups INSIDE one call is the whole point: it removes both the
// per-weight f32 dequant (MatmulBTQ4's M=1 bottleneck) and the ~18ns/call
// Go↔asm transition a per-group dotI8 loop would pay nGroups times per output.
//
// group is fixed at 32 (16 packed bytes / 32 activations per iteration); the Go
// caller routes other group sizes and any ragged tail (K % 32) to the scalar
// reference. nGroups = K/32.
//
// SDOT has no Go assembler mnemonic, so it is emitted as a raw WORD (same as
// dot_i8dp): SDOT Vd.4S,Vn.16B,Vm.16B = 0x4E809400 | (Rm<<16) | (Rn<<5) | Rd.
// Here V16 += V6·V3 → 0x4E8394D0 and V16 += V7·V4 → 0x4E8494F0. Only called
// after cpufeat.DetectDotProd() (HWCAP_ASIMDDP), so SDOT never traps.

#include "textflag.h"

// func dotW4A8FoldSDOT(act *int8, packed *byte, scales *float32, nGroups int) float32
//
// Validated on M1 Pro (quant_w4a8_test.go 1e-5 vs scalar + BenchmarkQ4vsQ8);
// written from the validated amd64 fold kernel (dotW4A8FoldAVX2), same algorithm
// and parity.
//
// The nibble-unpack + SDOT hot loop, but the per-group f32 weight scale is folded
// IN-REGISTER instead of via a per-group VADDV + SIMD→GP store + a Go fold loop:
// keep V16 as 4 UNREDUCED int32 lanes, SCVTF → f32, FMLA into a 4-lane f32
// accumulator V20 by the broadcast scale[g]. Because every lane of a group carries
// the same scale, ONE FADDP reduce of V20 at the end yields Σ_g scale[g]·groupdot[g].
// This removes the 63/64 per-group VADDVs, the V→GP moves, the int32 scratch
// round-trip, and the Go fold loop — the overhead that made W4A8 ~2× slower than
// W8A8 despite reading half the bytes (amd64: 2.07× → 1.13×; NEON expected to
// reach the byte-ratio ceiling since M1 decode is bandwidth-bound).
//
// SCVTF V18.4S,V16.4S has no Go mnemonic → raw WORD (like SDOT):
// SCVTF(vec,int,4S) = 0x4E21D800 | (Rn<<5) | Rd → 0x4E21DA12 for V16→V18.
TEXT ·dotW4A8FoldSDOT(SB), NOSPLIT, $0-36
	MOVD act+0(FP), R0
	MOVD packed+8(FP), R1
	MOVD scales+16(FP), R2  // &scales[0] (f32, one per group)
	MOVD nGroups+24(FP), R3

	VMOVI $0x0F, V30.B16
	VMOVI $8, V31.B16
	VEOR  V20.B16, V20.B16, V20.B16  // f32 accumulator (4 lanes) = 0

foldloop:
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VZIP1  V2.B16, V1.B16, V3.B16
	VZIP2  V2.B16, V1.B16, V4.B16
	VSUB   V31.B16, V3.B16, V3.B16
	VSUB   V31.B16, V4.B16, V4.B16
	VLD1.P 32(R0), [V6.B16, V7.B16]
	VMOVI  $0, V16.B16
	WORD   $0x4E8394D0               // SDOT V16.4S, V6.16B, V3.16B
	WORD   $0x4E8494F0               // SDOT V16.4S, V7.16B, V4.16B

	// In-register f32 fold (no per-group horizontal reduce):
	WORD   $0x4E21DA12               // SCVTF V18.4S, V16.4S  (int32 → f32)
	VLD1R  (R2), [V19.S4]            // broadcast scale[g] to 4 lanes
	VFMLA  V19.S4, V18.S4, V20.S4    // V20 += V18 · scale[g]
	ADD    $4, R2, R2

	SUBS $1, R3, R3
	BNE  foldloop

	// One pairwise f32 reduce of the 4-lane accumulator → return value. FADDP
	// (vector f32) has no Go mnemonic → raw WORD: FADDP Vd.4S,Vn.4S,Vm.4S =
	// 0x6E20D400 | (Rm<<16) | (Rn<<5) | Rd → 0x6E34D694 for V20,V20,V20.
	WORD  $0x6E34D694                // FADDP V20.4S, V20.4S, V20.4S → [s0+s1, s2+s3, …]
	WORD  $0x6E34D694                // FADDP again → lane0 = Σ
	FMOVS F20, ret+32(FP)
	RET

// func dotW4A8FoldSDOTv2(act *int8, packed *byte, scales *float32, sumAct *int32, nGroups int) float32
//
// docs/task-w4a8-neon-bandwidth.md (goinfer) Gate 1, items 1+2 only (the repack,
// item 3, is a separate follow-up — it needs a second on-disk-adjacent byte
// layout, out of scope here; see that doc's correction note on why the
// canonical packed format itself cannot change).
//
// Drops dotW4A8FoldSDOT's two per-group VSUB centering ops: SDOT runs directly
// on the raw unsigned nibble values (0-15, safe as signed int8 operands — no
// overflow since 15 ≪ 128), computing Σnib_k·act_k instead of Σ(nib_k-8)·act_k.
// The correction this owes back — Σ(nib_k-8)·act_k = Σnib_k·act_k - 8·Σact_k —
// is NOT folded per-group into the same loop (that would cost a GP load + shift
// + lane-insert + subtract per group: MORE instructions than the 2 VSUBs it
// replaces, a measured-before-committing net loss). It runs as a SEPARATE pass
// after the main fold, over the per-group scale[] and the caller-precomputed
// sumAct[] (SumActGroupsInto in quant.go — computed once per token, shared
// across every output row a decode-time W4A8 GEMV evaluates): corr =
// Σ_g scale[g]·float32(sumAct[g]), batched 4 groups per SIMD iteration (this is
// a dot of two PER-GROUP arrays, so 4-wide processes 4 groups, not 4 elements —
// genuinely cheap relative to the main loop's per-element work), scalar tail
// for nGroups%4 so it never reads past either array's real length (avoids
// touching the next row's scales, or worse, running off an mmap'd .giw's
// backing array on the last row). Final result: mainFold - 8·corr.
//
// Bit-identical to dotW4A8UncenteredScalar (quant_w4a8.go) given the same
// sumAct, and dotW4A8UncenteredScalar is itself proven bit-identical to
// dotW4A8Scalar (w4a8_algebra_test.go) — the rearrangement changes no numeric
// result, only which instructions pay for it.
TEXT ·dotW4A8FoldSDOTv2(SB), NOSPLIT, $0-44
	MOVD act+0(FP), R0
	MOVD packed+8(FP), R1
	MOVD scales+16(FP), R2
	MOVD sumAct+24(FP), R4
	MOVD nGroups+32(FP), R3

	MOVD R3, R7 // save nGroups (R3 is consumed by the main loop's SUBS below)
	MOVD R2, R8 // save &scales[0] (R2 is advanced by the main loop below)

	VMOVI $0x0F, V30.B16
	VEOR  V20.B16, V20.B16, V20.B16 // main fold accumulator (4 lanes) = 0

foldloopv2:
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16 // low nibbles, UNCENTERED (0..15)
	VUSHR  $4, V0.B16, V2.B16      // high nibbles, UNCENTERED (0..15)
	VZIP1  V2.B16, V1.B16, V3.B16
	VZIP2  V2.B16, V1.B16, V4.B16
	VLD1.P 32(R0), [V6.B16, V7.B16]
	VMOVI  $0, V16.B16
	WORD   $0x4E8394D0 // SDOT V16.4S, V6.16B, V3.16B
	WORD   $0x4E8494F0 // SDOT V16.4S, V7.16B, V4.16B
	WORD   $0x4E21DA12 // SCVTF V18.4S, V16.4S
	VLD1R  (R2), [V19.S4]
	VFMLA  V19.S4, V18.S4, V20.S4
	ADD    $4, R2, R2

	SUBS $1, R3, R3
	BNE  foldloopv2

	WORD  $0x6E34D694 // FADDP V20.4S, V20.4S, V20.4S
	WORD  $0x6E34D694 // FADDP again → lane0 = main fold total
	FMOVS F20, F9      // stash the main fold result

	// Correction pass: corr = Σ_g scale[g]·float32(sumAct[g]), 4 groups/iter,
	// reusing V16/V18 (free now that the main loop above is done) for the
	// int32→f32 convert — same SCVTF encoding as the main loop.
	VEOR V21.B16, V21.B16, V21.B16 // vector correction accumulator
	VEOR V24.B16, V24.B16, V24.B16 // scalar-tail accumulator (F24 = lane 0)
	LSR  $2, R7, R9                // R9 = nGroups/4
	AND  $3, R7, R10               // R10 = nGroups%4
	CBZ  R9, corrtailcheck

corr4loop:
	VLD1.P 16(R8), [V22.S4] // 4 group scales (f32)
	VLD1.P 16(R4), [V16.S4] // 4 group sumAct (int32)
	WORD   $0x4E21DA12      // SCVTF V18.4S, V16.4S
	VFMLA  V22.S4, V18.S4, V21.S4

	SUBS $1, R9, R9
	BNE  corr4loop

corrtailcheck:
	CBZ R10, corrreduce

corrtailloop:
	FMOVS   (R8), F1
	ADD     $4, R8, R8
	MOVW    (R4), R11
	ADD     $4, R4, R4
	SCVTFWS R11, F2
	FMULS   F1, F2, F2
	FADDS   F2, F24, F24

	SUBS $1, R10, R10
	BNE  corrtailloop

corrreduce:
	WORD  $0x6E35D6B5 // FADDP V21.4S, V21.4S, V21.4S
	WORD  $0x6E35D6B5 // FADDP again → lane0 = vector-correction total
	FADDS F21, F24, F24 // += the scalar tail → F24 = full correction sum (unscaled)
	FADDS F24, F24, F24 // ×2
	FADDS F24, F24, F24 // ×4
	FADDS F24, F24, F24 // ×8 → F24 = 8·corr
	FSUBS F24, F9, F9   // F9 = mainFold - 8·corr
	FMOVS F9, ret+40(FP)
	RET

// func dotW4A8SplitHalfSDOT(act *int8, packed *byte, scales *float32, nGroups int) float32
//
// docs/prompts/w4a8-item3-harness.md (goinfer), item 3's core lever, harness
// phase: 1-row output, signed centering (matches dotW4A8FoldSDOT's contract
// exactly, layout only differs) — isolates the layout change's own effect
// before combining it with the item-4 row-interleave or the item-2
// uncentered-correction axes.
//
// packed must be in the SPLIT-HALF layout (repackSplitHalfRow, harness-only —
// the canonical on-disk/in-memory layout QuantizeGroupInt4Row produces is
// untouched): byte i of a 16-byte group holds block-0's k_local=i weight in
// its low nibble and block-1's k_local=i+16 weight in its high nibble, so
// both halves are ALREADY in sequential k order once masked/shifted out —
// unlike canonical's interleaved-nibble layout, no VZIP1/VZIP2 is needed to
// restore order. This removes exactly those two instructions per group from
// dotW4A8FoldSDOT's unpack prologue; everything else (centering VSUBs, the
// SDOT pair, the in-register f32 fold) is unchanged.
TEXT ·dotW4A8SplitHalfSDOT(SB), NOSPLIT, $0-36
	MOVD act+0(FP), R0
	MOVD packed+8(FP), R1
	MOVD scales+16(FP), R2
	MOVD nGroups+24(FP), R3

	VMOVI $0x0F, V30.B16
	VMOVI $8, V31.B16
	VEOR  V20.B16, V20.B16, V20.B16 // f32 accumulator (4 lanes) = 0

splithalfloop:
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16 // block0 nibbles, k_local 0..15, already sequential
	VUSHR  $4, V0.B16, V2.B16      // block1 nibbles, k_local 16..31, already sequential
	VSUB   V31.B16, V1.B16, V3.B16 // center block0
	VSUB   V31.B16, V2.B16, V4.B16 // center block1
	VLD1.P 32(R0), [V6.B16, V7.B16]
	VMOVI  $0, V16.B16
	WORD   $0x4E8394D0 // SDOT V16.4S, V6.16B, V3.16B  (act[0:16]  · block0)
	WORD   $0x4E8494F0 // SDOT V16.4S, V7.16B, V4.16B  (act[16:32] · block1)
	WORD   $0x4E21DA12 // SCVTF V18.4S, V16.4S
	VLD1R  (R2), [V19.S4]
	VFMLA  V19.S4, V18.S4, V20.S4
	ADD    $4, R2, R2

	SUBS $1, R3, R3
	BNE  splithalfloop

	WORD  $0x6E34D694 // FADDP V20.4S, V20.4S, V20.4S
	WORD  $0x6E34D694 // FADDP again → lane0 = Σ
	FMOVS F20, ret+32(FP)
	RET

// func dotW4A8FoldSDOT2Acc(act *int8, packed *byte, scales *float32, nGroups int) float32
//
// NOT part of the item-3 grid (docs/prompts/w4a8-item3-harness.md) — a probe
// built after that grid's first cell (dotW4A8SplitHalfSDOT) measured a flat
// 1.000x against dotW4A8FoldSDOT, which contradicted Gate 0's recorded
// "issue-limited" verdict (docs/task-w4a8-neon-bandwidth.md). Re-running the
// issue-width probe on a quiet box gave ratio ~0.99-1.03 (NOT issue-limited)
// stably across 4 runs — the original 1.11 reading does not reproduce.
// dotI8SDOT (dot_i8dp_arm64.s) already hides its own SDOT latency with four
// independent accumulators; this kernel's fold instead runs ONE VFMLA chain
// serially into V20, group after group — the same shape of bottleneck the
// attention A1 campaign found and fixed in causalAttention's QK^T fold
// (MatmulQKAcc64, 8 interleaved chains). This kernel tests that hypothesis
// directly: canonical layout UNCHANGED (isolates the accumulator-chain
// question from item 3's layout question), two independent 4-lane f32
// accumulators (V20, V21) each folding every other group, reduced once at
// the very end. Requires nGroups even (both real FFN shapes — 48, 280 — are;
// a real caller would need an odd-tail path, out of scope for this probe).
TEXT ·dotW4A8FoldSDOT2Acc(SB), NOSPLIT, $0-36
	MOVD act+0(FP), R0
	MOVD packed+8(FP), R1
	MOVD scales+16(FP), R2
	MOVD nGroups+24(FP), R3

	VMOVI $0x0F, V30.B16
	VMOVI $8, V31.B16
	VEOR  V20.B16, V20.B16, V20.B16 // accumulator A
	VEOR  V21.B16, V21.B16, V21.B16 // accumulator B

	LSR $1, R3, R3 // R3 = nGroups/2 (pairs)

pairloop2acc:
	// group A → accumulator V20
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VZIP1  V2.B16, V1.B16, V3.B16
	VZIP2  V2.B16, V1.B16, V4.B16
	VSUB   V31.B16, V3.B16, V3.B16
	VSUB   V31.B16, V4.B16, V4.B16
	VLD1.P 32(R0), [V6.B16, V7.B16]
	VMOVI  $0, V16.B16
	WORD   $0x4E8394D0 // SDOT V16.4S, V6.16B, V3.16B
	WORD   $0x4E8494F0 // SDOT V16.4S, V7.16B, V4.16B
	WORD   $0x4E21DA12 // SCVTF V18.4S, V16.4S
	VLD1R  (R2), [V19.S4]
	VFMLA  V19.S4, V18.S4, V20.S4
	ADD    $4, R2, R2

	// group B → accumulator V21 (independent chain: no register overlap with
	// group A's V16/V18/V19/V20 above, so the two FMLA chains can interleave
	// on an out-of-order core instead of serializing on one another)
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VZIP1  V2.B16, V1.B16, V3.B16
	VZIP2  V2.B16, V1.B16, V4.B16
	VSUB   V31.B16, V3.B16, V3.B16
	VSUB   V31.B16, V4.B16, V4.B16
	VLD1.P 32(R0), [V6.B16, V7.B16]
	VMOVI  $0, V17.B16
	WORD   $0x4E8394D1 // SDOT V17.4S, V6.16B, V3.16B
	WORD   $0x4E8494F1 // SDOT V17.4S, V7.16B, V4.16B
	WORD   $0x4E21DA36 // SCVTF V22.4S, V17.4S
	VLD1R  (R2), [V23.S4]
	VFMLA  V23.S4, V22.S4, V21.S4
	ADD    $4, R2, R2

	SUBS $1, R3, R3
	BNE  pairloop2acc

	WORD  $0x6E34D694 // FADDP V20.4S, V20.4S, V20.4S
	WORD  $0x6E34D694 // → lane0 = sum_A
	FMOVS F20, F9
	WORD  $0x6E35D6B5 // FADDP V21.4S, V21.4S, V21.4S
	WORD  $0x6E35D6B5 // → lane0 = sum_B
	FADDS F21, F9, F9 // F9 = sum_A + sum_B
	FMOVS F9, ret+32(FP)
	RET

// func dotW4A8FoldSDOT4Acc(act *int8, packed *byte, scales *float32, nGroups int) float32
//
// dotW4A8FoldSDOT2Acc's result (1.39-1.47x on the real FFN shape) confirmed
// the serial-fold hypothesis; dotI8SDOT (dot_i8dp_arm64.s) already uses FOUR
// independent accumulators for exactly this reason on this same core, so
// this extends the probe to four lanes to see whether the win keeps scaling
// the way it does there. Canonical layout, signed centering — same
// isolation as the 2-accumulator probe. Requires nGroups a multiple of 4.
TEXT ·dotW4A8FoldSDOT4Acc(SB), NOSPLIT, $0-36
	MOVD act+0(FP), R0
	MOVD packed+8(FP), R1
	MOVD scales+16(FP), R2
	MOVD nGroups+24(FP), R3

	VMOVI $0x0F, V30.B16
	VMOVI $8, V31.B16
	VEOR  V20.B16, V20.B16, V20.B16 // acc A
	VEOR  V21.B16, V21.B16, V21.B16 // acc B
	VEOR  V27.B16, V27.B16, V27.B16 // acc C
	VEOR  V10.B16, V10.B16, V10.B16 // acc D

	LSR $2, R3, R3 // R3 = nGroups/4 (quads)

quadloop4acc:
	// lane A → V20
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VZIP1  V2.B16, V1.B16, V3.B16
	VZIP2  V2.B16, V1.B16, V4.B16
	VSUB   V31.B16, V3.B16, V3.B16
	VSUB   V31.B16, V4.B16, V4.B16
	VLD1.P 32(R0), [V6.B16, V7.B16]
	VMOVI  $0, V16.B16
	WORD   $0x4E8394D0 // SDOT V16.4S, V6.16B, V3.16B
	WORD   $0x4E8494F0 // SDOT V16.4S, V7.16B, V4.16B
	WORD   $0x4E21DA12 // SCVTF V18.4S, V16.4S
	VLD1R  (R2), [V19.S4]
	VFMLA  V19.S4, V18.S4, V20.S4
	ADD    $4, R2, R2

	// lane B → V21
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VZIP1  V2.B16, V1.B16, V3.B16
	VZIP2  V2.B16, V1.B16, V4.B16
	VSUB   V31.B16, V3.B16, V3.B16
	VSUB   V31.B16, V4.B16, V4.B16
	VLD1.P 32(R0), [V6.B16, V7.B16]
	VMOVI  $0, V17.B16
	WORD   $0x4E8394D1 // SDOT V17.4S, V6.16B, V3.16B
	WORD   $0x4E8494F1 // SDOT V17.4S, V7.16B, V4.16B
	WORD   $0x4E21DA36 // SCVTF V22.4S, V17.4S
	VLD1R  (R2), [V23.S4]
	VFMLA  V23.S4, V22.S4, V21.S4
	ADD    $4, R2, R2

	// lane C → V27
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VZIP1  V2.B16, V1.B16, V3.B16
	VZIP2  V2.B16, V1.B16, V4.B16
	VSUB   V31.B16, V3.B16, V3.B16
	VSUB   V31.B16, V4.B16, V4.B16
	VLD1.P 32(R0), [V6.B16, V7.B16]
	VMOVI  $0, V24.B16
	WORD   $0x4E8394D8 // SDOT V24.4S, V6.16B, V3.16B
	WORD   $0x4E8494F8 // SDOT V24.4S, V7.16B, V4.16B
	WORD   $0x4E21DB19 // SCVTF V25.4S, V24.4S
	VLD1R  (R2), [V26.S4]
	VFMLA  V26.S4, V25.S4, V27.S4
	ADD    $4, R2, R2

	// lane D → V10
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VZIP1  V2.B16, V1.B16, V3.B16
	VZIP2  V2.B16, V1.B16, V4.B16
	VSUB   V31.B16, V3.B16, V3.B16
	VSUB   V31.B16, V4.B16, V4.B16
	VLD1.P 32(R0), [V6.B16, V7.B16]
	VMOVI  $0, V28.B16
	WORD   $0x4E8394DC // SDOT V28.4S, V6.16B, V3.16B
	WORD   $0x4E8494FC // SDOT V28.4S, V7.16B, V4.16B
	WORD   $0x4E21DB88 // SCVTF V8.4S, V28.4S
	VLD1R  (R2), [V9.S4]
	VFMLA  V9.S4, V8.S4, V10.S4
	ADD    $4, R2, R2

	SUBS $1, R3, R3
	BNE  quadloop4acc

	WORD  $0x6E34D694 // FADDP V20 → lane0 = sum_A
	WORD  $0x6E34D694
	FMOVS F20, F9
	WORD  $0x6E35D6B5 // FADDP V21 → lane0 = sum_B
	WORD  $0x6E35D6B5
	FADDS F21, F9, F9 // F9 = sum_A + sum_B
	WORD  $0x6E3BD77B // FADDP V27 → lane0 = sum_C
	WORD  $0x6E3BD77B
	FADDS F27, F9, F9 // F9 += sum_C
	WORD  $0x6E2AD54A // FADDP V10 → lane0 = sum_D
	WORD  $0x6E2AD54A
	FADDS F10, F9, F9 // F9 += sum_D
	FMOVS F9, ret+32(FP)
	RET

// func dotW4A8SplitHalf4Row(act *int8, packed4 *byte, scales4 *float32, dst *float32, nGroups int)
//
// docs/prompts/w4a8-item3-harness.md item 4: 4 REAL output rows computed per
// call (unlike dotW4A8SplitHalf4Acc, which splits ONE row's own fold into 4
// artificial lanes) — packed4/scales4 must be repackSplitHalf4RowBlock /
// interleaveScales4Row's interleaved layout (4 rows' data per group,
// contiguous). Per group: the activation chunk is loaded ONCE and reused
// across all 4 rows' SDOT (down from 4 separate reloads of the identical
// bytes in the current per-row-call production path); each row gets its own
// accumulator, so the 4 independent FMLA chains that hide fold latency now
// come from 4 genuine distinct outputs rather than an artificial split of
// one. dst must have room for 4 float32s (one per row). nGroups a multiple
// of 32 is NOT required here (unlike the 2/4-lane single-row kernels) — this
// has no artificial residue of its own, just one group per outer iteration.
TEXT ·dotW4A8SplitHalf4Row(SB), NOSPLIT, $0-40
	MOVD act+0(FP), R0
	MOVD packed4+8(FP), R1
	MOVD scales4+16(FP), R2
	MOVD dst+24(FP), R4
	MOVD nGroups+32(FP), R3

	VMOVI $0x0F, V30.B16
	VMOVI $8, V31.B16
	VEOR  V20.B16, V20.B16, V20.B16 // row0 acc
	VEOR  V21.B16, V21.B16, V21.B16 // row1 acc
	VEOR  V27.B16, V27.B16, V27.B16 // row2 acc
	VEOR  V10.B16, V10.B16, V10.B16 // row3 acc

row4loop:
	VLD1.P 32(R0), [V6.B16, V7.B16] // activation for this group — loaded ONCE

	// row0 → V20
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VSUB   V31.B16, V1.B16, V3.B16
	VSUB   V31.B16, V2.B16, V4.B16
	VMOVI  $0, V16.B16
	WORD   $0x4E8394D0 // SDOT V16.4S, V6.16B, V3.16B
	WORD   $0x4E8494F0 // SDOT V16.4S, V7.16B, V4.16B
	WORD   $0x4E21DA12 // SCVTF V18.4S, V16.4S
	VLD1R  (R2), [V19.S4]
	VFMLA  V19.S4, V18.S4, V20.S4
	ADD    $4, R2, R2

	// row1 → V21
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VSUB   V31.B16, V1.B16, V3.B16
	VSUB   V31.B16, V2.B16, V4.B16
	VMOVI  $0, V17.B16
	WORD   $0x4E8394D1 // SDOT V17.4S, V6.16B, V3.16B
	WORD   $0x4E8494F1 // SDOT V17.4S, V7.16B, V4.16B
	WORD   $0x4E21DA36 // SCVTF V22.4S, V17.4S
	VLD1R  (R2), [V23.S4]
	VFMLA  V23.S4, V22.S4, V21.S4
	ADD    $4, R2, R2

	// row2 → V27
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VSUB   V31.B16, V1.B16, V3.B16
	VSUB   V31.B16, V2.B16, V4.B16
	VMOVI  $0, V24.B16
	WORD   $0x4E8394D8 // SDOT V24.4S, V6.16B, V3.16B
	WORD   $0x4E8494F8 // SDOT V24.4S, V7.16B, V4.16B
	WORD   $0x4E21DB19 // SCVTF V25.4S, V24.4S
	VLD1R  (R2), [V26.S4]
	VFMLA  V26.S4, V25.S4, V27.S4
	ADD    $4, R2, R2

	// row3 → V10
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VSUB   V31.B16, V1.B16, V3.B16
	VSUB   V31.B16, V2.B16, V4.B16
	VMOVI  $0, V28.B16
	WORD   $0x4E8394DC // SDOT V28.4S, V6.16B, V3.16B
	WORD   $0x4E8494FC // SDOT V28.4S, V7.16B, V4.16B
	WORD   $0x4E21DB88 // SCVTF V8.4S, V28.4S
	VLD1R  (R2), [V9.S4]
	VFMLA  V9.S4, V8.S4, V10.S4
	ADD    $4, R2, R2

	SUBS $1, R3, R3
	BNE  row4loop

	WORD  $0x6E34D694 // FADDP V20 → lane0 = row0 sum
	WORD  $0x6E34D694
	FMOVS F20, (R4)
	WORD  $0x6E35D6B5 // FADDP V21 → lane0 = row1 sum
	WORD  $0x6E35D6B5
	FMOVS F21, 4(R4)
	WORD  $0x6E3BD77B // FADDP V27 → lane0 = row2 sum
	WORD  $0x6E3BD77B
	FMOVS F27, 8(R4)
	WORD  $0x6E2AD54A // FADDP V10 → lane0 = row3 sum
	WORD  $0x6E2AD54A
	FMOVS F10, 12(R4)
	RET

// func dotW4A8SplitHalf2Acc(act *int8, packed *byte, scales *float32, nGroups int) float32
//
// Combines the two confirmed-real levers: dotW4A8FoldSDOT2Acc's two
// independent accumulator chains (the actual bottleneck fix) with the
// split-half layout (repackSplitHalfRow) so each lane's unpack drops
// VZIP1/VZIP2, same as dotW4A8SplitHalfSDOT. packed must be split-half
// layout. nGroups must be even. Tests whether the layout saving — a null
// result on its own (dotW4A8SplitHalfSDOT measured a flat 1.000x) — compounds
// once the accumulator-chain bottleneck that was masking it is fixed.
TEXT ·dotW4A8SplitHalf2Acc(SB), NOSPLIT, $0-36
	MOVD act+0(FP), R0
	MOVD packed+8(FP), R1
	MOVD scales+16(FP), R2
	MOVD nGroups+24(FP), R3

	VMOVI $0x0F, V30.B16
	VMOVI $8, V31.B16
	VEOR  V20.B16, V20.B16, V20.B16 // acc A
	VEOR  V21.B16, V21.B16, V21.B16 // acc B

	LSR $1, R3, R3 // R3 = nGroups/2 (pairs)

splithalfpairloop:
	// lane A → V20 (split-half unpack: AND+SHR only, no ZIP)
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16 // block0, already sequential
	VUSHR  $4, V0.B16, V2.B16      // block1, already sequential
	VSUB   V31.B16, V1.B16, V3.B16 // center block0
	VSUB   V31.B16, V2.B16, V4.B16 // center block1
	VLD1.P 32(R0), [V6.B16, V7.B16]
	VMOVI  $0, V16.B16
	WORD   $0x4E8394D0 // SDOT V16.4S, V6.16B, V3.16B
	WORD   $0x4E8494F0 // SDOT V16.4S, V7.16B, V4.16B
	WORD   $0x4E21DA12 // SCVTF V18.4S, V16.4S
	VLD1R  (R2), [V19.S4]
	VFMLA  V19.S4, V18.S4, V20.S4
	ADD    $4, R2, R2

	// lane B → V21
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VSUB   V31.B16, V1.B16, V3.B16
	VSUB   V31.B16, V2.B16, V4.B16
	VLD1.P 32(R0), [V6.B16, V7.B16]
	VMOVI  $0, V17.B16
	WORD   $0x4E8394D1 // SDOT V17.4S, V6.16B, V3.16B
	WORD   $0x4E8494F1 // SDOT V17.4S, V7.16B, V4.16B
	WORD   $0x4E21DA36 // SCVTF V22.4S, V17.4S
	VLD1R  (R2), [V23.S4]
	VFMLA  V23.S4, V22.S4, V21.S4
	ADD    $4, R2, R2

	SUBS $1, R3, R3
	BNE  splithalfpairloop

	WORD  $0x6E34D694 // FADDP V20 → lane0 = sum_A
	WORD  $0x6E34D694
	FMOVS F20, F9
	WORD  $0x6E35D6B5 // FADDP V21 → lane0 = sum_B
	WORD  $0x6E35D6B5
	FADDS F21, F9, F9 // F9 = sum_A + sum_B
	FMOVS F9, ret+32(FP)
	RET

// func dotW4A8SplitHalf4RowPrefetch(act *int8, packed4 *byte, scales4 *float32, dst *float32, nGroups int, prefetchDistance int)
//
// docs/task-w4a8-neon-bandwidth.md's cold-fix harness pass (goinfer's
// docs/task-zeno-compare.md confirmed the mechanism: dotW4A8SplitHalf4Row's 4
// simultaneous accumulator chains share each cold cache-line region, so one
// miss stalls 4 rows' in-flight work at once). Identical to
// dotW4A8SplitHalf4Row byte-for-byte except for one added PRFM PLDL1KEEP per
// outer iteration, issued prefetchDistance BYTES ahead of the current group's
// packed4 pointer — a pure hint (never faults, changes no register any SDOT
// or FMLA reads), so this is bit-identical to dotW4A8SplitHalf4Row by
// construction, not by a separate proof. prefetchDistance is a harness knob
// (2/4/8 cache lines, one page-crossing distance — see the campaign doc for
// the actual sweep points), not a compile-time constant, so one kernel
// covers the whole sweep instead of one hand-copy per distance.
//
// PRFM (immediate) has no Go mnemonic → raw WORD, PLDL1KEEP (Rt=0b00000),
// register-computed address (imm12=0) so the harness-variable distance never
// needs encoding into the immediate field itself: PRFM [Rn] = 0xF9800000 |
// (Rn<<5). Here Rn=R9 (the precomputed prefetch address) → 0xF9800120.
TEXT ·dotW4A8SplitHalf4RowPrefetch(SB), NOSPLIT, $0-48
	MOVD act+0(FP), R0
	MOVD packed4+8(FP), R1
	MOVD scales4+16(FP), R2
	MOVD dst+24(FP), R4
	MOVD nGroups+32(FP), R3
	MOVD prefetchDistance+40(FP), R14

	VMOVI $0x0F, V30.B16
	VMOVI $8, V31.B16
	VEOR  V20.B16, V20.B16, V20.B16 // row0 acc
	VEOR  V21.B16, V21.B16, V21.B16 // row1 acc
	VEOR  V27.B16, V27.B16, V27.B16 // row2 acc
	VEOR  V10.B16, V10.B16, V10.B16 // row3 acc

row4prefetchloop:
	ADD  R14, R1, R9  // prefetch address = this group's packed4 ptr + distance
	WORD $0xF9800120  // PRFM PLDL1KEEP, [R9]

	VLD1.P 32(R0), [V6.B16, V7.B16] // activation for this group — loaded ONCE

	// row0 → V20
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VSUB   V31.B16, V1.B16, V3.B16
	VSUB   V31.B16, V2.B16, V4.B16
	VMOVI  $0, V16.B16
	WORD   $0x4E8394D0 // SDOT V16.4S, V6.16B, V3.16B
	WORD   $0x4E8494F0 // SDOT V16.4S, V7.16B, V4.16B
	WORD   $0x4E21DA12 // SCVTF V18.4S, V16.4S
	VLD1R  (R2), [V19.S4]
	VFMLA  V19.S4, V18.S4, V20.S4
	ADD    $4, R2, R2

	// row1 → V21
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VSUB   V31.B16, V1.B16, V3.B16
	VSUB   V31.B16, V2.B16, V4.B16
	VMOVI  $0, V17.B16
	WORD   $0x4E8394D1 // SDOT V17.4S, V6.16B, V3.16B
	WORD   $0x4E8494F1 // SDOT V17.4S, V7.16B, V4.16B
	WORD   $0x4E21DA36 // SCVTF V22.4S, V17.4S
	VLD1R  (R2), [V23.S4]
	VFMLA  V23.S4, V22.S4, V21.S4
	ADD    $4, R2, R2

	// row2 → V27
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VSUB   V31.B16, V1.B16, V3.B16
	VSUB   V31.B16, V2.B16, V4.B16
	VMOVI  $0, V24.B16
	WORD   $0x4E8394D8 // SDOT V24.4S, V6.16B, V3.16B
	WORD   $0x4E8494F8 // SDOT V24.4S, V7.16B, V4.16B
	WORD   $0x4E21DB19 // SCVTF V25.4S, V24.4S
	VLD1R  (R2), [V26.S4]
	VFMLA  V26.S4, V25.S4, V27.S4
	ADD    $4, R2, R2

	// row3 → V10
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VSUB   V31.B16, V1.B16, V3.B16
	VSUB   V31.B16, V2.B16, V4.B16
	VMOVI  $0, V28.B16
	WORD   $0x4E8394DC // SDOT V28.4S, V6.16B, V3.16B
	WORD   $0x4E8494FC // SDOT V28.4S, V7.16B, V4.16B
	WORD   $0x4E21DB88 // SCVTF V8.4S, V28.4S
	VLD1R  (R2), [V9.S4]
	VFMLA  V9.S4, V8.S4, V10.S4
	ADD    $4, R2, R2

	SUBS $1, R3, R3
	BNE  row4prefetchloop

	WORD  $0x6E34D694 // FADDP V20 → lane0 = row0 sum
	WORD  $0x6E34D694
	FMOVS F20, (R4)
	WORD  $0x6E35D6B5 // FADDP V21 → lane0 = row1 sum
	WORD  $0x6E35D6B5
	FMOVS F21, 4(R4)
	WORD  $0x6E3BD77B // FADDP V27 → lane0 = row2 sum
	WORD  $0x6E3BD77B
	FMOVS F27, 8(R4)
	WORD  $0x6E2AD54A // FADDP V10 → lane0 = row3 sum
	WORD  $0x6E2AD54A
	FMOVS F10, 12(R4)
	RET

// func dotW4A8SplitHalf4RowDeshared(act *int8, packed0, packed1, packed2, packed3 *byte, scales0, scales1, scales2, scales3 *float32, dst *float32, nGroups int)
//
// docs/task-w4a8-neon-bandwidth.md's cold-fix harness pass, second remedy:
// instead of interleaving the 4 rows' split-half bytes contiguously
// (repackSplitHalf4RowBlock — the production row4 layout, which is exactly
// what puts all 4 chains' data on the SAME cold cache line), keep each row's
// split-half bytes and scales in 4 SEPARATE slices (each in
// repackSplitHalfRow's plain per-row layout — no new packing scheme, just no
// final interleave step). The activation is still loaded ONCE per group and
// reused across all 4 SDOTs (the actual warm-path win), but each row's own
// memory now lives in a different Go allocation, de-sharing the cache line 4
// concurrent chains would otherwise contend on a cold miss. Per-row math is
// byte-for-byte identical to dotW4A8SplitHalf4Row's (same
// AND/USHR/SUB/SDOT/SCVTF/FMLA sequence per row) — only which pointer each
// row reads from differs, so this is bit-identical by construction.
TEXT ·dotW4A8SplitHalf4RowDeshared(SB), NOSPLIT, $0-88
	MOVD act+0(FP), R0
	MOVD packed0+8(FP), R1
	MOVD packed1+16(FP), R5
	MOVD packed2+24(FP), R6
	MOVD packed3+32(FP), R7
	MOVD scales0+40(FP), R2
	MOVD scales1+48(FP), R8
	MOVD scales2+56(FP), R9
	MOVD scales3+64(FP), R10
	MOVD dst+72(FP), R4
	MOVD nGroups+80(FP), R3

	VMOVI $0x0F, V30.B16
	VMOVI $8, V31.B16
	VEOR  V20.B16, V20.B16, V20.B16 // row0 acc
	VEOR  V21.B16, V21.B16, V21.B16 // row1 acc
	VEOR  V27.B16, V27.B16, V27.B16 // row2 acc
	VEOR  V10.B16, V10.B16, V10.B16 // row3 acc

desharedloop:
	VLD1.P 32(R0), [V6.B16, V7.B16] // activation for this group — loaded ONCE

	// row0 → V20 (from its own separate packed0/scales0 stream)
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VSUB   V31.B16, V1.B16, V3.B16
	VSUB   V31.B16, V2.B16, V4.B16
	VMOVI  $0, V16.B16
	WORD   $0x4E8394D0 // SDOT V16.4S, V6.16B, V3.16B
	WORD   $0x4E8494F0 // SDOT V16.4S, V7.16B, V4.16B
	WORD   $0x4E21DA12 // SCVTF V18.4S, V16.4S
	VLD1R  (R2), [V19.S4]
	VFMLA  V19.S4, V18.S4, V20.S4
	ADD    $4, R2, R2

	// row1 → V21 (from packed1/scales1)
	VLD1.P 16(R5), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VSUB   V31.B16, V1.B16, V3.B16
	VSUB   V31.B16, V2.B16, V4.B16
	VMOVI  $0, V17.B16
	WORD   $0x4E8394D1 // SDOT V17.4S, V6.16B, V3.16B
	WORD   $0x4E8494F1 // SDOT V17.4S, V7.16B, V4.16B
	WORD   $0x4E21DA36 // SCVTF V22.4S, V17.4S
	VLD1R  (R8), [V23.S4]
	VFMLA  V23.S4, V22.S4, V21.S4
	ADD    $4, R8, R8

	// row2 → V27 (from packed2/scales2)
	VLD1.P 16(R6), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VSUB   V31.B16, V1.B16, V3.B16
	VSUB   V31.B16, V2.B16, V4.B16
	VMOVI  $0, V24.B16
	WORD   $0x4E8394D8 // SDOT V24.4S, V6.16B, V3.16B
	WORD   $0x4E8494F8 // SDOT V24.4S, V7.16B, V4.16B
	WORD   $0x4E21DB19 // SCVTF V25.4S, V24.4S
	VLD1R  (R9), [V26.S4]
	VFMLA  V26.S4, V25.S4, V27.S4
	ADD    $4, R9, R9

	// row3 → V10 (from packed3/scales3)
	VLD1.P 16(R7), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VSUB   V31.B16, V1.B16, V3.B16
	VSUB   V31.B16, V2.B16, V4.B16
	VMOVI  $0, V28.B16
	WORD   $0x4E8394DC // SDOT V28.4S, V6.16B, V3.16B
	WORD   $0x4E8494FC // SDOT V28.4S, V7.16B, V4.16B
	WORD   $0x4E21DB88 // SCVTF V8.4S, V28.4S
	VLD1R  (R10), [V9.S4]
	VFMLA  V9.S4, V8.S4, V10.S4
	ADD    $4, R10, R10

	SUBS $1, R3, R3
	BNE  desharedloop

	WORD  $0x6E34D694 // FADDP V20 → lane0 = row0 sum
	WORD  $0x6E34D694
	FMOVS F20, (R4)
	WORD  $0x6E35D6B5 // FADDP V21 → lane0 = row1 sum
	WORD  $0x6E35D6B5
	FMOVS F21, 4(R4)
	WORD  $0x6E3BD77B // FADDP V27 → lane0 = row2 sum
	WORD  $0x6E3BD77B
	FMOVS F27, 8(R4)
	WORD  $0x6E2AD54A // FADDP V10 → lane0 = row3 sum
	WORD  $0x6E2AD54A
	FMOVS F10, 12(R4)
	RET

// func dotW4A8SplitHalf4Acc(act *int8, packed *byte, scales *float32, nGroups int) float32
//
// dotW4A8SplitHalf2Acc extended to four lanes, matching dotW4A8FoldSDOT4Acc's
// register assignment (lanes C/D: SDOT dst V24/V28, SCVTF dst V25/V8, scale
// V26/V9, acc V27/V10) but with the split-half unpack (AND+SHR only, no ZIP)
// in every lane. packed must be split-half layout. nGroups must be a
// multiple of 4. Tests whether the layout+accumulator compounding seen at 2
// lanes (dotW4A8SplitHalf2Acc, 1.6-1.75x) continues to 4, given
// dotW4A8FoldSDOT4Acc alone (canonical layout) showed no further gain over
// 2 lanes — this checks whether that saturation point moves once the
// unpack prologue is also shorter.
TEXT ·dotW4A8SplitHalf4Acc(SB), NOSPLIT, $0-36
	MOVD act+0(FP), R0
	MOVD packed+8(FP), R1
	MOVD scales+16(FP), R2
	MOVD nGroups+24(FP), R3

	VMOVI $0x0F, V30.B16
	VMOVI $8, V31.B16
	VEOR  V20.B16, V20.B16, V20.B16 // acc A
	VEOR  V21.B16, V21.B16, V21.B16 // acc B
	VEOR  V27.B16, V27.B16, V27.B16 // acc C
	VEOR  V10.B16, V10.B16, V10.B16 // acc D

	LSR $2, R3, R3 // R3 = nGroups/4 (quads)

splithalfquadloop4acc:
	// lane A → V20 (split-half unpack: AND+SHR only, no ZIP)
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VSUB   V31.B16, V1.B16, V3.B16
	VSUB   V31.B16, V2.B16, V4.B16
	VLD1.P 32(R0), [V6.B16, V7.B16]
	VMOVI  $0, V16.B16
	WORD   $0x4E8394D0 // SDOT V16.4S, V6.16B, V3.16B
	WORD   $0x4E8494F0 // SDOT V16.4S, V7.16B, V4.16B
	WORD   $0x4E21DA12 // SCVTF V18.4S, V16.4S
	VLD1R  (R2), [V19.S4]
	VFMLA  V19.S4, V18.S4, V20.S4
	ADD    $4, R2, R2

	// lane B → V21
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VSUB   V31.B16, V1.B16, V3.B16
	VSUB   V31.B16, V2.B16, V4.B16
	VLD1.P 32(R0), [V6.B16, V7.B16]
	VMOVI  $0, V17.B16
	WORD   $0x4E8394D1 // SDOT V17.4S, V6.16B, V3.16B
	WORD   $0x4E8494F1 // SDOT V17.4S, V7.16B, V4.16B
	WORD   $0x4E21DA36 // SCVTF V22.4S, V17.4S
	VLD1R  (R2), [V23.S4]
	VFMLA  V23.S4, V22.S4, V21.S4
	ADD    $4, R2, R2

	// lane C → V27
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VSUB   V31.B16, V1.B16, V3.B16
	VSUB   V31.B16, V2.B16, V4.B16
	VLD1.P 32(R0), [V6.B16, V7.B16]
	VMOVI  $0, V24.B16
	WORD   $0x4E8394D8 // SDOT V24.4S, V6.16B, V3.16B
	WORD   $0x4E8494F8 // SDOT V24.4S, V7.16B, V4.16B
	WORD   $0x4E21DB19 // SCVTF V25.4S, V24.4S
	VLD1R  (R2), [V26.S4]
	VFMLA  V26.S4, V25.S4, V27.S4
	ADD    $4, R2, R2

	// lane D → V10
	VLD1.P 16(R1), [V0.B16]
	VAND   V30.B16, V0.B16, V1.B16
	VUSHR  $4, V0.B16, V2.B16
	VSUB   V31.B16, V1.B16, V3.B16
	VSUB   V31.B16, V2.B16, V4.B16
	VLD1.P 32(R0), [V6.B16, V7.B16]
	VMOVI  $0, V28.B16
	WORD   $0x4E8394DC // SDOT V28.4S, V6.16B, V3.16B
	WORD   $0x4E8494FC // SDOT V28.4S, V7.16B, V4.16B
	WORD   $0x4E21DB88 // SCVTF V8.4S, V28.4S
	VLD1R  (R2), [V9.S4]
	VFMLA  V9.S4, V8.S4, V10.S4
	ADD    $4, R2, R2

	SUBS $1, R3, R3
	BNE  splithalfquadloop4acc

	WORD  $0x6E34D694 // FADDP V20 → lane0 = sum_A
	WORD  $0x6E34D694
	FMOVS F20, F9
	WORD  $0x6E35D6B5 // FADDP V21 → lane0 = sum_B
	WORD  $0x6E35D6B5
	FADDS F21, F9, F9 // F9 = sum_A + sum_B
	WORD  $0x6E3BD77B // FADDP V27 → lane0 = sum_C
	WORD  $0x6E3BD77B
	FADDS F27, F9, F9 // F9 += sum_C
	WORD  $0x6E2AD54A // FADDP V10 → lane0 = sum_D
	WORD  $0x6E2AD54A
	FADDS F10, F9, F9 // F9 += sum_D
	FMOVS F9, ret+32(FP)
	RET
