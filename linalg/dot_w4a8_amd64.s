// Fused int4(weight)×int8(activation) GEMV dot using AVX2 — the amd64 W4A8
// decode kernel's hot loop, counterpart of dot_w4a8_arm64.s. For ONE output it
// streams a whole K-wide weight row and returns the per-group-scaled f32 dot
// Σ_g scale[g]·(act·w)_g (the activation scale is applied by the Go caller).
//
// The nibble-unpack prologue (16 packed bytes → 32 centered int8 weights:
// low nibble = even k, high = odd k, −8) feeds the proven dotI8AVX2 sign-extend
// body (VPMOVSXBW + VPMADDWD). The KEY over the older per-group variant: the
// f32 weight scale is folded IN-REGISTER. Each group's 8 int32 lane-partials are
// converted to f32, multiplied by the group's broadcast scale, and accumulated
// into an 8-lane f32 accumulator — WITHOUT a per-group horizontal reduce. Because
// every lane of a group carries the same scale[g], one final reduce of the
// accumulator yields Σ_g scale[g]·Σ_lane(partials) = Σ_g scale[g]·groupdot[g].
// This removes the 63/64 per-group reductions, the SIMD→GP moves, the int32
// scratch round-trip, and the Go-side fold loop the old kernel paid — the
// overhead that made W4A8 ~2× slower than W8A8 despite reading half the bytes.
//
// group is fixed at 32 (16 packed bytes / 32 activations per iteration); the Go
// caller routes other group sizes and any ragged tail (K % 32) to the scalar
// reference. nGroups = K/32. AVX2 baseline (no VNNI); only called when hasAVX2.

#include "textflag.h"

DATA mask0F<>+0(SB)/8, $0x0F0F0F0F0F0F0F0F
DATA mask0F<>+8(SB)/8, $0x0F0F0F0F0F0F0F0F
GLOBL mask0F<>(SB), RODATA|NOPTR, $16

DATA const8<>+0(SB)/8, $0x0808080808080808
DATA const8<>+8(SB)/8, $0x0808080808080808
GLOBL const8<>(SB), RODATA|NOPTR, $16

// func dotW4A8FoldAVX2(act *int8, packed *byte, scales *float32, nGroups int) float32
TEXT ·dotW4A8FoldAVX2(SB), NOSPLIT, $0-36
	MOVQ act+0(FP), SI     // &act[0]    (int8, 32 per group)
	MOVQ packed+8(FP), DI  // &packed[0] (16 bytes per group)
	MOVQ scales+16(FP), BX // &scales[0] (f32, one per group)
	MOVQ nGroups+24(FP), CX

	LEAQ    mask0F<>(SB), AX
	VMOVDQU (AX), X14       // low-nibble mask (hoisted)
	LEAQ    const8<>(SB), AX
	VMOVDQU (AX), X15       // bias 8 (hoisted)
	VXORPS  Y10, Y10, Y10   // f32 accumulator (8 lanes) = 0

loop:
	// Unpack 16 packed bytes → w0..w15 (X3) and w16..w31 (X4), centered −8.
	VMOVDQU    (DI), X0
	VPAND      X14, X0, X1       // low nibbles  (w0,w2,…,w30)
	VPSRLW     $4, X0, X2
	VPAND      X14, X2, X2       // high nibbles (w1,w3,…,w31)
	VPUNPCKLBW X2, X1, X3        // [w0,w1,…,w15]
	VPUNPCKHBW X2, X1, X4        // [w16,…,w31]
	VPSUBB     X15, X3, X3       // centered: nibble − 8 (signed int8)
	VPSUBB     X15, X4, X4
	VPMOVSXBW  X3, Y3
	VPMOVSXBW  X4, Y4

	// 32 int8 activations, sign-extended.
	VMOVDQU   (SI), X5
	VMOVDQU   16(SI), X6
	VPMOVSXBW X5, Y5
	VPMOVSXBW X6, Y6

	// Pairwise multiply-add → 8 int32 group partials (UNREDUCED).
	VPMADDWD Y5, Y3, Y7
	VPMADDWD Y6, Y4, Y8
	VPADDD   Y8, Y7, Y7

	// In-register f32 fold: convert lanes, broadcast scale[g], FMA into the
	// accumulator. No per-group horizontal reduce — each lane carries scale[g].
	VCVTDQ2PS    Y7, Y9
	VBROADCASTSS (BX), Y11
	VFMADD231PS  Y11, Y9, Y10     // Y10 += Y9 · scale[g]

	ADDQ $16, DI
	ADDQ $32, SI
	ADDQ $4, BX
	SUBQ $1, CX
	JNZ  loop

	// One horizontal f32 reduce of the 8-lane accumulator → return value.
	VEXTRACTF128 $1, Y10, X11
	VADDPS       X11, X10, X10
	VHADDPS      X10, X10, X10
	VHADDPS      X10, X10, X10
	MOVSS        X10, ret+32(FP)
	VZEROUPPER
	RET

// func dotW4A8Fold2AccAVX2(act *int8, packed *byte, scales *float32, nGroups int) float32
//
// dotW4A8FoldAVX2 with the f32 fold split across TWO independent accumulator chains.
//
// WHY. dotW4A8FoldAVX2's inner loop ends in `VFMADD231PS Y11, Y9, Y10`, and every
// iteration's FMA reads the previous iteration's Y10 — one serial dependency chain
// across the whole row. On Zen 2 an FMA is ~5 cycles of latency at 2/cycle throughput,
// so a single chain bounds the loop at ~5 cycles per 32-MAC group no matter how many
// issue slots are free. TestW4A8IssueWidthProbe measures exactly that signature: the
// cold kernel is NOT issue-limited — idle slots exist — which is the shape of a
// latency-bound chain rather than a throughput-bound one.
//
// The same change on arm64 (dotW4A8FoldSDOT2Acc) measured a real 1.4-1.75x, and that
// case is on record precisely because a "not issue-limited" reading would have argued
// against trying it. Two chains halve the dependent latency per group; the unpack and
// VPMADDWD work is unchanged and simply fills the slots that were idle.
//
// nGroups MUST BE EVEN — the caller routes odd counts elsewhere. Same operand contract
// as dotW4A8FoldAVX2 otherwise, and bit-identical to it only in exact arithmetic: the
// two accumulators sum groups in a different order, so f32 rounding may differ in the
// last ulp. Callers that need the original's exact bits must keep calling it.
//
// MEASURED NEGATIVE, 2026-08-31, Ryzen 7 3700X, K=5120, order-alternated, two passes:
//
//	1Acc  17.26 / 17.38 GMAC/s
//	2Acc  17.45 / 17.41 GMAC/s     ~0.5%, inside noise
//
// So the serial fold is NOT this kernel's ceiling, unlike arm64's, and the hypothesis
// this function was built to test is refuted. Kept rather than deleted, matching the
// arm64 2Acc/4Acc probes and this repo's "a documented negative is worth something, a
// silently reverted one is not" convention — and because a future 128-bit or
// shorter-prologue variant would want to re-ask the question against a different
// instruction budget.
//
// LEADING EXPLANATION, UNVERIFIED (no uop counters were read). The loop body is ~20
// vector instructions per 32-MAC group, and Zen 2 cracks every 256-bit AVX2 op into two
// 128-bit uops — so ~40 uops/group, a 6.7-10 cycle/group floor at 4-6 uops/cycle. The
// measurement is 1.854 ns/group = 6.7-8.2 cycles depending on clock, i.e. already at
// that floor. Breaking a 5-cycle FMA chain frees nothing when uop throughput is the
// binding constraint. arm64 NEON is 128-bit natively and pays no such split, which is
// also the leading explanation for its 1.47x advantage on the same algorithm.
//
// WHAT THAT IMPLIES FOR THE NEXT ATTEMPT: the lever is fewer instructions per group —
// the unpack prologue — not chain depth. That is what the split-half repack
// (dotW4A8SplitHalfSDOT, and goinfer's task-w4a8-neon-bandwidth.md item 3) attacks, and
// this result is independent evidence for the same conclusion that page reached on
// arm64 by a different route.
TEXT ·dotW4A8Fold2AccAVX2(SB), NOSPLIT, $0-36
	MOVQ act+0(FP), SI
	MOVQ packed+8(FP), DI
	MOVQ scales+16(FP), BX
	MOVQ nGroups+24(FP), CX
	SHRQ $1, CX             // two groups per iteration

	LEAQ    mask0F<>(SB), AX
	VMOVDQU (AX), X14
	LEAQ    const8<>(SB), AX
	VMOVDQU (AX), X15
	VXORPS  Y10, Y10, Y10   // accumulator A
	VXORPS  Y12, Y12, Y12   // accumulator B

loop2:
	// ---- group A → Y10
	VMOVDQU    (DI), X0
	VPAND      X14, X0, X1
	VPSRLW     $4, X0, X2
	VPAND      X14, X2, X2
	VPUNPCKLBW X2, X1, X3
	VPUNPCKHBW X2, X1, X4
	VPSUBB     X15, X3, X3
	VPSUBB     X15, X4, X4
	VPMOVSXBW  X3, Y3
	VPMOVSXBW  X4, Y4

	VMOVDQU   (SI), X5
	VMOVDQU   16(SI), X6
	VPMOVSXBW X5, Y5
	VPMOVSXBW X6, Y6

	VPMADDWD Y5, Y3, Y7
	VPMADDWD Y6, Y4, Y8
	VPADDD   Y8, Y7, Y7

	VCVTDQ2PS    Y7, Y9
	VBROADCASTSS (BX), Y11
	VFMADD231PS  Y11, Y9, Y10

	// ---- group B → Y12 (independent chain)
	VMOVDQU    16(DI), X0
	VPAND      X14, X0, X1
	VPSRLW     $4, X0, X2
	VPAND      X14, X2, X2
	VPUNPCKLBW X2, X1, X3
	VPUNPCKHBW X2, X1, X4
	VPSUBB     X15, X3, X3
	VPSUBB     X15, X4, X4
	VPMOVSXBW  X3, Y3
	VPMOVSXBW  X4, Y4

	VMOVDQU   32(SI), X5
	VMOVDQU   48(SI), X6
	VPMOVSXBW X5, Y5
	VPMOVSXBW X6, Y6

	VPMADDWD Y5, Y3, Y7
	VPMADDWD Y6, Y4, Y8
	VPADDD   Y8, Y7, Y7

	VCVTDQ2PS    Y7, Y9
	VBROADCASTSS 4(BX), Y11
	VFMADD231PS  Y11, Y9, Y12

	ADDQ $32, DI
	ADDQ $64, SI
	ADDQ $8, BX
	SUBQ $1, CX
	JNZ  loop2

	// Join the two chains, then the same single horizontal reduce.
	VADDPS       Y12, Y10, Y10
	VEXTRACTF128 $1, Y10, X11
	VADDPS       X11, X10, X10
	VHADDPS      X10, X10, X10
	VHADDPS      X10, X10, X10
	MOVSS        X10, ret+32(FP)
	VZEROUPPER
	RET

// func dotW4A8SplitHalfAVX2(act *int8, packed *byte, scales *float32, nGroups int) float32
//
// dotW4A8FoldAVX2 over the SPLIT-HALF weight layout: byte i of a group holds weight i in its
// low nibble and weight i+16 in its high nibble, so one 16-byte load yields two contiguous
// 16-weight halves with no interleave to undo. Against the canonical layout that deletes the
// two VPUNPCKLBW/VPUNPCKHBW per group and nothing else — 8 shuffle/logic prologue ops become 6.
//
// WHY THIS AND NOT MORE ACCUMULATORS. Two accumulator attempts on this kernel are recorded
// negatives: dotW4A8Fold4AVX2 (perf-dead-ends.md §8.9, ~1%) and dotW4A8Fold2AccAVX2 above
// (~0.5%). §8.9's own explanation is that the marginal-FMA probe's "not issue-limited" reading
// only means idle capacity on the FMA ports, and that the nibble-unpack prologue is "exactly the
// kind of work that would bottleneck a shuffle port while leaving FMA ports idle". VPUNPCKLBW and
// VPUNPCKHBW are shuffle-port instructions. If that diagnosis is right, deleting two of them is
// the lever that fits this kernel's actual bottleneck — where deeper accumulator chains, twice
// measured, are not.
//
// arm64 is the counter-case and explains why this was not obviously worth trying: there,
// dotW4A8SplitHalfSDOT measured a FLAT 1.000x alone, because that kernel is latency-bound on its
// FMLA chain and shortening the prologue changed nothing until 2Acc removed the stall. AVX2 is
// diagnosed as the opposite — port-bound, not latency-bound — so the same change should behave
// differently here. That is the prediction; the A/B is the arbiter.
//
// LAYOUT IS THE CALLER'S PROBLEM. packed must already be split-half (harness repack). The
// canonical packer, the .giw on-disk format and dotW4A8Scalar are untouched — the same discipline
// arm64's repack follows, and the reason the .giw kind=3 zero-copy load path cannot be disturbed.
TEXT ·dotW4A8SplitHalfAVX2(SB), NOSPLIT, $0-36
	MOVQ act+0(FP), SI
	MOVQ packed+8(FP), DI
	MOVQ scales+16(FP), BX
	MOVQ nGroups+24(FP), CX

	LEAQ    mask0F<>(SB), AX
	VMOVDQU (AX), X14
	LEAQ    const8<>(SB), AX
	VMOVDQU (AX), X15
	VXORPS  Y10, Y10, Y10

loopsh:
	// Split-half unpack: low nibbles ARE w0..w15, high nibbles ARE w16..w31.
	// No VPUNPCK — that is the whole change.
	VMOVDQU   (DI), X0
	VPAND     X14, X0, X1
	VPSRLW    $4, X0, X2
	VPAND     X14, X2, X2
	VPSUBB    X15, X1, X1
	VPSUBB    X15, X2, X2
	VPMOVSXBW X1, Y3
	VPMOVSXBW X2, Y4

	VMOVDQU   (SI), X5
	VMOVDQU   16(SI), X6
	VPMOVSXBW X5, Y5
	VPMOVSXBW X6, Y6

	VPMADDWD Y5, Y3, Y7
	VPMADDWD Y6, Y4, Y8
	VPADDD   Y8, Y7, Y7

	VCVTDQ2PS    Y7, Y9
	VBROADCASTSS (BX), Y11
	VFMADD231PS  Y11, Y9, Y10

	ADDQ $16, DI
	ADDQ $32, SI
	ADDQ $4, BX
	SUBQ $1, CX
	JNZ  loopsh

	VEXTRACTF128 $1, Y10, X11
	VADDPS       X11, X10, X10
	VHADDPS      X10, X10, X10
	VHADDPS      X10, X10, X10
	MOVSS        X10, ret+32(FP)
	VZEROUPPER
	RET

// func dotW4A8SplitHalfScaledAVX2(act *int8, packed *byte, scales *float32, aScales *float32, nGroups int) float32
//
// dotW4A8SplitHalfAVX2 with a per-group ACTIVATION scale (actgroup.go, per-32 activations): each
// group's int32 partial is multiplied by scales[g]*aScales[g] (one VMULPS, the same f32 product the Go
// side formed before) instead of scales[g] alone. The scale costs two instructions per group; to keep
// that from lengthening a compute-bound small matmul (the 1.5B speed-gate case), the loop does two
// groups per iteration into two independent accumulators (Y10, Y13), summed once at the end, with a
// one-group tail for an odd count.
TEXT ·dotW4A8SplitHalfScaledAVX2(SB), NOSPLIT, $0-44
	MOVQ act+0(FP), SI
	MOVQ packed+8(FP), DI
	MOVQ scales+16(FP), BX
	MOVQ aScales+24(FP), R8
	MOVQ nGroups+32(FP), CX

	LEAQ    mask0F<>(SB), AX
	VMOVDQU (AX), X14
	LEAQ    const8<>(SB), AX
	VMOVDQU (AX), X15
	VXORPS  Y10, Y10, Y10
	VXORPS  Y13, Y13, Y13

	CMPQ CX, $2
	JLT  tailshs

loopshs2:
	// group g -> Y10
	VMOVDQU   (DI), X0
	VPAND     X14, X0, X1
	VPSRLW    $4, X0, X2
	VPAND     X14, X2, X2
	VPSUBB    X15, X1, X1
	VPSUBB    X15, X2, X2
	VPMOVSXBW X1, Y3
	VPMOVSXBW X2, Y4
	VMOVDQU   (SI), X5
	VMOVDQU   16(SI), X6
	VPMOVSXBW X5, Y5
	VPMOVSXBW X6, Y6
	VPMADDWD  Y5, Y3, Y7
	VPMADDWD  Y6, Y4, Y8
	VPADDD    Y8, Y7, Y7
	VCVTDQ2PS    Y7, Y9
	VBROADCASTSS (BX), Y11
	VBROADCASTSS (R8), Y12
	VMULPS       Y12, Y11, Y11
	VFMADD231PS  Y11, Y9, Y10

	// group g+1 -> Y13
	VMOVDQU   16(DI), X0
	VPAND     X14, X0, X1
	VPSRLW    $4, X0, X2
	VPAND     X14, X2, X2
	VPSUBB    X15, X1, X1
	VPSUBB    X15, X2, X2
	VPMOVSXBW X1, Y3
	VPMOVSXBW X2, Y4
	VMOVDQU   32(SI), X5
	VMOVDQU   48(SI), X6
	VPMOVSXBW X5, Y5
	VPMOVSXBW X6, Y6
	VPMADDWD  Y5, Y3, Y7
	VPMADDWD  Y6, Y4, Y8
	VPADDD    Y8, Y7, Y7
	VCVTDQ2PS    Y7, Y9
	VBROADCASTSS 4(BX), Y11
	VBROADCASTSS 4(R8), Y12
	VMULPS       Y12, Y11, Y11
	VFMADD231PS  Y11, Y9, Y13

	ADDQ $32, DI
	ADDQ $64, SI
	ADDQ $8, BX
	ADDQ $8, R8
	SUBQ $2, CX
	CMPQ CX, $2
	JGE  loopshs2

tailshs:
	TESTQ CX, CX
	JZ    donehs
	VMOVDQU   (DI), X0
	VPAND     X14, X0, X1
	VPSRLW    $4, X0, X2
	VPAND     X14, X2, X2
	VPSUBB    X15, X1, X1
	VPSUBB    X15, X2, X2
	VPMOVSXBW X1, Y3
	VPMOVSXBW X2, Y4
	VMOVDQU   (SI), X5
	VMOVDQU   16(SI), X6
	VPMOVSXBW X5, Y5
	VPMOVSXBW X6, Y6
	VPMADDWD  Y5, Y3, Y7
	VPMADDWD  Y6, Y4, Y8
	VPADDD    Y8, Y7, Y7
	VCVTDQ2PS    Y7, Y9
	VBROADCASTSS (BX), Y11
	VBROADCASTSS (R8), Y12
	VMULPS       Y12, Y11, Y11
	VFMADD231PS  Y11, Y9, Y10

donehs:
	VADDPS       Y13, Y10, Y10
	VEXTRACTF128 $1, Y10, X11
	VADDPS       X11, X10, X10
	VHADDPS      X10, X10, X10
	VHADDPS      X10, X10, X10
	MOVSS        X10, ret+40(FP)
	VZEROUPPER
	RET
