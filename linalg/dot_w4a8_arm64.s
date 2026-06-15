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
// after detectDotProd() (HWCAP_ASIMDDP), so SDOT never traps.

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
