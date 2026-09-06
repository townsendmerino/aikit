// NEON f32 exp kernel for the S-06 step-2 numeric contract.
//
// This implements expF32Contract (exp_contract.go) instruction for instruction:
// same algorithm, same constants, same operation ORDER, and every multiply-add a
// true fused FMLA/FMLS. The Go oracle reaches the same correctly-rounded f32 FMA
// through f64 (see fma32 for why that is exact), so the two agree BIT FOR BIT and
// TestExpF32NEON_bitIdenticalToContract asserts raw Float32bits equality rather
// than a tolerance.
//
// Operand order was verified empirically before this was written, because a
// reversed VFSUB or VFMLS is silent: VFSUB Vm,Vn,Vd is Vd = Vn - Vm, VFMLA Vc,Vb,Vd
// is Vd += Vb*Vc, VFMLS Vc,Vb,Vd is Vd -= Vb*Vc, VSSHR is arithmetic.
//
// SCALING IS TWO-STEP AND THAT IS DELIBERATE. The scalar builds 2^k in one shift
// and needs a branch for k=128 (where the exponent field would encode Inf and
// poison a finite answer). Here k is split as k1 = k>>1, k2 = k-k1, and the result
// is p * 2^k1 * 2^k2. For the caller-guaranteed range x in [-87.34, 88.72] we have
// k in [-126,128], so k1,k2 in [-63,64] and both exponent fields are always valid
// normals -- no branch, no Inf encoding. It is EXACT and therefore bit-identical to
// the single-step form, because multiplying by a power of two rounds nothing when
// the intermediate stays normal, and p is in [0.7,1.42] so it does.
//
// Underflow still needs the flush the scalar performs (e<=0 returns 0 rather than a
// subnormal), done branchlessly with a compare-and-mask.

//go:build arm64

#include "textflag.h"

DATA cLog2e<>+0(SB)/4, $0x3fb8aa3b
DATA cLog2e<>+4(SB)/4, $0x3fb8aa3b
DATA cLog2e<>+8(SB)/4, $0x3fb8aa3b
DATA cLog2e<>+12(SB)/4, $0x3fb8aa3b
GLOBL cLog2e<>(SB), RODATA|NOPTR, $16
DATA cMagic<>+0(SB)/4, $0x4b400000
DATA cMagic<>+4(SB)/4, $0x4b400000
DATA cMagic<>+8(SB)/4, $0x4b400000
DATA cMagic<>+12(SB)/4, $0x4b400000
GLOBL cMagic<>(SB), RODATA|NOPTR, $16
DATA cLn2Hi<>+0(SB)/4, $0x3f317200
DATA cLn2Hi<>+4(SB)/4, $0x3f317200
DATA cLn2Hi<>+8(SB)/4, $0x3f317200
DATA cLn2Hi<>+12(SB)/4, $0x3f317200
GLOBL cLn2Hi<>(SB), RODATA|NOPTR, $16
DATA cLn2Lo<>+0(SB)/4, $0x35bfbe8e
DATA cLn2Lo<>+4(SB)/4, $0x35bfbe8e
DATA cLn2Lo<>+8(SB)/4, $0x35bfbe8e
DATA cLn2Lo<>+12(SB)/4, $0x35bfbe8e
GLOBL cLn2Lo<>(SB), RODATA|NOPTR, $16
DATA cP0<>+0(SB)/4, $0x39506967
DATA cP0<>+4(SB)/4, $0x39506967
DATA cP0<>+8(SB)/4, $0x39506967
DATA cP0<>+12(SB)/4, $0x39506967
GLOBL cP0<>(SB), RODATA|NOPTR, $16
DATA cP1<>+0(SB)/4, $0x3ab743ce
DATA cP1<>+4(SB)/4, $0x3ab743ce
DATA cP1<>+8(SB)/4, $0x3ab743ce
DATA cP1<>+12(SB)/4, $0x3ab743ce
GLOBL cP1<>(SB), RODATA|NOPTR, $16
DATA cP2<>+0(SB)/4, $0x3c088908
DATA cP2<>+4(SB)/4, $0x3c088908
DATA cP2<>+8(SB)/4, $0x3c088908
DATA cP2<>+12(SB)/4, $0x3c088908
GLOBL cP2<>(SB), RODATA|NOPTR, $16
DATA cP3<>+0(SB)/4, $0x3d2aa9c1
DATA cP3<>+4(SB)/4, $0x3d2aa9c1
DATA cP3<>+8(SB)/4, $0x3d2aa9c1
DATA cP3<>+12(SB)/4, $0x3d2aa9c1
GLOBL cP3<>(SB), RODATA|NOPTR, $16
DATA cP4<>+0(SB)/4, $0x3e2aaaaa
DATA cP4<>+4(SB)/4, $0x3e2aaaaa
DATA cP4<>+8(SB)/4, $0x3e2aaaaa
DATA cP4<>+12(SB)/4, $0x3e2aaaaa
GLOBL cP4<>(SB), RODATA|NOPTR, $16
DATA cP5<>+0(SB)/4, $0x3f000000
DATA cP5<>+4(SB)/4, $0x3f000000
DATA cP5<>+8(SB)/4, $0x3f000000
DATA cP5<>+12(SB)/4, $0x3f000000
GLOBL cP5<>(SB), RODATA|NOPTR, $16
DATA cOne<>+0(SB)/4, $0x3f800000
DATA cOne<>+4(SB)/4, $0x3f800000
DATA cOne<>+8(SB)/4, $0x3f800000
DATA cOne<>+12(SB)/4, $0x3f800000
GLOBL cOne<>(SB), RODATA|NOPTR, $16

DATA cClamp<>+0(SB)/4, $0xc2d00000
DATA cClamp<>+4(SB)/4, $0xc2d00000
DATA cClamp<>+8(SB)/4, $0xc2d00000
DATA cClamp<>+12(SB)/4, $0xc2d00000
GLOBL cClamp<>(SB), RODATA|NOPTR, $16

// func expF32ContractNEON(dst, src *float32, n int)
// n must be a multiple of 4; the Go caller handles the tail. Inputs must be finite
// and within [expUnderflowF32, expOverflowF32] -- the caller guards, exactly as
// ExpF32 guards expF32Core.
TEXT ·expF32ContractNEON(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD n+16(FP), R2
	CBZ  R2, done

	MOVD $cLog2e<>(SB), R3
	VLD1 (R3), [V16.S4]
	MOVD $cMagic<>(SB), R3
	VLD1 (R3), [V17.S4]
	MOVD $cLn2Hi<>(SB), R3
	VLD1 (R3), [V18.S4]
	MOVD $cLn2Lo<>(SB), R3
	VLD1 (R3), [V19.S4]
	MOVD $cP0<>(SB), R3
	VLD1 (R3), [V20.S4]
	MOVD $cP1<>(SB), R3
	VLD1 (R3), [V21.S4]
	MOVD $cP2<>(SB), R3
	VLD1 (R3), [V22.S4]
	MOVD $cP3<>(SB), R3
	VLD1 (R3), [V23.S4]
	MOVD $cP4<>(SB), R3
	VLD1 (R3), [V24.S4]
	MOVD $cP5<>(SB), R3
	VLD1 (R3), [V25.S4]
	MOVD $cOne<>(SB), R3
	VLD1 (R3), [V26.S4]
	MOVD $127, R4
	VDUP R4, V27.S4          // 127, int32 lanes
	VMOVI $0, V28.B16        // zero

loop:
	VLD1.P 16(R1), [V0.S4]   // x
	VFMUL V16.S4, V0.S4, V1.S4   // z  = x * log2e
	VFADD V17.S4, V1.S4, V1.S4   // t  = z + magic
	VFSUB V17.S4, V1.S4, V1.S4   // kf = t - magic   (round to nearest even)
	VFCVTZS V1.S4, V2.S4         // k  = int32(kf), exact: kf is integral
	VMOV V0.B16, V3.B16          // r  = x
	VFMLS V18.S4, V1.S4, V3.S4   // r -= kf*ln2Hi   (one fused op)
	VFMLS V19.S4, V1.S4, V3.S4   // r -= kf*ln2Lo

	// Horner, alternating destinations so each step is one FMLA into a
	// freshly-seeded coefficient -- the same six steps, same order, as the oracle.
	VMOV V21.B16, V4.B16
	VFMLA V3.S4, V20.S4, V4.S4   // p = c0*r + c1
	VMOV V22.B16, V5.B16
	VFMLA V3.S4, V4.S4, V5.S4    // p = p*r + c2
	VMOV V23.B16, V4.B16
	VFMLA V3.S4, V5.S4, V4.S4    // p = p*r + c3
	VMOV V24.B16, V5.B16
	VFMLA V3.S4, V4.S4, V5.S4    // p = p*r + c4
	VMOV V25.B16, V4.B16
	VFMLA V3.S4, V5.S4, V4.S4    // p = p*r + c5
	VMOV V26.B16, V5.B16
	VFMLA V3.S4, V4.S4, V5.S4    // q = p*r + 1
	VMOV V26.B16, V4.B16
	VFMLA V3.S4, V5.S4, V4.S4    // p = q*r + 1

	// Two-step 2^k (see the header): k1 = k>>1, k2 = k-k1, both exponents valid.
	VSSHR $1, V2.S4, V6.S4       // k1
	VSUB V6.S4, V2.S4, V7.S4     // k2 = k - k1
	VADD V27.S4, V6.S4, V6.S4    // e1 = k1 + 127
	VADD V27.S4, V7.S4, V7.S4    // e2 = k2 + 127
	VSHL $23, V6.S4, V6.S4       // 2^k1 as float bits
	VSHL $23, V7.S4, V7.S4       // 2^k2
	VFMUL V6.S4, V4.S4, V4.S4    // p * 2^k1   (exact)
	VFMUL V7.S4, V4.S4, V4.S4    // * 2^k2     (exact)

	// Flush to zero where the scalar would: e = k+127 <= 0.
	VADD V27.S4, V2.S4, V8.S4    // e = k + 127
	VCMGT V28.S4, V8.S4, V9.S4   // mask = (e > 0) ? ~0 : 0
	VAND V9.B16, V4.B16, V4.B16

	VST1.P [V4.S4], 16(R0)
	SUBS $4, R2, R2
	BNE loop
done:
	RET


// ---------------------------------------------------------------------------
// The softmax kernel lives in THIS file rather than its own, and that is not
// tidiness: the constant blocks above are declared with the <> suffix, which
// makes them FILE-LOCAL. A second .s file referencing them compiles clean and
// fails at LINK time ("relocation target cLog2e not defined") — `go build` on the
// package does not surface it, only an actual link does. Keeping both kernels
// beside the constants they share is the fix; splitting them again would need
// the constants un-scoped into package-global symbols.
// ---------------------------------------------------------------------------

// func expBiasSumF32NEON(dst, src *float32, n int, bias float32, partials *float64)
// n must be a multiple of 4; inputs (after bias) must be finite and within
// [expUnderflowF32, expOverflowF32].
TEXT ·expBiasSumF32NEON(SB), NOSPLIT, $0-40
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD n+16(FP), R2
	FMOVS bias+24(FP), F29
	MOVD partials+32(FP), R5
	VDUP V29.S[0], V29.S4
	MOVD $cLog2e<>(SB), R3
	VLD1 (R3), [V16.S4]
	MOVD $cMagic<>(SB), R3
	VLD1 (R3), [V17.S4]
	MOVD $cLn2Hi<>(SB), R3
	VLD1 (R3), [V18.S4]
	MOVD $cLn2Lo<>(SB), R3
	VLD1 (R3), [V19.S4]
	MOVD $cP0<>(SB), R3
	VLD1 (R3), [V20.S4]
	MOVD $cP1<>(SB), R3
	VLD1 (R3), [V21.S4]
	MOVD $cP2<>(SB), R3
	VLD1 (R3), [V22.S4]
	MOVD $cP3<>(SB), R3
	VLD1 (R3), [V23.S4]
	MOVD $cP4<>(SB), R3
	VLD1 (R3), [V24.S4]
	MOVD $cP5<>(SB), R3
	VLD1 (R3), [V25.S4]
	MOVD $cOne<>(SB), R3
	VLD1 (R3), [V26.S4]
	MOVD $127, R4
	VDUP R4, V27.S4
	VMOVI $0, V28.B16
	VMOVI $0, V30.B16          // partials[0..1]
	VMOVI $0, V31.B16          // partials[2..3]
	MOVD $cClamp<>(SB), R3
	VLD1 (R3), [V15.S4]        // -104: below the flush threshold, but FINITE
	CBZ R2, sumdone

sumloop:
	VLD1.P 16(R1), [V0.S4]
	VFADD V29.S4, V0.S4, V0.S4   // x + bias  (= x - rowMax)
	// Clamp to a finite value that still flushes. A row containing -Inf makes
	// x+bias = -Inf, and VFCVTZS(-Inf) saturates to INT32_MIN while Go's own
	// float->int conversion of -Inf is implementation-specific -- so the two paths
	// would produce DIFFERENT garbage rather than the same answer. -104 is below
	// expUnderflowF32, so exp of it is 0 either way and nothing observable moves.
	VFMAX V15.S4, V0.S4, V0.S4
	// ---- expF32Contract body, four lanes; V0 in, V4 out ----
	VFMUL V16.S4, V0.S4, V1.S4   // z  = x * log2e
	VFADD V17.S4, V1.S4, V1.S4   // t  = z + magic
	VFSUB V17.S4, V1.S4, V1.S4   // kf = t - magic
	VFCVTZS V1.S4, V2.S4         // k  = int32(kf)
	VMOV V0.B16, V3.B16          // r  = x
	VFMLS V18.S4, V1.S4, V3.S4   // r -= kf*ln2Hi
	VFMLS V19.S4, V1.S4, V3.S4   // r -= kf*ln2Lo
	VMOV V21.B16, V4.B16
	VFMLA V3.S4, V20.S4, V4.S4
	VMOV V22.B16, V5.B16
	VFMLA V3.S4, V4.S4, V5.S4
	VMOV V23.B16, V4.B16
	VFMLA V3.S4, V5.S4, V4.S4
	VMOV V24.B16, V5.B16
	VFMLA V3.S4, V4.S4, V5.S4
	VMOV V25.B16, V4.B16
	VFMLA V3.S4, V5.S4, V4.S4
	VMOV V26.B16, V5.B16
	VFMLA V3.S4, V4.S4, V5.S4    // q = p*r + 1
	VMOV V26.B16, V4.B16
	VFMLA V3.S4, V5.S4, V4.S4    // p = q*r + 1
	VSSHR $1, V2.S4, V6.S4
	VSUB V6.S4, V2.S4, V7.S4
	VADD V27.S4, V6.S4, V6.S4
	VADD V27.S4, V7.S4, V7.S4
	VSHL $23, V6.S4, V6.S4
	VSHL $23, V7.S4, V7.S4
	VFMUL V6.S4, V4.S4, V4.S4
	VFMUL V7.S4, V4.S4, V4.S4
	VADD V27.S4, V2.S4, V8.S4
	VCMGT V28.S4, V8.S4, V9.S4
	VAND V9.B16, V4.B16, V4.B16
	VST1.P [V4.S4], 16(R0)
	// widen and accumulate, lanes 0,1 -> partials[0,1]; lanes 2,3 -> partials[2,3]
	VFCVTL V4.S2, V10.D2
	VFCVTL2 V4.S4, V11.D2
	VFADD V10.D2, V30.D2, V30.D2
	VFADD V11.D2, V31.D2, V31.D2
	SUBS $4, R2, R2
	BNE sumloop
sumdone:
	VST1 [V30.D2, V31.D2], (R5)
	RET
