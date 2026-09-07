// AVX2 + FMA3 f32 exp for the S-06 step-2 numeric contract — the amd64 twin of
// exp_neon_arm64.s, and bit-identical to it AND to expF32Contract.
//
// That three-way identity is the whole point of the contract: every multiply-add
// here is a real VFMADD/VFNMADD, the Go oracle reaches the same correctly-rounded
// f32 FMA through f64 (see fma32), and FMLA does it on arm64. A model produces the
// same bits on an M1 and on a Zen 2, so a golden generated on one is valid on the
// other.
//
// S-08 FLAGGED GOAMD64 AS UNPINNED. It does not need pinning for this kernel, and
// that is a consequence of the contract rather than luck. The dispatch is a RUNTIME
// check (hasAVX2, which already requires FMA3 and OS YMM state), and the scalar
// fallback is bit-identical, so a v1 build and a v3 build produce identical output
// and differ only in speed. No build flag has to be mandated on consumers.
//
// Constants live in memory and are used as operands directly: AVX2 allows a memory
// source on VMULPS/VFMADD213PS/VPADDD and friends, so eleven coefficients cost zero
// registers and the 16-YMM budget is never in question. That is the one place this
// is EASIER than the NEON version, which had to hold them or reload them.
//
// Operand order was verified by running a probe on the target box before this was
// written, because these are silent when reversed:
//   VFMADD213PS  mem, Ys, Yd  ->  Yd = Ys*Yd + mem
//   VFNMADD231PS mem, Ys, Yd  ->  Yd = Yd - Ys*mem
//   VSUBPS       mem, Ys, Yd  ->  Yd = Ys - mem
//   VPSUBD       Ya,  Yb, Yd  ->  Yd = Yb - Ya
//   VPCMPGTD     Ya,  Yb, Yd  ->  Yd = (Yb > Ya)

//go:build amd64

#include "textflag.h"

DATA acLog2e<>+0(SB)/4, $0x3fb8aa3b
DATA acLog2e<>+4(SB)/4, $0x3fb8aa3b
DATA acLog2e<>+8(SB)/4, $0x3fb8aa3b
DATA acLog2e<>+12(SB)/4, $0x3fb8aa3b
DATA acLog2e<>+16(SB)/4, $0x3fb8aa3b
DATA acLog2e<>+20(SB)/4, $0x3fb8aa3b
DATA acLog2e<>+24(SB)/4, $0x3fb8aa3b
DATA acLog2e<>+28(SB)/4, $0x3fb8aa3b
GLOBL acLog2e<>(SB), RODATA|NOPTR, $32

DATA acMagic<>+0(SB)/4, $0x4b400000
DATA acMagic<>+4(SB)/4, $0x4b400000
DATA acMagic<>+8(SB)/4, $0x4b400000
DATA acMagic<>+12(SB)/4, $0x4b400000
DATA acMagic<>+16(SB)/4, $0x4b400000
DATA acMagic<>+20(SB)/4, $0x4b400000
DATA acMagic<>+24(SB)/4, $0x4b400000
DATA acMagic<>+28(SB)/4, $0x4b400000
GLOBL acMagic<>(SB), RODATA|NOPTR, $32

DATA acLn2Hi<>+0(SB)/4, $0x3f317200
DATA acLn2Hi<>+4(SB)/4, $0x3f317200
DATA acLn2Hi<>+8(SB)/4, $0x3f317200
DATA acLn2Hi<>+12(SB)/4, $0x3f317200
DATA acLn2Hi<>+16(SB)/4, $0x3f317200
DATA acLn2Hi<>+20(SB)/4, $0x3f317200
DATA acLn2Hi<>+24(SB)/4, $0x3f317200
DATA acLn2Hi<>+28(SB)/4, $0x3f317200
GLOBL acLn2Hi<>(SB), RODATA|NOPTR, $32

DATA acLn2Lo<>+0(SB)/4, $0x35bfbe8e
DATA acLn2Lo<>+4(SB)/4, $0x35bfbe8e
DATA acLn2Lo<>+8(SB)/4, $0x35bfbe8e
DATA acLn2Lo<>+12(SB)/4, $0x35bfbe8e
DATA acLn2Lo<>+16(SB)/4, $0x35bfbe8e
DATA acLn2Lo<>+20(SB)/4, $0x35bfbe8e
DATA acLn2Lo<>+24(SB)/4, $0x35bfbe8e
DATA acLn2Lo<>+28(SB)/4, $0x35bfbe8e
GLOBL acLn2Lo<>(SB), RODATA|NOPTR, $32

DATA acP0<>+0(SB)/4, $0x39506967
DATA acP0<>+4(SB)/4, $0x39506967
DATA acP0<>+8(SB)/4, $0x39506967
DATA acP0<>+12(SB)/4, $0x39506967
DATA acP0<>+16(SB)/4, $0x39506967
DATA acP0<>+20(SB)/4, $0x39506967
DATA acP0<>+24(SB)/4, $0x39506967
DATA acP0<>+28(SB)/4, $0x39506967
GLOBL acP0<>(SB), RODATA|NOPTR, $32

DATA acP1<>+0(SB)/4, $0x3ab743ce
DATA acP1<>+4(SB)/4, $0x3ab743ce
DATA acP1<>+8(SB)/4, $0x3ab743ce
DATA acP1<>+12(SB)/4, $0x3ab743ce
DATA acP1<>+16(SB)/4, $0x3ab743ce
DATA acP1<>+20(SB)/4, $0x3ab743ce
DATA acP1<>+24(SB)/4, $0x3ab743ce
DATA acP1<>+28(SB)/4, $0x3ab743ce
GLOBL acP1<>(SB), RODATA|NOPTR, $32

DATA acP2<>+0(SB)/4, $0x3c088908
DATA acP2<>+4(SB)/4, $0x3c088908
DATA acP2<>+8(SB)/4, $0x3c088908
DATA acP2<>+12(SB)/4, $0x3c088908
DATA acP2<>+16(SB)/4, $0x3c088908
DATA acP2<>+20(SB)/4, $0x3c088908
DATA acP2<>+24(SB)/4, $0x3c088908
DATA acP2<>+28(SB)/4, $0x3c088908
GLOBL acP2<>(SB), RODATA|NOPTR, $32

DATA acP3<>+0(SB)/4, $0x3d2aa9c1
DATA acP3<>+4(SB)/4, $0x3d2aa9c1
DATA acP3<>+8(SB)/4, $0x3d2aa9c1
DATA acP3<>+12(SB)/4, $0x3d2aa9c1
DATA acP3<>+16(SB)/4, $0x3d2aa9c1
DATA acP3<>+20(SB)/4, $0x3d2aa9c1
DATA acP3<>+24(SB)/4, $0x3d2aa9c1
DATA acP3<>+28(SB)/4, $0x3d2aa9c1
GLOBL acP3<>(SB), RODATA|NOPTR, $32

DATA acP4<>+0(SB)/4, $0x3e2aaaaa
DATA acP4<>+4(SB)/4, $0x3e2aaaaa
DATA acP4<>+8(SB)/4, $0x3e2aaaaa
DATA acP4<>+12(SB)/4, $0x3e2aaaaa
DATA acP4<>+16(SB)/4, $0x3e2aaaaa
DATA acP4<>+20(SB)/4, $0x3e2aaaaa
DATA acP4<>+24(SB)/4, $0x3e2aaaaa
DATA acP4<>+28(SB)/4, $0x3e2aaaaa
GLOBL acP4<>(SB), RODATA|NOPTR, $32

DATA acP5<>+0(SB)/4, $0x3f000000
DATA acP5<>+4(SB)/4, $0x3f000000
DATA acP5<>+8(SB)/4, $0x3f000000
DATA acP5<>+12(SB)/4, $0x3f000000
DATA acP5<>+16(SB)/4, $0x3f000000
DATA acP5<>+20(SB)/4, $0x3f000000
DATA acP5<>+24(SB)/4, $0x3f000000
DATA acP5<>+28(SB)/4, $0x3f000000
GLOBL acP5<>(SB), RODATA|NOPTR, $32

DATA acOne<>+0(SB)/4, $0x3f800000
DATA acOne<>+4(SB)/4, $0x3f800000
DATA acOne<>+8(SB)/4, $0x3f800000
DATA acOne<>+12(SB)/4, $0x3f800000
DATA acOne<>+16(SB)/4, $0x3f800000
DATA acOne<>+20(SB)/4, $0x3f800000
DATA acOne<>+24(SB)/4, $0x3f800000
DATA acOne<>+28(SB)/4, $0x3f800000
GLOBL acOne<>(SB), RODATA|NOPTR, $32

DATA acClampLo<>+0(SB)/4, $0xc2d00000
DATA acClampLo<>+4(SB)/4, $0xc2d00000
DATA acClampLo<>+8(SB)/4, $0xc2d00000
DATA acClampLo<>+12(SB)/4, $0xc2d00000
DATA acClampLo<>+16(SB)/4, $0xc2d00000
DATA acClampLo<>+20(SB)/4, $0xc2d00000
DATA acClampLo<>+24(SB)/4, $0xc2d00000
DATA acClampLo<>+28(SB)/4, $0xc2d00000
GLOBL acClampLo<>(SB), RODATA|NOPTR, $32

DATA acClampHi<>+0(SB)/4, $0x42b17217
DATA acClampHi<>+4(SB)/4, $0x42b17217
DATA acClampHi<>+8(SB)/4, $0x42b17217
DATA acClampHi<>+12(SB)/4, $0x42b17217
DATA acClampHi<>+16(SB)/4, $0x42b17217
DATA acClampHi<>+20(SB)/4, $0x42b17217
DATA acClampHi<>+24(SB)/4, $0x42b17217
DATA acClampHi<>+28(SB)/4, $0x42b17217
GLOBL acClampHi<>(SB), RODATA|NOPTR, $32

DATA acI127<>+0(SB)/4, $0x0000007f
DATA acI127<>+4(SB)/4, $0x0000007f
DATA acI127<>+8(SB)/4, $0x0000007f
DATA acI127<>+12(SB)/4, $0x0000007f
DATA acI127<>+16(SB)/4, $0x0000007f
DATA acI127<>+20(SB)/4, $0x0000007f
DATA acI127<>+24(SB)/4, $0x0000007f
DATA acI127<>+28(SB)/4, $0x0000007f
GLOBL acI127<>(SB), RODATA|NOPTR, $32


DATA acSignMask<>+0(SB)/4, $0x80000000
DATA acSignMask<>+4(SB)/4, $0x80000000
DATA acSignMask<>+8(SB)/4, $0x80000000
DATA acSignMask<>+12(SB)/4, $0x80000000
DATA acSignMask<>+16(SB)/4, $0x80000000
DATA acSignMask<>+20(SB)/4, $0x80000000
DATA acSignMask<>+24(SB)/4, $0x80000000
DATA acSignMask<>+28(SB)/4, $0x80000000
GLOBL acSignMask<>(SB), RODATA|NOPTR, $32

DATA acAbsMask<>+0(SB)/4, $0x7fffffff
DATA acAbsMask<>+4(SB)/4, $0x7fffffff
DATA acAbsMask<>+8(SB)/4, $0x7fffffff
DATA acAbsMask<>+12(SB)/4, $0x7fffffff
DATA acAbsMask<>+16(SB)/4, $0x7fffffff
DATA acAbsMask<>+20(SB)/4, $0x7fffffff
DATA acAbsMask<>+24(SB)/4, $0x7fffffff
DATA acAbsMask<>+28(SB)/4, $0x7fffffff
GLOBL acAbsMask<>(SB), RODATA|NOPTR, $32

DATA ac0625<>+0(SB)/4, $0x3f200000
DATA ac0625<>+4(SB)/4, $0x3f200000
DATA ac0625<>+8(SB)/4, $0x3f200000
DATA ac0625<>+12(SB)/4, $0x3f200000
DATA ac0625<>+16(SB)/4, $0x3f200000
DATA ac0625<>+20(SB)/4, $0x3f200000
DATA ac0625<>+24(SB)/4, $0x3f200000
DATA ac0625<>+28(SB)/4, $0x3f200000
GLOBL ac0625<>(SB), RODATA|NOPTR, $32

DATA acT0<>+0(SB)/4, $0xbbbaf0ea
DATA acT0<>+4(SB)/4, $0xbbbaf0ea
DATA acT0<>+8(SB)/4, $0xbbbaf0ea
DATA acT0<>+12(SB)/4, $0xbbbaf0ea
DATA acT0<>+16(SB)/4, $0xbbbaf0ea
DATA acT0<>+20(SB)/4, $0xbbbaf0ea
DATA acT0<>+24(SB)/4, $0xbbbaf0ea
DATA acT0<>+28(SB)/4, $0xbbbaf0ea
GLOBL acT0<>(SB), RODATA|NOPTR, $32

DATA acT1<>+0(SB)/4, $0x3ca9134e
DATA acT1<>+4(SB)/4, $0x3ca9134e
DATA acT1<>+8(SB)/4, $0x3ca9134e
DATA acT1<>+12(SB)/4, $0x3ca9134e
DATA acT1<>+16(SB)/4, $0x3ca9134e
DATA acT1<>+20(SB)/4, $0x3ca9134e
DATA acT1<>+24(SB)/4, $0x3ca9134e
DATA acT1<>+28(SB)/4, $0x3ca9134e
GLOBL acT1<>(SB), RODATA|NOPTR, $32

DATA acT2<>+0(SB)/4, $0xbd5c1e2d
DATA acT2<>+4(SB)/4, $0xbd5c1e2d
DATA acT2<>+8(SB)/4, $0xbd5c1e2d
DATA acT2<>+12(SB)/4, $0xbd5c1e2d
DATA acT2<>+16(SB)/4, $0xbd5c1e2d
DATA acT2<>+20(SB)/4, $0xbd5c1e2d
DATA acT2<>+24(SB)/4, $0xbd5c1e2d
DATA acT2<>+28(SB)/4, $0xbd5c1e2d
GLOBL acT2<>(SB), RODATA|NOPTR, $32

DATA acT3<>+0(SB)/4, $0x3e088393
DATA acT3<>+4(SB)/4, $0x3e088393
DATA acT3<>+8(SB)/4, $0x3e088393
DATA acT3<>+12(SB)/4, $0x3e088393
DATA acT3<>+16(SB)/4, $0x3e088393
DATA acT3<>+20(SB)/4, $0x3e088393
DATA acT3<>+24(SB)/4, $0x3e088393
DATA acT3<>+28(SB)/4, $0x3e088393
GLOBL acT3<>(SB), RODATA|NOPTR, $32

DATA acT4<>+0(SB)/4, $0xbeaaaa99
DATA acT4<>+4(SB)/4, $0xbeaaaa99
DATA acT4<>+8(SB)/4, $0xbeaaaa99
DATA acT4<>+12(SB)/4, $0xbeaaaa99
DATA acT4<>+16(SB)/4, $0xbeaaaa99
DATA acT4<>+20(SB)/4, $0xbeaaaa99
DATA acT4<>+24(SB)/4, $0xbeaaaa99
DATA acT4<>+28(SB)/4, $0xbeaaaa99
GLOBL acT4<>(SB), RODATA|NOPTR, $32

DATA acTwoOverSqrtPi<>+0(SB)/4, $0x3f906ebb
DATA acTwoOverSqrtPi<>+4(SB)/4, $0x3f906ebb
DATA acTwoOverSqrtPi<>+8(SB)/4, $0x3f906ebb
DATA acTwoOverSqrtPi<>+12(SB)/4, $0x3f906ebb
DATA acTwoOverSqrtPi<>+16(SB)/4, $0x3f906ebb
DATA acTwoOverSqrtPi<>+20(SB)/4, $0x3f906ebb
DATA acTwoOverSqrtPi<>+24(SB)/4, $0x3f906ebb
DATA acTwoOverSqrtPi<>+28(SB)/4, $0x3f906ebb
GLOBL acTwoOverSqrtPi<>(SB), RODATA|NOPTR, $32

DATA acASk<>+0(SB)/4, $0x3ea7ba05
DATA acASk<>+4(SB)/4, $0x3ea7ba05
DATA acASk<>+8(SB)/4, $0x3ea7ba05
DATA acASk<>+12(SB)/4, $0x3ea7ba05
DATA acASk<>+16(SB)/4, $0x3ea7ba05
DATA acASk<>+20(SB)/4, $0x3ea7ba05
DATA acASk<>+24(SB)/4, $0x3ea7ba05
DATA acASk<>+28(SB)/4, $0x3ea7ba05
GLOBL acASk<>(SB), RODATA|NOPTR, $32

DATA acErfSeriesS<>+0(SB)/4, $0x32617183
DATA acErfSeriesS<>+4(SB)/4, $0xb41bbbe3
DATA acErfSeriesS<>+8(SB)/4, $0x35c3d001
DATA acErfSeriesS<>+12(SB)/4, $0xb75debbd
DATA acErfSeriesS<>+16(SB)/4, $0x38e00e01
DATA acErfSeriesS<>+20(SB)/4, $0xba46980c
DATA acErfSeriesS<>+24(SB)/4, $0x3b97b426
DATA acErfSeriesS<>+28(SB)/4, $0xbcc30c31
DATA acErfSeriesS<>+32(SB)/4, $0x3dcccccd
DATA acErfSeriesS<>+36(SB)/4, $0xbeaaaaab
DATA acErfSeriesS<>+40(SB)/4, $0x3f800000
GLOBL acErfSeriesS<>(SB), RODATA|NOPTR, $44

DATA acErfASS<>+0(SB)/4, $0x3f87dc22
DATA acErfASS<>+4(SB)/4, $0xbfba00e3
DATA acErfASS<>+8(SB)/4, $0x3fb5f0e3
DATA acErfASS<>+12(SB)/4, $0xbe91a98e
DATA acErfASS<>+16(SB)/4, $0x3e827906
GLOBL acErfASS<>(SB), RODATA|NOPTR, $20

// func expF32ContractAVX2(dst, src *float32, n int)
// n must be a multiple of 8; the Go caller mops the tail and guards the range,
// exactly as on arm64.
TEXT ·expF32ContractAVX2(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DX
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	TESTQ CX, CX
	JZ expdone

exploop:
	VMOVUPS (SI), Y0
	VMULPS acLog2e<>(SB), Y0, Y1      // z  = x * log2e
	VADDPS acMagic<>(SB), Y1, Y1      // t  = z + magic
	VSUBPS acMagic<>(SB), Y1, Y1      // kf = t - magic
	VCVTTPS2DQ Y1, Y2                 // k  = int32(kf)
	VMOVUPS Y0, Y3                    // r  = x
	VFNMADD231PS acLn2Hi<>(SB), Y1, Y3 // r -= kf*ln2Hi
	VFNMADD231PS acLn2Lo<>(SB), Y1, Y3 // r -= kf*ln2Lo
	VMOVUPS acP0<>(SB), Y4
	VFMADD213PS acP1<>(SB), Y3, Y4
	VFMADD213PS acP2<>(SB), Y3, Y4
	VFMADD213PS acP3<>(SB), Y3, Y4
	VFMADD213PS acP4<>(SB), Y3, Y4
	VFMADD213PS acP5<>(SB), Y3, Y4
	VFMADD213PS acOne<>(SB), Y3, Y4   // q = p*r + 1
	VFMADD213PS acOne<>(SB), Y3, Y4   // p = q*r + 1
	// two-step 2^k, same argument as the NEON version: both exponent fields stay
	// valid over the caller-guaranteed range, and each multiply is exact.
	VPSRAD $1, Y2, Y5
	VPSUBD Y5, Y2, Y6
	VPADDD acI127<>(SB), Y5, Y5
	VPADDD acI127<>(SB), Y6, Y6
	VPSLLD $23, Y5, Y5
	VPSLLD $23, Y6, Y6
	VMULPS Y5, Y4, Y4
	VMULPS Y6, Y4, Y4
	// flush where the scalar does: e = k+127 <= 0
	VPADDD acI127<>(SB), Y2, Y7
	VPXOR Y8, Y8, Y8
	VPCMPGTD Y8, Y7, Y7
	VANDPS Y7, Y4, Y4
	VMOVUPS Y4, (DX)
	ADDQ $32, SI
	ADDQ $32, DX
	SUBQ $8, CX
	JNZ exploop
expdone:
	VZEROUPPER
	RET


// func tanhF32ContractAVX2(dst, src *float32, n int)
//
// tanh under the contract, eight lanes at a time, n a multiple of 8. The amd64
// twin of tanhF32ContractNEON and bit-identical to it and to tanhF32Contract.
//
// TWO BRANCHES, BOTH ALWAYS EVALUATED. The scalar has a polynomial below
// |x| = 0.625 and an exponential form above it; a vector kernel cannot branch per
// lane, so it computes both and blends with VBLENDVPS. Neither side misbehaves
// outside its own range -- the polynomial merely diverges (finite) and the
// exponential form tends to 0 -- so the unused half is wasted work, never a NaN.
//
// The same LATE saturation as the NEON version, and for the same reason: this
// form reaches exactly 1 when 2/(e^2x+1) drops below half an ULP, at |x| >= 9.02
// rather than TanhF32's explicit x > 9, so in the band (9, 9.02) it returns
// 0.99999994. That is a deliberate 1-ULP difference from TanhF32, shared with the
// arm64 kernel and with the scalar contract, which is what keeps all three
// bit-identical. Clamping 2|x| to the exp range keeps larger arguments finite.
//
// Sign is carried as a BIT rather than a multiply: tanh is odd and the computed
// magnitude is non-negative, so masking x's sign bit out up front and OR-ing it
// back reproduces `sign * result` -- everywhere EXCEPT x = -0, where the scalar's
// `a < 0` branch treats -0 as positive and returns +0. One extra compare fixes
// that; see the same note on the NEON twin, which shipped with the bug.
//
// Operand orders below were probed on the target box before this was written --
// VBLENDVPS, VCMPPS and VDIVPS are all silent when reversed:
//   VBLENDVPS Ymask, Ysrc2, Ysrc1, Ydst -> Ydst = mask ? Ysrc2 : Ysrc1
//   VCMPPS $1, Yy, Yx, Yd               -> Yd = (Yx < Yy)
//   VDIVPS Yy, Yx, Yd                   -> Yd = Yx / Yy
TEXT ·tanhF32ContractAVX2(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DX
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	TESTQ CX, CX
	JZ tanhdone

tanhloop:
	VMOVUPS (SI), Y0
	VANDPS acSignMask<>(SB), Y0, Y1    // sign bit, kept for the end
	VANDPS acAbsMask<>(SB), Y0, Y2     // a = |x|
	// -0 is POSITIVE to the scalar contract: its branch is `a < 0`, and -0 < 0 is
	// false, so tanh(-0) is +0 rather than -0. The sign BIT says otherwise, so
	// clear it wherever a is zero. An INTEGER compare suffices: |x| has all-zero
	// bits only for +0.
	VPXOR Y3, Y3, Y3
	VPCMPEQD Y3, Y2, Y4                // a == 0
	VBLENDVPS Y4, Y3, Y1, Y1           // sign = (a == 0) ? 0 : sign

	// ---- polynomial branch, a < 0.625 ----
	VMULPS Y2, Y2, Y3                  // z = a*a
	VMOVUPS acT0<>(SB), Y4
	VFMADD213PS acT1<>(SB), Y3, Y4
	VFMADD213PS acT2<>(SB), Y3, Y4
	VFMADD213PS acT3<>(SB), Y3, Y4
	VFMADD213PS acT4<>(SB), Y3, Y4     // -> p
	VMULPS Y3, Y4, Y4                  // pz = p*z
	VMOVUPS Y2, Y5                     // poly = a
	VFMADD231PS Y2, Y4, Y5             // poly = pz*a + a

	// ---- exponential branch, a >= 0.625 ----
	VADDPS Y2, Y2, Y6                  // t = 2a (exact, no constant needed)
	VMINPS acClampHi<>(SB), Y6, Y6     // t = min(t, 88.72283)
	VMULPS acLog2e<>(SB), Y6, Y7
	VADDPS acMagic<>(SB), Y7, Y7
	VSUBPS acMagic<>(SB), Y7, Y7     // kf
	VCVTTPS2DQ Y7, Y8                // k
	VMOVUPS Y6, Y9                  // r
	VFNMADD231PS acLn2Hi<>(SB), Y7, Y9
	VFNMADD231PS acLn2Lo<>(SB), Y7, Y9
	VMOVUPS acP0<>(SB), Y10
	VFMADD213PS acP1<>(SB), Y9, Y10
	VFMADD213PS acP2<>(SB), Y9, Y10
	VFMADD213PS acP3<>(SB), Y9, Y10
	VFMADD213PS acP4<>(SB), Y9, Y10
	VFMADD213PS acP5<>(SB), Y9, Y10
	VFMADD213PS acOne<>(SB), Y9, Y10 // q = p*r + 1
	VFMADD213PS acOne<>(SB), Y9, Y10 // p = q*r + 1
	VPSRAD $1, Y8, Y11
	VPSUBD Y11, Y8, Y12
	VPADDD acI127<>(SB), Y11, Y11
	VPADDD acI127<>(SB), Y12, Y12
	VPSLLD $23, Y11, Y11
	VPSLLD $23, Y12, Y12
	VMULPS Y11, Y10, Y10
	VMULPS Y12, Y10, Y10
	VPADDD acI127<>(SB), Y8, Y11
	VPXOR Y12, Y12, Y12
	VPCMPGTD Y12, Y11, Y11
	VANDPS Y11, Y10, Y10
	VADDPS acOne<>(SB), Y10, Y10       // e + 1
	VMOVUPS acOne<>(SB), Y11
	VADDPS Y11, Y11, Y11               // 2.0, built from 1+1
	VDIVPS Y10, Y11, Y11               // 2/(e+1)
	VMOVUPS acOne<>(SB), Y13
	VSUBPS Y11, Y13, Y13               // alt = 1 - 2/(e+1)

	// ---- select and re-sign ----
	VCMPPS $1, ac0625<>(SB), Y2, Y14   // mask = (a < 0.625)
	VBLENDVPS Y14, Y5, Y13, Y15        // mask ? poly : alt
	VORPS Y1, Y15, Y15                 // reapply sign
	VMOVUPS Y15, (DX)
	ADDQ $32, SI
	ADDQ $32, DX
	SUBQ $8, CX
	JNZ tanhloop
tanhdone:
	VZEROUPPER
	RET

// func erfF32ContractAVX2(dst, src *float32, n int)
//
// erf under the contract, eight lanes at a time, n a multiple of 8. The amd64
// twin of erfF32ContractNEON.
//
// THREE REGIONS COLLAPSED TO TWO, exactly as on arm64. ErfF32 has a Maclaurin
// series below |x| = 1, the Abramowitz & Stegun 7.1.26 tail above it, and an
// explicit |x| > 4 -> 1 saturation. Only the first split needs a select: the tail
// branch reaches exactly 1 on its own once e^(-x^2) underflows, so the saturation
// is a consequence rather than a case. What it DOES need is a clamp on the
// exponent argument -- at |x| = 100, -x^2 is -10000, far outside the range where
// the two-step 2^k construction is valid.
//
// The sixteen coefficients are walked with VBROADCASTSS off a post-incremented
// pointer rather than stored as sixteen 32-byte replicated blocks. That is half a
// kilobyte saved, but the real reason is that the tables stay in the same shape
// and order as erfSeriesCoeffs/erfASCoeffs in Go -- which is why those are named
// tables there rather than literals, and why a coefficient cannot drift between
// the scalar oracle and either kernel without the golden catching it.
TEXT ·erfF32ContractAVX2(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DX
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	TESTQ CX, CX
	JZ erfdone

erfloop:
	VMOVUPS (SI), Y0
	VANDPS acSignMask<>(SB), Y0, Y1    // sign bit
	VANDPS acAbsMask<>(SB), Y0, Y2     // a = |x|

	// ---- Maclaurin branch, a < 1 ----
	VMULPS Y2, Y2, Y3                  // z = a*a
	LEAQ acErfSeriesS<>(SB), R8
	VBROADCASTSS (R8), Y4
	ADDQ $4, R8
	VBROADCASTSS (R8), Y5
	ADDQ $4, R8
	VFMADD213PS Y5, Y3, Y4
	VBROADCASTSS (R8), Y5
	ADDQ $4, R8
	VFMADD213PS Y5, Y3, Y4
	VBROADCASTSS (R8), Y5
	ADDQ $4, R8
	VFMADD213PS Y5, Y3, Y4
	VBROADCASTSS (R8), Y5
	ADDQ $4, R8
	VFMADD213PS Y5, Y3, Y4
	VBROADCASTSS (R8), Y5
	ADDQ $4, R8
	VFMADD213PS Y5, Y3, Y4
	VBROADCASTSS (R8), Y5
	ADDQ $4, R8
	VFMADD213PS Y5, Y3, Y4
	VBROADCASTSS (R8), Y5
	ADDQ $4, R8
	VFMADD213PS Y5, Y3, Y4
	VBROADCASTSS (R8), Y5
	ADDQ $4, R8
	VFMADD213PS Y5, Y3, Y4
	VBROADCASTSS (R8), Y5
	ADDQ $4, R8
	VFMADD213PS Y5, Y3, Y4
	VBROADCASTSS (R8), Y5
	ADDQ $4, R8
	VFMADD213PS Y5, Y3, Y4
	VMULPS acTwoOverSqrtPi<>(SB), Y2, Y5 // (2/sqrt(pi)) * a
	VMULPS Y4, Y5, Y5                  // series = that * p

	// ---- A&S tail branch, a >= 1 ----
	VMOVUPS acOne<>(SB), Y6
	VFMADD231PS acASk<>(SB), Y2, Y6    // 1 + 0.3275911*a  (fused, as the contract says)
	VMOVUPS acOne<>(SB), Y7
	VDIVPS Y6, Y7, Y6                  // t = 1/that
	LEAQ acErfASS<>(SB), R8
	VBROADCASTSS (R8), Y7
	ADDQ $4, R8
	VBROADCASTSS (R8), Y8
	ADDQ $4, R8
	VFMADD213PS Y8, Y6, Y7
	VBROADCASTSS (R8), Y8
	ADDQ $4, R8
	VFMADD213PS Y8, Y6, Y7
	VBROADCASTSS (R8), Y8
	ADDQ $4, R8
	VFMADD213PS Y8, Y6, Y7
	VBROADCASTSS (R8), Y8
	ADDQ $4, R8
	VFMADD213PS Y8, Y6, Y7
	// Y7 = q

	// ---- e^(-a*a), clamped so the exponent build stays in range ----
	VMULPS Y2, Y2, Y8
	VXORPS acSignMask<>(SB), Y8, Y8    // -a*a
	VMAXPS acClampLo<>(SB), Y8, Y8     // max(e, -104)
	VMULPS acLog2e<>(SB), Y8, Y3
	VADDPS acMagic<>(SB), Y3, Y3
	VSUBPS acMagic<>(SB), Y3, Y3     // kf
	VCVTTPS2DQ Y3, Y4                // k
	VMOVUPS Y8, Y9                  // r
	VFNMADD231PS acLn2Hi<>(SB), Y3, Y9
	VFNMADD231PS acLn2Lo<>(SB), Y3, Y9
	VMOVUPS acP0<>(SB), Y8
	VFMADD213PS acP1<>(SB), Y9, Y8
	VFMADD213PS acP2<>(SB), Y9, Y8
	VFMADD213PS acP3<>(SB), Y9, Y8
	VFMADD213PS acP4<>(SB), Y9, Y8
	VFMADD213PS acP5<>(SB), Y9, Y8
	VFMADD213PS acOne<>(SB), Y9, Y8 // q = p*r + 1
	VFMADD213PS acOne<>(SB), Y9, Y8 // p = q*r + 1
	VPSRAD $1, Y4, Y10
	VPSUBD Y10, Y4, Y11
	VPADDD acI127<>(SB), Y10, Y10
	VPADDD acI127<>(SB), Y11, Y11
	VPSLLD $23, Y10, Y10
	VPSLLD $23, Y11, Y11
	VMULPS Y10, Y8, Y8
	VMULPS Y11, Y8, Y8
	VPADDD acI127<>(SB), Y4, Y10
	VPXOR Y11, Y11, Y11
	VPCMPGTD Y11, Y10, Y10
	VANDPS Y10, Y8, Y8
	VMULPS Y6, Y7, Y9                  // q*t
	VMOVUPS acOne<>(SB), Y13
	VFNMADD231PS Y8, Y9, Y13           // tail = 1 - (q*t)*e   (one fused op)

	// ---- blend and re-sign ----
	VCMPPS $1, acOne<>(SB), Y2, Y14    // mask = (a < 1)
	VBLENDVPS Y14, Y5, Y13, Y15        // mask ? series : tail
	VORPS Y1, Y15, Y15
	VMOVUPS Y15, (DX)
	ADDQ $32, SI
	ADDQ $32, DX
	SUBQ $8, CX
	JNZ erfloop
erfdone:
	VZEROUPPER
	RET
