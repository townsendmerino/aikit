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
