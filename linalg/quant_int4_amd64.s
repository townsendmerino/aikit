// AVX2 int4 weight quantizer for amd64: the round-clamp-pack pass of quantizeInt4GroupScalar, 8 weights a turn.
//
// Per lane: y = x*inv (VMULPS, the scalar's single f32 multiply). Round half away from zero as
// trunc(y + copysign(h, y)) with h = nextafter32(0.5, 0) = 0.49999997: x86 has no ties-away rounding mode, and adding
// exactly 0.5 misrounds 0.49999997 itself (quant_act_amd64.s has the argument; TestQuantizeInt4Block_matchesScalar
// holds this kernel to the scalar over every float32). VCVTTPS2DQ truncates to int32, then an int32 clamp to [-7, 7],
// the bias of 8, and the pack.
//
// What the scalar does on amd64 with a product no int holds, and how the kernel matches it. The scalar converts a
// float64 with CVTTSD2SQ, whose answer for NaN, either infinity and anything past int64 is the int64 minimum, which
// the clamp turns into -7. VCVTTPS2DQ answers the int32 minimum for NaN, either infinity and anything past int32, and
// the clamp turns that into -7 as well. The two differ only for a finite positive product in [2^31, 2^63): the
// scalar's conversion is a large positive int there, so 7. Those lanes are found with two ordered compares and set
// to 7 before the clamp. (A product of a real group never gets there: it is at most 7 and a rounding, or it is not
// finite. The kernel is exact for every float32 anyway.)
//
// The pack: VPACKSSDW gives the eight nibbles as words w0..w7 (each 128-bit half packed alone, order kept; no
// saturation, the values are 1..15). Read as four dwords, lane j is w[2j] | w[2j+1]<<16; OR-ing in the lane shifted
// right by 12 puts w[2j+1]<<4 beside w[2j] in the low byte. Mask to that byte, then VPACKUSDW and VPACKUSWB narrow
// the four dwords to four bytes.

#include "textflag.h"

// The kernel runs once per group, so its constants are read from memory: a move from a general register into an XMM
// register is a legacy-SSE instruction in Go's assembler, and with the upper halves of the YMM registers in use each
// one costs a state transition.
DATA int4h<>(SB)/4, $0x3EFFFFFF
GLOBL int4h<>(SB), RODATA|NOPTR, $4
DATA int4p7<>(SB)/4, $7
GLOBL int4p7<>(SB), RODATA|NOPTR, $4
DATA int4m7<>(SB)/4, $0xFFFFFFF9
GLOBL int4m7<>(SB), RODATA|NOPTR, $4
DATA int4bias<>(SB)/4, $8
GLOBL int4bias<>(SB), RODATA|NOPTR, $4
DATA int4two31<>(SB)/4, $0x4F000000
GLOBL int4two31<>(SB), RODATA|NOPTR, $4
DATA int4two63<>(SB)/4, $0x5F000000
GLOBL int4two63<>(SB), RODATA|NOPTR, $4

// func quantizeInt4F32AVX2(row *float32, packed *byte, n int, inv float32)
TEXT ·quantizeInt4F32AVX2(SB), NOSPLIT, $0-28
	MOVQ         row+0(FP), SI
	MOVQ         packed+8(FP), DI
	MOVQ         n+16(FP), CX
	VBROADCASTSS inv+24(FP), Y14

	VPCMPEQD Y15, Y15, Y15
	VPSLLD   $31, Y15, Y13   // 0x80000000 in every lane: the sign mask
	VPSRLD   $24, Y15, Y9    // 0x000000FF in every lane: the packed byte

	VPBROADCASTD int4h<>(SB), Y12     // h = nextafter32(0.5, 0)
	VPBROADCASTD int4p7<>(SB), Y11    // +7, int32 lanes
	VPBROADCASTD int4m7<>(SB), Y10    // -7, int32 lanes
	VPBROADCASTD int4bias<>(SB), Y8   // the nibble bias
	VPBROADCASTD int4two31<>(SB), Y7  // 2^31 as a float32
	VPBROADCASTD int4two63<>(SB), Y6  // 2^63 as a float32

quant8:
	VMOVUPS (SI), Y0
	VMULPS  Y14, Y0, Y0           // y = x * inv

	VANDPS  Y13, Y0, Y1           // the sign bit of y
	VORPS   Y12, Y1, Y1           // h with y's sign
	VADDPS  Y1, Y0, Y2            // r = y + copysign(h, y)
	VCVTTPS2DQ Y2, Y3             // q = trunc(r); the int32 minimum for NaN, an infinity, or |r| >= 2^31

	VCMPPS    $0x1D, Y7, Y2, Y4   // r >= 2^31 (ordered)
	VCMPPS    $0x11, Y6, Y2, Y5   // r <  2^63 (ordered)
	VPAND     Y5, Y4, Y4          // a finite positive product the scalar's int64 conversion keeps positive
	VPBLENDVB Y4, Y11, Y3, Y3     // q = 7 in those lanes

	VPMINSD Y11, Y3, Y3           // min(q, 7)
	VPMAXSD Y10, Y3, Y3           // max(q, -7)
	VPADDD  Y8, Y3, Y3            // a nibble, 1..15

	VEXTRACTI128 $1, Y3, X5
	VPACKSSDW    X5, X3, X3       // the eight nibbles as words, order kept
	VPSRLD       $12, X3, X4      // each dword: the odd nibble moved down to bits 4..7
	VPOR         X4, X3, X3
	VPAND        X9, X3, X3       // each dword's low byte: even nibble | odd nibble << 4
	VPACKUSDW    X3, X3, X3
	VPACKUSWB    X3, X3, X3
	VMOVD        X3, (DI)         // four packed bytes

	ADDQ $32, SI
	ADDQ $4, DI
	SUBQ $8, CX
	JNE  quant8

	VZEROUPPER
	RET
