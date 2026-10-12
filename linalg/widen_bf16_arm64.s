//go:build arm64

#include "textflag.h"

// func cvtBF16ToF32x8NEON(dst *float32, src *uint16, n8 int)
//
// dst[0:8*n8] = f32(src[0:8*n8]): n8 blocks of eight bfloat16 patterns, each widened by SHLL (low four halves) and SHLL2 (high four), which
// shift every 16-bit element left by 16 into a 32-bit lane: exactly the bf16 -> f32 widen, every bit pattern (NaN payloads and subnormals
// included) unchanged. Sixteen per iteration while two blocks remain, then one. The arm64 twin of cvtBF16ToF32x8AVX2 (widen_bf16_amd64.s).
//
// WORDs from the encoding SHLL{2} Vd.{4S,..}, Vn.{4H,8H}, #16 = 0x2E613800 | Q<<30 | Rn<<5 | Rd (AdvSIMD two-register misc, size=01),
// i.e. SHLL V2.4S, V0.4H = 0x2E613802, SHLL2 V3.4S, V0.8H = 0x6E613803, SHLL V4.4S, V1.4H = 0x2E613824, SHLL2 V5.4S, V1.8H = 0x6E613825.
// No assembler was at hand to cross-check them (the f16 file's clang cross-check could not be repeated); TestWidenBF16_exhaustive runs all 65,536
// patterns through this routine, which a wrong encoding cannot pass.
TEXT ·cvtBF16ToF32x8NEON(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD n8+16(FP), R2
	CMP  $2, R2
	BLT  one

two:
	VLD1.P 32(R1), [V0.H8, V1.H8]
	WORD   $0x2E613802 // SHLL  V2.4S, V0.4H, #16
	WORD   $0x6E613803 // SHLL2 V3.4S, V0.8H, #16
	WORD   $0x2E613824 // SHLL  V4.4S, V1.4H, #16
	WORD   $0x6E613825 // SHLL2 V5.4S, V1.8H, #16
	VST1.P [V2.S4, V3.S4, V4.S4, V5.S4], 64(R0)
	SUB    $2, R2, R2
	CMP    $2, R2
	BGE    two

one:
	CBZ    R2, done
	VLD1.P 16(R1), [V0.H8]
	WORD   $0x2E613802 // SHLL  V2.4S, V0.4H, #16
	WORD   $0x6E613803 // SHLL2 V3.4S, V0.8H, #16
	VST1.P [V2.S4, V3.S4], 32(R0)

done:
	RET
