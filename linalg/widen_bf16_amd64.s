// bf16 -> f32 widen on amd64 (widen_bf16_amd64.go): zero-extend each 16-bit pattern to 32 bits and shift it into the high half. Exact. Needs AVX2.

#include "textflag.h"

// func cvtBF16ToF32x8AVX2(dst *float32, src *uint16, n8 int)
// dst[0:8*n8] = f32(src[0:8*n8]), sixteen per iteration while two blocks remain, then eight.
TEXT ·cvtBF16ToF32x8AVX2(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n8+16(FP), CX
	CMPQ CX, $2
	JB   one
two:
	VPMOVZXWD (SI), Y0
	VPMOVZXWD 16(SI), Y1
	VPSLLD    $16, Y0, Y0
	VPSLLD    $16, Y1, Y1
	VMOVDQU   Y0, (DI)
	VMOVDQU   Y1, 32(DI)
	ADDQ      $32, SI
	ADDQ      $64, DI
	SUBQ      $2, CX
	CMPQ      CX, $2
	JAE       two
one:
	TESTQ CX, CX
	JZ    done
	VPMOVZXWD (SI), Y0
	VPSLLD    $16, Y0, Y0
	VMOVDQU   Y0, (DI)
done:
	VZEROUPPER
	RET
