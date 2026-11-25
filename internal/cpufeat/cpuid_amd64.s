// CPUID and XGETBV, the probes the feature detection in x86_amd64.go runs once at init.

#include "textflag.h"

// func CPUID(eaxIn, ecxIn uint32) (eax, ebx, ecx, edx uint32)
TEXT ·CPUID(SB), NOSPLIT, $0-24
	MOVL eaxIn+0(FP), AX
	MOVL ecxIn+4(FP), CX
	CPUID
	MOVL AX, eax+8(FP)
	MOVL BX, ebx+12(FP)
	MOVL CX, ecx+16(FP)
	MOVL DX, edx+20(FP)
	RET

// func XGETBV() (eax, edx uint32)
TEXT ·XGETBV(SB), NOSPLIT, $0-8
	MOVL $0, CX
	XGETBV
	MOVL AX, eax+0(FP)
	MOVL DX, edx+4(FP)
	RET
