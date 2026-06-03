// AVX2 + FMA f32 dot product for amd64, plus the CPUID/XGETBV probes
// the Go-side feature detection in dot_amd64.go calls once at init.
//
//   dotFMA  — single-row dot product, Σ a[i]*b[i] over n elements,
//             returned as a float32. Four YMM accumulators (32 floats
//             per main-loop iter) for instruction-level parallelism,
//             then an 8-wide tail, then a scalar tail. VEX-encoded
//             throughout + VZEROUPPER so there's no AVX/SSE transition
//             penalty.
//   dotFMA4 — FOUR rows at once: a·b0 … a·b3 over n elements, full sums
//             written to out[0..3]. Loads each 8-float chunk of `a`
//             ONCE and runs 4 FMAs against b0..b3 (memory operands) into
//             4 YMM accumulators — the a-reuse register-blocking the
//             arm64 NEON kernel does. n is a multiple of 4 (caller's
//             contract); an 8-wide main loop + one 4-wide step covers it.
//   dotFMA8 — same, EIGHT rows (a·b0 … a·b7 → out[0..7]); one a-load
//             feeds 8 FMA chains, the matmul workhorse.
//   cpuid   — raw CPUID leaf/subleaf probe.
//   xgetbv  — XCR0 read (OS extended-state mask), subleaf 0.
//
// The horizontal reduction is done here in-asm (unlike the arm64 path)
// because amd64 has cheap VHADDPS.

#include "textflag.h"

// func dotFMA(a *float32, b *float32, n int) float32
TEXT ·dotFMA(SB), NOSPLIT, $0-28
	MOVQ a+0(FP), SI
	MOVQ b+8(FP), DI
	MOVQ n+16(FP), CX

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3

	MOVQ CX, AX // AX = remaining element count

loop32:
	CMPQ AX, $32
	JL   loop8
	VMOVUPS 0(SI), Y4
	VMOVUPS 32(SI), Y5
	VMOVUPS 64(SI), Y6
	VMOVUPS 96(SI), Y7
	VMOVUPS 0(DI), Y8
	VMOVUPS 32(DI), Y9
	VMOVUPS 64(DI), Y10
	VMOVUPS 96(DI), Y11
	VFMADD231PS Y8, Y4, Y0
	VFMADD231PS Y9, Y5, Y1
	VFMADD231PS Y10, Y6, Y2
	VFMADD231PS Y11, Y7, Y3
	ADDQ $128, SI // 32 floats × 4 bytes
	ADDQ $128, DI
	SUBQ $32, AX
	JMP  loop32

loop8:
	CMPQ AX, $8
	JL   reduce
	VMOVUPS 0(SI), Y4
	VMOVUPS 0(DI), Y8
	VFMADD231PS Y8, Y4, Y0
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, AX
	JMP  loop8

reduce:
	// Fold the 4 YMM accumulators into Y0, then horizontally sum its
	// 8 lanes into the low f32 of X0.
	VADDPS       Y1, Y0, Y0
	VADDPS       Y3, Y2, Y2
	VADDPS       Y2, Y0, Y0
	VEXTRACTF128 $1, Y0, X1 // high 128 bits → X1
	VADDPS       X1, X0, X0 // X0 = 4 partial sums
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0 // X0 low lane = total

tail:
	// Scalar remainder (AX < 8). VEX scalar forms keep us out of legacy
	// SSE so VZEROUPPER below is the only transition guard needed.
	CMPQ AX, $0
	JE   done
	VMOVSS 0(SI), X2
	VMULSS 0(DI), X2, X2
	VADDSS X2, X0, X0
	ADDQ   $4, SI
	ADDQ   $4, DI
	DECQ   AX
	JMP    tail

done:
	VZEROUPPER
	MOVSS X0, ret+24(FP)
	RET

// func dotFMA4(a, b0, b1, b2, b3 *float32, n int, out *[4]float32)
TEXT ·dotFMA4(SB), NOSPLIT, $0-56
	MOVQ a+0(FP), SI
	MOVQ b0+8(FP), R8
	MOVQ b1+16(FP), R9
	MOVQ b2+24(FP), R10
	MOVQ b3+32(FP), R11
	MOVQ n+40(FP), CX

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3

dotFMA4_loop8:
	CMPQ CX, $8
	JL   dotFMA4_tail4
	VMOVUPS     0(SI), Y8
	VFMADD231PS 0(R8), Y8, Y0
	VFMADD231PS 0(R9), Y8, Y1
	VFMADD231PS 0(R10), Y8, Y2
	VFMADD231PS 0(R11), Y8, Y3
	ADDQ        $32, SI
	ADDQ        $32, R8
	ADDQ        $32, R9
	ADDQ        $32, R10
	ADDQ        $32, R11
	SUBQ        $8, CX
	JMP         dotFMA4_loop8

dotFMA4_tail4:
	CMPQ CX, $4
	JL   dotFMA4_reduce
	VMOVUPS     0(SI), X8
	VFMADD231PS 0(R8), X8, X0
	VFMADD231PS 0(R9), X8, X1
	VFMADD231PS 0(R10), X8, X2
	VFMADD231PS 0(R11), X8, X3
	ADDQ        $16, SI
	ADDQ        $16, R8
	ADDQ        $16, R9
	ADDQ        $16, R10
	ADDQ        $16, R11
	SUBQ        $4, CX

dotFMA4_reduce:
	MOVQ out+48(FP), DI

	VEXTRACTF128 $1, Y0, X9
	VADDPS       X9, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	MOVSS        X0, 0(DI)

	VEXTRACTF128 $1, Y1, X9
	VADDPS       X9, X1, X1
	VHADDPS      X1, X1, X1
	VHADDPS      X1, X1, X1
	MOVSS        X1, 4(DI)

	VEXTRACTF128 $1, Y2, X9
	VADDPS       X9, X2, X2
	VHADDPS      X2, X2, X2
	VHADDPS      X2, X2, X2
	MOVSS        X2, 8(DI)

	VEXTRACTF128 $1, Y3, X9
	VADDPS       X9, X3, X3
	VHADDPS      X3, X3, X3
	VHADDPS      X3, X3, X3
	MOVSS        X3, 12(DI)

	VZEROUPPER
	RET

// func dotFMA8(a, b0, b1, b2, b3, b4, b5, b6, b7 *float32, n int, out *[8]float32)
TEXT ·dotFMA8(SB), NOSPLIT, $0-88
	MOVQ a+0(FP), SI
	MOVQ b0+8(FP), R8
	MOVQ b1+16(FP), R9
	MOVQ b2+24(FP), R10
	MOVQ b3+32(FP), R11
	MOVQ b4+40(FP), R12
	MOVQ b5+48(FP), R13
	MOVQ b6+56(FP), R14
	MOVQ b7+64(FP), R15
	MOVQ n+72(FP), CX

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7

dotFMA8_loop8:
	CMPQ CX, $8
	JL   dotFMA8_tail4
	VMOVUPS     0(SI), Y8
	VFMADD231PS 0(R8), Y8, Y0
	VFMADD231PS 0(R9), Y8, Y1
	VFMADD231PS 0(R10), Y8, Y2
	VFMADD231PS 0(R11), Y8, Y3
	VFMADD231PS 0(R12), Y8, Y4
	VFMADD231PS 0(R13), Y8, Y5
	VFMADD231PS 0(R14), Y8, Y6
	VFMADD231PS 0(R15), Y8, Y7
	ADDQ        $32, SI
	ADDQ        $32, R8
	ADDQ        $32, R9
	ADDQ        $32, R10
	ADDQ        $32, R11
	ADDQ        $32, R12
	ADDQ        $32, R13
	ADDQ        $32, R14
	ADDQ        $32, R15
	SUBQ        $8, CX
	JMP         dotFMA8_loop8

dotFMA8_tail4:
	CMPQ CX, $4
	JL   dotFMA8_reduce
	VMOVUPS     0(SI), X8
	VFMADD231PS 0(R8), X8, X0
	VFMADD231PS 0(R9), X8, X1
	VFMADD231PS 0(R10), X8, X2
	VFMADD231PS 0(R11), X8, X3
	VFMADD231PS 0(R12), X8, X4
	VFMADD231PS 0(R13), X8, X5
	VFMADD231PS 0(R14), X8, X6
	VFMADD231PS 0(R15), X8, X7
	ADDQ        $16, SI
	ADDQ        $16, R8
	ADDQ        $16, R9
	ADDQ        $16, R10
	ADDQ        $16, R11
	ADDQ        $16, R12
	ADDQ        $16, R13
	ADDQ        $16, R14
	ADDQ        $16, R15
	SUBQ        $4, CX

dotFMA8_reduce:
	MOVQ out+80(FP), DI

	VEXTRACTF128 $1, Y0, X9
	VADDPS       X9, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	MOVSS        X0, 0(DI)

	VEXTRACTF128 $1, Y1, X9
	VADDPS       X9, X1, X1
	VHADDPS      X1, X1, X1
	VHADDPS      X1, X1, X1
	MOVSS        X1, 4(DI)

	VEXTRACTF128 $1, Y2, X9
	VADDPS       X9, X2, X2
	VHADDPS      X2, X2, X2
	VHADDPS      X2, X2, X2
	MOVSS        X2, 8(DI)

	VEXTRACTF128 $1, Y3, X9
	VADDPS       X9, X3, X3
	VHADDPS      X3, X3, X3
	VHADDPS      X3, X3, X3
	MOVSS        X3, 12(DI)

	VEXTRACTF128 $1, Y4, X9
	VADDPS       X9, X4, X4
	VHADDPS      X4, X4, X4
	VHADDPS      X4, X4, X4
	MOVSS        X4, 16(DI)

	VEXTRACTF128 $1, Y5, X9
	VADDPS       X9, X5, X5
	VHADDPS      X5, X5, X5
	VHADDPS      X5, X5, X5
	MOVSS        X5, 20(DI)

	VEXTRACTF128 $1, Y6, X9
	VADDPS       X9, X6, X6
	VHADDPS      X6, X6, X6
	VHADDPS      X6, X6, X6
	MOVSS        X6, 24(DI)

	VEXTRACTF128 $1, Y7, X9
	VADDPS       X9, X7, X7
	VHADDPS      X7, X7, X7
	VHADDPS      X7, X7, X7
	MOVSS        X7, 28(DI)

	VZEROUPPER
	RET

// func cpuid(eaxIn, ecxIn uint32) (eax, ebx, ecx, edx uint32)
TEXT ·cpuid(SB), NOSPLIT, $0-24
	MOVL eaxIn+0(FP), AX
	MOVL ecxIn+4(FP), CX
	CPUID
	MOVL AX, eax+8(FP)
	MOVL BX, ebx+12(FP)
	MOVL CX, ecx+16(FP)
	MOVL DX, edx+20(FP)
	RET

// func xgetbv() (eax, edx uint32)
TEXT ·xgetbv(SB), NOSPLIT, $0-8
	MOVL $0, CX
	XGETBV
	MOVL AX, eax+0(FP)
	MOVL DX, edx+4(FP)
	RET
