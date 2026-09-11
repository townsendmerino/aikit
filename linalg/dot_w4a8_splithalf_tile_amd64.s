// AVX2 register-blocked W4A8 tile over the SPLIT-HALF weight layout —
// dot_w4a8_tile_amd64.s's split-half twin, exactly the way
// dotW4A8SplitHalfAVX2 (dot_w4a8_amd64.s) is dotW4A8FoldAVX2's: same
// 4-activation-row register blocking as the canonical tile, only the
// nibble-unpack prologue changed — no VPUNPCKLBW/VPUNPCKHBW, because
// split-half's low/high nibbles of a group's 16 packed bytes ARE weight
// i (low, i=0..15) and weight i+16 (high) already, with no interleave to
// undo. See dot_w4a8_tile_amd64.s for the register-blocking rationale
// (why four rows, why this fixes the YMM budget) and dotW4A8SplitHalfAVX2
// for the unpack rationale; this file only combines the two — the unpack
// prologue is copied instruction-for-instruction from
// dotW4A8SplitHalfAVX2, the row-blocking body instruction-for-instruction
// from dotW4A8Tile4RowAVX2.
//
// BIT-IDENTICAL BY CONSTRUCTION to dotW4A8SplitHalfAVX2 for the same
// logical weights, the same way dotW4A8Tile4RowAVX2 is bit-identical to
// dotW4A8FoldAVX2: per-group VPMADDWD/VPMADDWD/VPADDD/VCVTDQ2PS/
// VFMADD231PS in ascending group order, four independent accumulator
// chains, the same VEXTRACTF128/VADDPS/VHADDPS/VHADDPS epilogue reduce.
// Only the unpack prologue differs from the canonical tile — exactly as
// it differs between the two M=1 kernels, and nothing else.

#include "textflag.h"

DATA shtmask0F<>+0(SB)/8, $0x0F0F0F0F0F0F0F0F
DATA shtmask0F<>+8(SB)/8, $0x0F0F0F0F0F0F0F0F
GLOBL shtmask0F<>(SB), RODATA|NOPTR, $16

DATA shtconst8<>+0(SB)/8, $0x0808080808080808
DATA shtconst8<>+8(SB)/8, $0x0808080808080808
GLOBL shtconst8<>(SB), RODATA|NOPTR, $16

// func dotW4A8SplitHalfTile4RowAVX2(act *int8, actStride int, packed *byte, scales *float32, dst *float32, nGroups int)
TEXT ·dotW4A8SplitHalfTile4RowAVX2(SB), NOSPLIT, $0-48
	MOVQ act+0(FP), SI
	MOVQ actStride+8(FP), R11
	MOVQ packed+16(FP), DI
	MOVQ scales+24(FP), BX
	MOVQ dst+32(FP), DX
	MOVQ nGroups+40(FP), CX

	// The other three activation rows. Each of the four advances by 32 per
	// group independently, so no index arithmetic runs in the loop.
	LEAQ (SI)(R11*1), R8
	LEAQ (R8)(R11*1), R9
	LEAQ (R9)(R11*1), R10

	LEAQ    shtmask0F<>(SB), AX
	VMOVDQU (AX), X14
	LEAQ    shtconst8<>(SB), AX
	VMOVDQU (AX), X15

	VXORPS Y8, Y8, Y8    // acc row 0
	VXORPS Y9, Y9, Y9    // acc row 1
	VXORPS Y10, Y10, Y10 // acc row 2
	VXORPS Y13, Y13, Y13 // acc row 3

shtile4rowloop:
	// ---- the weight row's 16 packed split-half bytes: unpacked ONCE for all four rows ----
	// Split-half unpack: low nibbles ARE w0..w15, high nibbles ARE w16..w31.
	// No VPUNPCK — that is the whole difference from the canonical tile.
	VMOVDQU   (DI), X0
	VPAND     X14, X0, X1
	VPSRLW    $4, X0, X2
	VPAND     X14, X2, X2
	VPSUBB    X15, X1, X1
	VPSUBB    X15, X2, X2
	VPMOVSXBW X1, Y3
	VPMOVSXBW X2, Y4
	VBROADCASTSS (BX), Y11

	// ---- activation row 0 ----
	VMOVDQU     (SI), X5
	VMOVDQU     16(SI), X6
	VPMOVSXBW   X5, Y5
	VPMOVSXBW   X6, Y6
	VPMADDWD    Y5, Y3, Y7
	VPMADDWD    Y6, Y4, Y12
	VPADDD      Y12, Y7, Y7
	VCVTDQ2PS   Y7, Y7
	VFMADD231PS Y11, Y7, Y8

	// ---- activation row 1 ----
	VMOVDQU     (R8), X5
	VMOVDQU     16(R8), X6
	VPMOVSXBW   X5, Y5
	VPMOVSXBW   X6, Y6
	VPMADDWD    Y5, Y3, Y7
	VPMADDWD    Y6, Y4, Y12
	VPADDD      Y12, Y7, Y7
	VCVTDQ2PS   Y7, Y7
	VFMADD231PS Y11, Y7, Y9

	// ---- activation row 2 ----
	VMOVDQU     (R9), X5
	VMOVDQU     16(R9), X6
	VPMOVSXBW   X5, Y5
	VPMOVSXBW   X6, Y6
	VPMADDWD    Y5, Y3, Y7
	VPMADDWD    Y6, Y4, Y12
	VPADDD      Y12, Y7, Y7
	VCVTDQ2PS   Y7, Y7
	VFMADD231PS Y11, Y7, Y10

	// ---- activation row 3 ----
	VMOVDQU     (R10), X5
	VMOVDQU     16(R10), X6
	VPMOVSXBW   X5, Y5
	VPMOVSXBW   X6, Y6
	VPMADDWD    Y5, Y3, Y7
	VPMADDWD    Y6, Y4, Y12
	VPADDD      Y12, Y7, Y7
	VCVTDQ2PS   Y7, Y7
	VFMADD231PS Y11, Y7, Y13

	ADDQ $16, DI
	ADDQ $4, BX
	ADDQ $32, SI
	ADDQ $32, R8
	ADDQ $32, R9
	ADDQ $32, R10
	SUBQ $1, CX
	JNZ  shtile4rowloop

	// Four horizontal f32 reduces, the same tree dotW4A8SplitHalfAVX2 ends with.
	VEXTRACTF128 $1, Y8, X11
	VADDPS       X11, X8, X8
	VHADDPS      X8, X8, X8
	VHADDPS      X8, X8, X8
	MOVSS        X8, (DX)

	VEXTRACTF128 $1, Y9, X11
	VADDPS       X11, X9, X9
	VHADDPS      X9, X9, X9
	VHADDPS      X9, X9, X9
	MOVSS        X9, 4(DX)

	VEXTRACTF128 $1, Y10, X11
	VADDPS       X11, X10, X10
	VHADDPS      X10, X10, X10
	VHADDPS      X10, X10, X10
	MOVSS        X10, 8(DX)

	VEXTRACTF128 $1, Y13, X11
	VADDPS       X11, X13, X13
	VHADDPS      X13, X13, X13
	VHADDPS      X13, X13, X13
	MOVSS        X13, 12(DX)

	VZEROUPPER
	RET
