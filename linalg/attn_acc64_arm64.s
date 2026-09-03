// NEON f64 lane-per-output ports of the acc64 attention kernels for arm64
// (task-simd-audit.md S-04, step 2).
//
//   avAcc64NEON32 — one query row's scores·V over a 32-dim block of the head:
//                   dst[d] = float32(Σ_s float64(scores[s]) · float64(V[s][d])),
//                   the sum taken in key order s = 0..nKeys-1, 32 dims at once.
//   qkAcc64NEON16 — one query row's q·Kᵀ over blocks of 16 keys:
//                   dst[j] = float32(Σ_d float64(q[d]) · float64(K[j][d])),
//                   the sum taken in d order, 16 keys at once.
//
// WHY THESE ARE BIT-IDENTICAL to the Go kernels (matmul_av_acc64.go,
// matmul_qk_acc64.go), and to each other's scalar tails. Every operand is an f32
// widened to f64, so every product is EXACT in f64 (24+24 significand bits ≤ 53,
// and the exponent range is nowhere near f64's). An accumulator step is therefore
// one rounding, RN(acc + q·k), whether it runs as the Go compiler's scalar FMADDD,
// as an unfused multiply-then-add, or as a vector FMLA lane — and the lanes here
// hold INDEPENDENT outputs (one dim per lane in AV, one key per lane in QK) that
// each receive their own adds in exactly the reference order. Nothing is
// reassociated; the reduction axis never crosses a lane. The final f64→f32 narrow
// is FCVTN (round-to-nearest-even under the default FPCR), the same rounding as
// Go's float32(f64) FCVTDS. The gates are TestMatmulAVAcc64_exactMatchesStrided and
// TestMatmulQKAcc64_* (raw-bit equality against independent kernels), plus the
// NEON-vs-Go block tests in attn_acc64_arm64_test.go.
//
// AV, per key: the score is loaded as f32 and widened once (FCVTSD) into V31.D[0];
// the 32-dim slice of the V row is loaded as eight f32 quads (two 4-register LD1),
// each quad widened with FCVTL/FCVTL2 into two 2-lane f64 vectors and folded into
// its two accumulators with an FMLA-by-element against V31.D[0]. Sixteen
// accumulators (V0–V15) hold 32 dims; per key that is 2 loads, 1 scalar widen, 16
// converts and 16 FMLAs for 32 MACs — ~1.1 SIMD µops per MAC against the Go
// kernel's ~3 instructions per MAC (load, convert, FMADD) with a memory round-trip
// per 16-block pass. hd%32 dims fall to the Go 16-block and tail code.
//
// QK, per d-quad (4 dims): q's quad is widened into V28 = [q_d0,q_d1] and
// V29 = [q_d2,q_d3]. For each of 8 key pairs (16 keys, accumulators V0–V7, lane =
// key): load each key's quad, ZIP1/ZIP2 to pair the two keys per dim, FCVTL/FCVTL2
// to four f64 vectors [k0_d,k1_d], and four FMLA-by-element against q_d0..q_d3 in
// ascending d. Each lane's accumulator sees d = 0,1,2,3 in order within the quad
// and quads in ascending order across the loop — the reference chain. N%16 keys
// fall to the Go 8-chain loop. Row pointers live in 16 GPRs (R18 skipped: it is
// the platform register on darwin).
//
// All of FCVTL/FCVTL2/FCVTN/FCVTN2/ZIP1/ZIP2 (vector) and FMLA (by element, .2D)
// lack Go arm64 assembler mnemonics and are raw WORDs, same convention as the
// other kernels here. Base ARMv8-A NEON: no feature detection.
//
// Encodings (Rn<<5 | Rd; Rm<<16 or M:Rm for the by-element form):
//   FCVTL   Vd.2D, Vn.2S            = 0x0E617800
//   FCVTL2  Vd.2D, Vn.4S            = 0x4E617800
//   FCVTN   Vd.2S, Vn.2D            = 0x0E616800
//   FCVTN2  Vd.4S, Vn.2D            = 0x4E616800
//   ZIP1    Vd.4S, Vn.4S, Vm.4S     = 0x4E803800
//   ZIP2    Vd.4S, Vn.4S, Vm.4S     = 0x4E807800
//   FMLA    Vd.2D, Vn.2D, Vm.D[i]   = 0x4FC01000 | M<<20 | Rm<<16 | i<<11

#include "textflag.h"

// func avAcc64NEON32(scores *float32, vals *float32, nKeys int, rowStrideBytes int, dst *float32)
TEXT ·avAcc64NEON32(SB), NOSPLIT, $0-40
	MOVD scores+0(FP), R0
	MOVD vals+8(FP), R1
	MOVD nKeys+16(FP), R2
	MOVD rowStrideBytes+24(FP), R3
	MOVD dst+32(FP), R4

	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	VEOR V14.B16, V14.B16, V14.B16
	VEOR V15.B16, V15.B16, V15.B16
	CBZ  R2, avstore

avkey:
	MOVWU.P 4(R0), R7          // score s as f32 bits
	FMOVS   R7, F30
	FCVTSD  F30, F31           // V31.D[0] = float64(score)
	VLD1 (R1), [V16.S4, V17.S4, V18.S4, V19.S4]   // V row dims 0..15
	ADD  $64, R1, R6
	VLD1 (R6), [V20.S4, V21.S4, V22.S4, V23.S4]   // dims 16..31
	ADD  R3, R1, R1            // next key's row
	WORD $0x0E617A18          // FCVTL  V24.2D, V16.2S        dims 0,1
	WORD $0x4E617A19          // FCVTL2 V25.2D, V16.4S        dims 2,3
	WORD $0x4FDF1300          // FMLA   V0.2D, V24.2D, V31.D[0]
	WORD $0x4FDF1321          // FMLA   V1.2D, V25.2D, V31.D[0]
	WORD $0x0E617A3A          // FCVTL  V26.2D, V17.2S        dims 4,5
	WORD $0x4E617A3B          // FCVTL2 V27.2D, V17.4S        dims 6,7
	WORD $0x4FDF1342          // FMLA   V2.2D, V26.2D, V31.D[0]
	WORD $0x4FDF1363          // FMLA   V3.2D, V27.2D, V31.D[0]
	WORD $0x0E617A58          // FCVTL  V24.2D, V18.2S        dims 8,9
	WORD $0x4E617A59          // FCVTL2 V25.2D, V18.4S        dims 10,11
	WORD $0x4FDF1304          // FMLA   V4.2D, V24.2D, V31.D[0]
	WORD $0x4FDF1325          // FMLA   V5.2D, V25.2D, V31.D[0]
	WORD $0x0E617A7A          // FCVTL  V26.2D, V19.2S        dims 12,13
	WORD $0x4E617A7B          // FCVTL2 V27.2D, V19.4S        dims 14,15
	WORD $0x4FDF1346          // FMLA   V6.2D, V26.2D, V31.D[0]
	WORD $0x4FDF1367          // FMLA   V7.2D, V27.2D, V31.D[0]
	WORD $0x0E617A98          // FCVTL  V24.2D, V20.2S        dims 16,17
	WORD $0x4E617A99          // FCVTL2 V25.2D, V20.4S        dims 18,19
	WORD $0x4FDF1308          // FMLA   V8.2D, V24.2D, V31.D[0]
	WORD $0x4FDF1329          // FMLA   V9.2D, V25.2D, V31.D[0]
	WORD $0x0E617ABA          // FCVTL  V26.2D, V21.2S        dims 20,21
	WORD $0x4E617ABB          // FCVTL2 V27.2D, V21.4S        dims 22,23
	WORD $0x4FDF134A          // FMLA   V10.2D, V26.2D, V31.D[0]
	WORD $0x4FDF136B          // FMLA   V11.2D, V27.2D, V31.D[0]
	WORD $0x0E617AD8          // FCVTL  V24.2D, V22.2S        dims 24,25
	WORD $0x4E617AD9          // FCVTL2 V25.2D, V22.4S        dims 26,27
	WORD $0x4FDF130C          // FMLA   V12.2D, V24.2D, V31.D[0]
	WORD $0x4FDF132D          // FMLA   V13.2D, V25.2D, V31.D[0]
	WORD $0x0E617AFA          // FCVTL  V26.2D, V23.2S        dims 28,29
	WORD $0x4E617AFB          // FCVTL2 V27.2D, V23.4S        dims 30,31
	WORD $0x4FDF134E          // FMLA   V14.2D, V26.2D, V31.D[0]
	WORD $0x4FDF136F          // FMLA   V15.2D, V27.2D, V31.D[0]
	SUBS $1, R2, R2
	BNE  avkey

avstore:
	WORD $0x0E616810          // FCVTN  V16.2S, V0.2D          dims 0,1 → f32
	WORD $0x4E616830          // FCVTN2 V16.4S, V1.2D          dims 2,3
	WORD $0x0E616851          // FCVTN  V17.2S, V2.2D          dims 4,5 → f32
	WORD $0x4E616871          // FCVTN2 V17.4S, V3.2D          dims 6,7
	WORD $0x0E616892          // FCVTN  V18.2S, V4.2D          dims 8,9 → f32
	WORD $0x4E6168B2          // FCVTN2 V18.4S, V5.2D          dims 10,11
	WORD $0x0E6168D3          // FCVTN  V19.2S, V6.2D          dims 12,13 → f32
	WORD $0x4E6168F3          // FCVTN2 V19.4S, V7.2D          dims 14,15
	WORD $0x0E616914          // FCVTN  V20.2S, V8.2D          dims 16,17 → f32
	WORD $0x4E616934          // FCVTN2 V20.4S, V9.2D          dims 18,19
	WORD $0x0E616955          // FCVTN  V21.2S, V10.2D          dims 20,21 → f32
	WORD $0x4E616975          // FCVTN2 V21.4S, V11.2D          dims 22,23
	WORD $0x0E616996          // FCVTN  V22.2S, V12.2D          dims 24,25 → f32
	WORD $0x4E6169B6          // FCVTN2 V22.4S, V13.2D          dims 26,27
	WORD $0x0E6169D7          // FCVTN  V23.2S, V14.2D          dims 28,29 → f32
	WORD $0x4E6169F7          // FCVTN2 V23.4S, V15.2D          dims 30,31
	VST1.P [V16.S4, V17.S4, V18.S4, V19.S4], 64(R4)
	VST1   [V20.S4, V21.S4, V22.S4, V23.S4], (R4)
	RET

// func qkAcc64NEON16(q *float32, rows *float32, rowStrideBytes int, k4 int, nBlocks int, dst *float32)
TEXT ·qkAcc64NEON16(SB), NOSPLIT, $0-48
	MOVD q+0(FP), R22
	MOVD rows+8(FP), R1
	MOVD rowStrideBytes+16(FP), R2
	MOVD k4+24(FP), R23
	MOVD nBlocks+32(FP), R24
	MOVD dst+40(FP), R4

qkblock:
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	MOVD R1, R5                 // 16 row pointers, rowStride apart
	ADD  R2, R5, R6
	ADD  R2, R6, R7
	ADD  R2, R7, R8
	ADD  R2, R8, R9
	ADD  R2, R9, R10
	ADD  R2, R10, R11
	ADD  R2, R11, R12
	ADD  R2, R12, R13
	ADD  R2, R13, R14
	ADD  R2, R14, R15
	ADD  R2, R15, R16
	ADD  R2, R16, R17
	ADD  R2, R17, R19
	ADD  R2, R19, R20
	ADD  R2, R20, R21
	MOVD R22, R0               // q, from the start
	MOVD R23, R3               // K/4 quads

qkquad:
	VLD1.P 16(R0), [V24.S4]    // q[d..d+3]
	WORD $0x0E617B1C          // FCVTL  V28.2D, V24.2S   [q_d0, q_d1]
	WORD $0x4E617B1D          // FCVTL2 V29.2D, V24.4S   [q_d2, q_d3]
	// keys 0,1 → V0
	VLD1.P 16(R5), [V16.S4]
	VLD1.P 16(R6), [V17.S4]
	WORD $0x4E913A12          // ZIP1   V18.4S, V16.4S, V17.4S  [k0d0,k1d0,k0d1,k1d1]
	WORD $0x4E917A13          // ZIP2   V19.4S, V16.4S, V17.4S  [k0d2,k1d2,k0d3,k1d3]
	WORD $0x0E617A54          // FCVTL  V20.2D, V18.2S
	WORD $0x4E617A55          // FCVTL2 V21.2D, V18.4S
	WORD $0x0E617A76          // FCVTL  V22.2D, V19.2S
	WORD $0x4E617A77          // FCVTL2 V23.2D, V19.4S
	WORD $0x4FDC1280          // FMLA   V0.2D, V20.2D, V28.D[0]
	WORD $0x4FDC1AA0          // FMLA   V0.2D, V21.2D, V28.D[1]
	WORD $0x4FDD12C0          // FMLA   V0.2D, V22.2D, V29.D[0]
	WORD $0x4FDD1AE0          // FMLA   V0.2D, V23.2D, V29.D[1]
	// keys 2,3 → V1
	VLD1.P 16(R7), [V16.S4]
	VLD1.P 16(R8), [V17.S4]
	WORD $0x4E913A12          // ZIP1   V18.4S, V16.4S, V17.4S  [k0d0,k1d0,k0d1,k1d1]
	WORD $0x4E917A13          // ZIP2   V19.4S, V16.4S, V17.4S  [k0d2,k1d2,k0d3,k1d3]
	WORD $0x0E617A54          // FCVTL  V20.2D, V18.2S
	WORD $0x4E617A55          // FCVTL2 V21.2D, V18.4S
	WORD $0x0E617A76          // FCVTL  V22.2D, V19.2S
	WORD $0x4E617A77          // FCVTL2 V23.2D, V19.4S
	WORD $0x4FDC1281          // FMLA   V1.2D, V20.2D, V28.D[0]
	WORD $0x4FDC1AA1          // FMLA   V1.2D, V21.2D, V28.D[1]
	WORD $0x4FDD12C1          // FMLA   V1.2D, V22.2D, V29.D[0]
	WORD $0x4FDD1AE1          // FMLA   V1.2D, V23.2D, V29.D[1]
	// keys 4,5 → V2
	VLD1.P 16(R9), [V16.S4]
	VLD1.P 16(R10), [V17.S4]
	WORD $0x4E913A12          // ZIP1   V18.4S, V16.4S, V17.4S  [k0d0,k1d0,k0d1,k1d1]
	WORD $0x4E917A13          // ZIP2   V19.4S, V16.4S, V17.4S  [k0d2,k1d2,k0d3,k1d3]
	WORD $0x0E617A54          // FCVTL  V20.2D, V18.2S
	WORD $0x4E617A55          // FCVTL2 V21.2D, V18.4S
	WORD $0x0E617A76          // FCVTL  V22.2D, V19.2S
	WORD $0x4E617A77          // FCVTL2 V23.2D, V19.4S
	WORD $0x4FDC1282          // FMLA   V2.2D, V20.2D, V28.D[0]
	WORD $0x4FDC1AA2          // FMLA   V2.2D, V21.2D, V28.D[1]
	WORD $0x4FDD12C2          // FMLA   V2.2D, V22.2D, V29.D[0]
	WORD $0x4FDD1AE2          // FMLA   V2.2D, V23.2D, V29.D[1]
	// keys 6,7 → V3
	VLD1.P 16(R11), [V16.S4]
	VLD1.P 16(R12), [V17.S4]
	WORD $0x4E913A12          // ZIP1   V18.4S, V16.4S, V17.4S  [k0d0,k1d0,k0d1,k1d1]
	WORD $0x4E917A13          // ZIP2   V19.4S, V16.4S, V17.4S  [k0d2,k1d2,k0d3,k1d3]
	WORD $0x0E617A54          // FCVTL  V20.2D, V18.2S
	WORD $0x4E617A55          // FCVTL2 V21.2D, V18.4S
	WORD $0x0E617A76          // FCVTL  V22.2D, V19.2S
	WORD $0x4E617A77          // FCVTL2 V23.2D, V19.4S
	WORD $0x4FDC1283          // FMLA   V3.2D, V20.2D, V28.D[0]
	WORD $0x4FDC1AA3          // FMLA   V3.2D, V21.2D, V28.D[1]
	WORD $0x4FDD12C3          // FMLA   V3.2D, V22.2D, V29.D[0]
	WORD $0x4FDD1AE3          // FMLA   V3.2D, V23.2D, V29.D[1]
	// keys 8,9 → V4
	VLD1.P 16(R13), [V16.S4]
	VLD1.P 16(R14), [V17.S4]
	WORD $0x4E913A12          // ZIP1   V18.4S, V16.4S, V17.4S  [k0d0,k1d0,k0d1,k1d1]
	WORD $0x4E917A13          // ZIP2   V19.4S, V16.4S, V17.4S  [k0d2,k1d2,k0d3,k1d3]
	WORD $0x0E617A54          // FCVTL  V20.2D, V18.2S
	WORD $0x4E617A55          // FCVTL2 V21.2D, V18.4S
	WORD $0x0E617A76          // FCVTL  V22.2D, V19.2S
	WORD $0x4E617A77          // FCVTL2 V23.2D, V19.4S
	WORD $0x4FDC1284          // FMLA   V4.2D, V20.2D, V28.D[0]
	WORD $0x4FDC1AA4          // FMLA   V4.2D, V21.2D, V28.D[1]
	WORD $0x4FDD12C4          // FMLA   V4.2D, V22.2D, V29.D[0]
	WORD $0x4FDD1AE4          // FMLA   V4.2D, V23.2D, V29.D[1]
	// keys 10,11 → V5
	VLD1.P 16(R15), [V16.S4]
	VLD1.P 16(R16), [V17.S4]
	WORD $0x4E913A12          // ZIP1   V18.4S, V16.4S, V17.4S  [k0d0,k1d0,k0d1,k1d1]
	WORD $0x4E917A13          // ZIP2   V19.4S, V16.4S, V17.4S  [k0d2,k1d2,k0d3,k1d3]
	WORD $0x0E617A54          // FCVTL  V20.2D, V18.2S
	WORD $0x4E617A55          // FCVTL2 V21.2D, V18.4S
	WORD $0x0E617A76          // FCVTL  V22.2D, V19.2S
	WORD $0x4E617A77          // FCVTL2 V23.2D, V19.4S
	WORD $0x4FDC1285          // FMLA   V5.2D, V20.2D, V28.D[0]
	WORD $0x4FDC1AA5          // FMLA   V5.2D, V21.2D, V28.D[1]
	WORD $0x4FDD12C5          // FMLA   V5.2D, V22.2D, V29.D[0]
	WORD $0x4FDD1AE5          // FMLA   V5.2D, V23.2D, V29.D[1]
	// keys 12,13 → V6
	VLD1.P 16(R17), [V16.S4]
	VLD1.P 16(R19), [V17.S4]
	WORD $0x4E913A12          // ZIP1   V18.4S, V16.4S, V17.4S  [k0d0,k1d0,k0d1,k1d1]
	WORD $0x4E917A13          // ZIP2   V19.4S, V16.4S, V17.4S  [k0d2,k1d2,k0d3,k1d3]
	WORD $0x0E617A54          // FCVTL  V20.2D, V18.2S
	WORD $0x4E617A55          // FCVTL2 V21.2D, V18.4S
	WORD $0x0E617A76          // FCVTL  V22.2D, V19.2S
	WORD $0x4E617A77          // FCVTL2 V23.2D, V19.4S
	WORD $0x4FDC1286          // FMLA   V6.2D, V20.2D, V28.D[0]
	WORD $0x4FDC1AA6          // FMLA   V6.2D, V21.2D, V28.D[1]
	WORD $0x4FDD12C6          // FMLA   V6.2D, V22.2D, V29.D[0]
	WORD $0x4FDD1AE6          // FMLA   V6.2D, V23.2D, V29.D[1]
	// keys 14,15 → V7
	VLD1.P 16(R20), [V16.S4]
	VLD1.P 16(R21), [V17.S4]
	WORD $0x4E913A12          // ZIP1   V18.4S, V16.4S, V17.4S  [k0d0,k1d0,k0d1,k1d1]
	WORD $0x4E917A13          // ZIP2   V19.4S, V16.4S, V17.4S  [k0d2,k1d2,k0d3,k1d3]
	WORD $0x0E617A54          // FCVTL  V20.2D, V18.2S
	WORD $0x4E617A55          // FCVTL2 V21.2D, V18.4S
	WORD $0x0E617A76          // FCVTL  V22.2D, V19.2S
	WORD $0x4E617A77          // FCVTL2 V23.2D, V19.4S
	WORD $0x4FDC1287          // FMLA   V7.2D, V20.2D, V28.D[0]
	WORD $0x4FDC1AA7          // FMLA   V7.2D, V21.2D, V28.D[1]
	WORD $0x4FDD12C7          // FMLA   V7.2D, V22.2D, V29.D[0]
	WORD $0x4FDD1AE7          // FMLA   V7.2D, V23.2D, V29.D[1]
	SUBS $1, R3, R3
	BNE  qkquad

	WORD $0x0E616810          // FCVTN  V16.2S, V0.2D          keys 0,1 → f32
	WORD $0x4E616830          // FCVTN2 V16.4S, V1.2D          keys 2,3
	WORD $0x0E616851          // FCVTN  V17.2S, V2.2D          keys 4,5 → f32
	WORD $0x4E616871          // FCVTN2 V17.4S, V3.2D          keys 6,7
	WORD $0x0E616892          // FCVTN  V18.2S, V4.2D          keys 8,9 → f32
	WORD $0x4E6168B2          // FCVTN2 V18.4S, V5.2D          keys 10,11
	WORD $0x0E6168D3          // FCVTN  V19.2S, V6.2D          keys 12,13 → f32
	WORD $0x4E6168F3          // FCVTN2 V19.4S, V7.2D          keys 14,15
	VST1.P [V16.S4, V17.S4, V18.S4, V19.S4], 64(R4)
	ADD  R2<<4, R1, R1         // next block of 16 rows
	SUBS $1, R24, R24
	BNE  qkblock
	RET
