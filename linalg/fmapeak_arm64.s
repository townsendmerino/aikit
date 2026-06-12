#include "textflag.h"

// func fmaPeakARM64(iters int64)
// Measures sustained single-core f32 NEON FMA throughput. 20 INDEPENDENT 4-lane FMLA
// accumulators (V0..V19) updated per iteration, with V20/V21 = 0 as the (read-only)
// multiplicands so the accumulators stay finite (acc += 0*0). No memory traffic — pure
// FP-pipe throughput. 20 in-flight FMLAs comfortably exceed 4 pipes × ~4-cycle latency,
// so the pipes saturate: this is the achievable f32 peak, not a load-bound kernel.
TEXT ·fmaPeakARM64(SB), NOSPLIT, $0-8
	MOVD	iters+0(FP), R0
	VMOVI	$0, V0.B16
	VMOVI	$0, V1.B16
	VMOVI	$0, V2.B16
	VMOVI	$0, V3.B16
	VMOVI	$0, V4.B16
	VMOVI	$0, V5.B16
	VMOVI	$0, V6.B16
	VMOVI	$0, V7.B16
	VMOVI	$0, V8.B16
	VMOVI	$0, V9.B16
	VMOVI	$0, V10.B16
	VMOVI	$0, V11.B16
	VMOVI	$0, V12.B16
	VMOVI	$0, V13.B16
	VMOVI	$0, V14.B16
	VMOVI	$0, V15.B16
	VMOVI	$0, V16.B16
	VMOVI	$0, V17.B16
	VMOVI	$0, V18.B16
	VMOVI	$0, V19.B16
	VMOVI	$0, V20.B16
	VMOVI	$0, V21.B16
	CBZ	R0, done
loop:
	VFMLA	V20.S4, V21.S4, V0.S4
	VFMLA	V20.S4, V21.S4, V1.S4
	VFMLA	V20.S4, V21.S4, V2.S4
	VFMLA	V20.S4, V21.S4, V3.S4
	VFMLA	V20.S4, V21.S4, V4.S4
	VFMLA	V20.S4, V21.S4, V5.S4
	VFMLA	V20.S4, V21.S4, V6.S4
	VFMLA	V20.S4, V21.S4, V7.S4
	VFMLA	V20.S4, V21.S4, V8.S4
	VFMLA	V20.S4, V21.S4, V9.S4
	VFMLA	V20.S4, V21.S4, V10.S4
	VFMLA	V20.S4, V21.S4, V11.S4
	VFMLA	V20.S4, V21.S4, V12.S4
	VFMLA	V20.S4, V21.S4, V13.S4
	VFMLA	V20.S4, V21.S4, V14.S4
	VFMLA	V20.S4, V21.S4, V15.S4
	VFMLA	V20.S4, V21.S4, V16.S4
	VFMLA	V20.S4, V21.S4, V17.S4
	VFMLA	V20.S4, V21.S4, V18.S4
	VFMLA	V20.S4, V21.S4, V19.S4
	SUBS	$1, R0, R0
	BNE	loop
done:
	RET
