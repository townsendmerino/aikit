package linalg

// MatmulAVAcc64 computes dst[M,hd] = scores[M,nKeys] · V_head[nKeys,hd], where
// V_head is one KV head's slice of a row-major [nKeys, rowStride] buffer
// (rowStride is the full kvDim; headOff selects this head's hd-wide column
// range within each row) — the attention scores·V step, f64-accumulated.
//
// WHY THIS EXISTS, over MatmulBTAcc64Strided (attention's other current option).
// That kernel's strided accessor reads b[j][k] = vals[headOff + j + k*rowStride]
// — for a FIXED output dim j, it walks k=0..nKeys-1 at stride rowStride (one
// full V row apart, ~1KB at typical hd), i.e. one cache line touched per f64
// MAC, repeated for every one of the hd output dims. This kernel instead reads
// each key's V row ONCE, contiguously (rowStride-1 wasted floats aside, the
// hd-wide head slice itself is contiguous), and folds it into hd INDEPENDENT
// f64 accumulators — one per output dim — in one pass over the keys.
//
// BIT-IDENTICAL BY CONSTRUCTION, not by parity test. Each output dim's
// accumulator receives EXACTLY the same sequence of adds, in the exact same
// key-ascending order, as MatmulBTAcc64Strided's per-dim reduction: only the
// loop NESTING changes (keys-outer/dims-inner instead of dims-outer/keys-
// inner), and floating-point addition depends on the order of operations
// applied to ONE accumulator, not on what unrelated accumulators do between
// those operations. This is the "split the independent axis, never the
// reduction" principle (docs/task-decode-splitkv-attention.md in goinfer),
// applied by loop-nest interchange rather than parallelism — but it composes
// with parallelism too: MatmulAVAcc64PerQuery below runs each query row's own
// hd-accumulator sweep independently, so distributing queries (or, in the
// decode caller, heads) across goroutines touches none of this ordering.
//
// acc is caller-provided [hd]-float64 scratch (steady-state decode calls this
// hundreds of times per token; a fresh make([]float64, hd) per call would be
// real allocation pressure). Zeroed on entry, not preserved on exit.
func MatmulAVAcc64(scores, vals, dst []float32, acc []float64, M, nKeys, hd, headOff, rowStride int) {
	checkMatmulAVAcc64(len(scores), len(vals), len(dst), len(acc), M, nKeys, hd, headOff, rowStride)
	for i := range M {
		srow := scores[i*nKeys : i*nKeys+nKeys]
		// REGISTER-BLOCKED OVER THE OUTPUT DIMS (task-simd-audit.md S-04, step 1).
		//
		// The obvious loop — `for s { for d { acc[d] += w * float64(vrow[d]) } }` —
		// keeps the accumulators in MEMORY, because acc is a slice. Every MAC then
		// costs a load of acc[d], a load of vrow[d], a convert, an FMADD, a store
		// back to acc[d], and index arithmetic: ~7 instructions to do one multiply
		// and one add. The audit measured the consequence as ~1.86 GMAC/s, roughly
		// 40% of what the QK kernel next door achieves.
		//
		// Blocking fixes that without any assembly. Sixteen NAMED f64 locals are
		// register-allocatable, so within a block the accumulator never touches
		// memory: the inner statement becomes a load, a convert and an FMADD.
		//
		// BIT-IDENTICAL, and by the same argument this file already makes for its
		// loop-nest interchange. Each output dim's accumulator still receives
		// exactly the same adds, in the same key-ascending order — only the nesting
		// changes (dim-block outer, keys inner). Floating-point addition depends on
		// the order of operations applied to ONE accumulator, not on what unrelated
		// accumulators do in between. TestMatmulAVAcc64_exactMatchesStrided is the
		// gate and it compares against the independent strided kernel, so this is
		// checked rather than argued.
		//
		// THE COST, stated because it is real: the V slice is now read once per dim
		// block rather than once in total — hd/16 passes, 8 at the reference model's
		// hd=128. That is cheap while V is cache-resident and is the reason the
		// audit's assembly form uses 24 registers for 48 dims (3 passes) instead.
		// Whether 16 is the right block here is a measurement, not an argument, and
		// it has not been made: see the note in the test file.
		drow := dst[i*hd : i*hd+hd]
		// STEP 2 (arm64): whole 32-dim blocks go through the NEON lane-per-dim
		// kernel (attn_acc64_arm64.s) — 16 f64 lane-pair accumulators in
		// registers, the V row widened with FCVTL instead of a scalar convert
		// per element, the score widened once per key. Same adds, same
		// key-ascending order per dim; the argument below covers it unchanged,
		// and TestMatmulAVAcc64_neonMatchesGo pins the kernel against this Go
		// code block for block. Elsewhere avAcc64Blocks returns 0 and the Go
		// blocks below do everything.
		d0 := avAcc64Blocks(srow, vals, drow, nKeys, hd, headOff, rowStride)
		for ; d0+16 <= hd; d0 += 16 {
			var a0, a1, a2, a3, a4, a5, a6, a7 float64
			var a8, a9, a10, a11, a12, a13, a14, a15 float64
			for s := range nKeys {
				w := float64(srow[s])
				v := vals[headOff+s*rowStride+d0 : headOff+s*rowStride+d0+16 : headOff+s*rowStride+d0+16]
				a0 += w * float64(v[0])
				a1 += w * float64(v[1])
				a2 += w * float64(v[2])
				a3 += w * float64(v[3])
				a4 += w * float64(v[4])
				a5 += w * float64(v[5])
				a6 += w * float64(v[6])
				a7 += w * float64(v[7])
				a8 += w * float64(v[8])
				a9 += w * float64(v[9])
				a10 += w * float64(v[10])
				a11 += w * float64(v[11])
				a12 += w * float64(v[12])
				a13 += w * float64(v[13])
				a14 += w * float64(v[14])
				a15 += w * float64(v[15])
			}
			drow[d0+0], drow[d0+1], drow[d0+2], drow[d0+3] = float32(a0), float32(a1), float32(a2), float32(a3)
			drow[d0+4], drow[d0+5], drow[d0+6], drow[d0+7] = float32(a4), float32(a5), float32(a6), float32(a7)
			drow[d0+8], drow[d0+9], drow[d0+10], drow[d0+11] = float32(a8), float32(a9), float32(a10), float32(a11)
			drow[d0+12], drow[d0+13], drow[d0+14], drow[d0+15] = float32(a12), float32(a13), float32(a14), float32(a15)
		}
		// Ragged tail (hd % 16): the original memory-accumulator form, unchanged.
		// Same order, same adds — the tail is not a different kernel, just an
		// unblocked one, so it inherits the identity argument rather than needing
		// its own.
		if d0 < hd {
			for d := d0; d < hd; d++ {
				acc[d] = 0
			}
			for s := range nKeys {
				w := float64(srow[s])
				vrow := vals[headOff+s*rowStride : headOff+s*rowStride+hd]
				for d := d0; d < hd; d++ {
					acc[d] += w * float64(vrow[d])
				}
			}
			for d := d0; d < hd; d++ {
				drow[d] = float32(acc[d])
			}
		}
	}
}

func checkMatmulAVAcc64(scoresLen, valsLen, dstLen, accLen, M, nKeys, hd, headOff, rowStride int) {
	if M < 0 || nKeys < 0 || hd < 0 || headOff < 0 || rowStride < hd {
		panic("linalg: MatmulAVAcc64 invalid shape")
	}
	requireExactLen("MatmulAVAcc64", "scores", scoresLen, mul(M, nKeys))
	requireExactLen("MatmulAVAcc64", "dst", dstLen, mul(M, hd))
	if accLen < hd {
		panic("linalg: MatmulAVAcc64 acc scratch shorter than hd")
	}
	need := headOff + max(0, nKeys-1)*rowStride + hd
	if valsLen < need {
		panic("linalg: MatmulAVAcc64 vals too short for the given shape/strides")
	}
}
