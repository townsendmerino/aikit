//go:build arm64

package linalg

// The eight-column form of q8Span for arm64 (task-simd-audit.md S-07).
//
// Weight-only int8 (MatmulBTQ8: `--quant int8` in goinfer, and the LM head before
// it moved to W8A8) ran every output through dotNEON4 — one FMLA accumulator, so
// one lane-vector per ~4-cycle FMLA latency: 1 MAC per cycle, 3.2 GMAC/s per
// core. That single chain, not bandwidth and not allocation, was the mechanism
// behind the LM head's 11–13 GB/s at 6–8 workers.
//
// dotNEON8x4 already exists (the f32 GEMM's 1×8 micro-kernel): it loads a once
// and runs eight independent FMLA chains against eight weight rows, so it is
// load-bound at ~3 FMLAs/cycle instead of latency-bound at 1/4. This span
// widens eight weight rows into an 8×K scratch and feeds them to it, per
// activation row — ~4× per row including the widen, more at M>1 where the
// widen amortizes.
//
// BIT-IDENTICAL to the single-column form (q8SpanColumn, what every arch ran
// before and what !arm64 still runs), by construction:
//
//   - the widen is the same dequantRowInt8(…, 1.0) per row — exact;
//   - per weight row r, dotNEON8x4's accumulator V_r receives exactly the
//     sequence dotNEON4's V0 receives: lane l += a[4i+l]·b_r[4i+l] for i
//     ascending, one fused multiply-add per step (dot_arm64.s, both kernels);
//   - the four lanes are folded left to right, s0+s1+s2+s3, in both
//     (dotNEON in dot_arm64.go; dot8ColsGeneric in dot8cols.go, which is
//     dot8ColsInto on arm64);
//   - the K%4 tail is added after the fold in the same order with the same
//     `s += a[k] * b[k]` form dotF32 uses (the arm64 compiler fuses it to
//     FMADD in both places);
//   - the row scale multiplies last, unchanged.
//
// So each dst[i,j] is the same float32 expression as before; only which
// accumulators share a loop changes. TestQ8Span8Cols_bitIdenticalToColumnForm
// holds it against q8SpanColumn over the tails and remainders; the older
// TestQ8Span_bitIdenticalToScalarWiden holds MatmulBTQ8 against the scalar
// definition and stays green unchanged.

// q8SpanScratchRows: the span widens eight weight rows at a time.
const q8SpanScratchRows = 8

func q8Span(a []float32, bQ []int8, bScales, dst []float32, M, K, N, j0, j1 int, deq []float32) {
	n4 := K / 4
	tail := n4 * 4
	j := j0
	if n4 > 0 {
		_ = deq[8*K-1]
		for ; j+8 <= j1; j += 8 {
			for r := range 8 {
				dequantRowInt8(deq[r*K:r*K+K], bQ[(j+r)*K:(j+r+1)*K], 1.0)
			}
			d0, d1, d2, d3 := deq[0:K], deq[K:2*K], deq[2*K:3*K], deq[3*K:4*K]
			d4, d5, d6, d7 := deq[4*K:5*K], deq[5*K:6*K], deq[6*K:7*K], deq[7*K:8*K]
			for i := range M {
				arow := a[i*K : i*K+K]
				var out [8]float32
				dot8ColsInto(&arow[0], &d0[0], &d1[0], &d2[0], &d3[0], &d4[0], &d5[0], &d6[0], &d7[0], n4, &out)
				drow := dst[i*N+j : i*N+j+8]
				for r := range 8 {
					s := out[r]
					if tail < K {
						b := deq[r*K : r*K+K]
						for k := tail; k < K; k++ {
							s += arow[k] * b[k]
						}
					}
					drow[r] = s * bScales[j+r]
				}
			}
		}
	}
	// Fewer than eight columns left (or K < 4): the column form, which is the
	// definition.
	for ; j < j1; j++ {
		q8SpanColumn(a, bQ, bScales, dst, M, K, N, j, deq[:K])
	}
}
