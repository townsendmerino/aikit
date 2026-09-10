package encoder

import (
	"github.com/townsendmerino/aikit/linalg"
)

// matmulBT computes dst = a · bᵀ where:
//
//	a: [M, K] row-major
//	b: [N, K] row-major (already transposed relative to the matmul —
//	   this matches PyTorch's [out, in] weight layout, so calls like
//	   `h · Wᵀ` don't need a transpose copy)
//	dst: [M, N] row-major, freshly allocated
//
// Accumulators are float32 (plan §7: f32 GEMM is acceptable when the
// acceptance gate is end-to-end NDCG; cosine-bar loose). For
// reductions where parity matters (LayerNorm, softmax, final L2)
// callers should use float64 explicitly.
//
// ALL shapes route through the blocked kernel. There used to be a 4-MFLOP
// threshold below which this took the i-n-k naive path, on the theory that it
// beat the blocked kernel's prologue at tiny tiles. Both halves of that were
// wrong (perf-campaign items 27 + 41):
//
//   - It was slower, not faster, at every shape it diverted. Measured on a
//     3700X by BenchmarkNaiveVsBlocked — per-head attention QKᵀ at L=128
//     528→99 µs (5.3×), scores·V at L=250 2649→197 µs (13.5×). The threshold
//     sent EVERY attention matmul below L≈250 to a scalar triple loop.
//   - It made the reduction order observable. The naive span accumulates in a
//     different order than the blocked kernel's k-tiling, so which side of the
//     threshold a shape landed on changed the output: 13489 of 16384 elements
//     differ at M=128,K=64,N=128, worst |Δ| 9.5e-06. linalg deleted its own
//     copy of this threshold for exactly that reason (see MatmulBT's
//     M-INVARIANT note) and measured the same speed result; the encoder kept a
//     private one. Removing it also removes the last source of batch-vs-single
//     divergence in forwardBatch.
func matmulBT(a, b []float32, M, K, N int) []float32 {
	return matmulBTBlocked(a, b, M, K, N)
}

// matmulBTNaive is the correctness-first baseline, kept as the reference the
// blocked kernel is tested against (TestNaiveVsBlocked_reductionOrderDiffers,
// linalg's own equivalence tests). It is NOT on any production path — see
// matmulBT for why the threshold that used to select it is gone. Loop order
// i-n-k: the inner reduction reads a[i,:] and b[n,:] sequentially, both
// contiguous in row-major.
func matmulBTNaive(a, b []float32, M, K, N int) []float32 {
	dst := make([]float32, M*N)
	for i := range M {
		aRow := a[i*K : (i+1)*K]
		for n := range N {
			bRow := b[n*K : (n+1)*K]
			var s float32
			for k := range K {
				s += aRow[k] * bRow[k]
			}
			dst[i*N+n] = s
		}
	}
	return dst
}

// matmulBTBlockedInto runs the shared linalg cache+register-blocked GEMM into a
// caller-provided dst (len ≥ M*N), serially. The encoder owns its parallelism
// (EncodeBatch at the document level, lone forwards via matmulBTBlockedIntoParallel),
// so it wants the serial kernel; linalg.MatmulBTInto zeroes dst and fills it.
func matmulBTBlockedInto(a, b, dst []float32, M, K, N int) {
	linalg.MatmulBTInto(dst, a, b, M, K, N)
}

// matmulBTInto is matmulBT into a caller-provided dst, plus the intra-op
// parallelism dispatch. Like matmulBT it has no naive path — see there.
func matmulBTInto(a, b, dst []float32, M, K, N int) {
	// Intra-op parallelism kicks in only for a lone forward with a large
	// enough shape. Under EncodeBatch's per-worker parallelism both gates
	// are false, so the batch path stays on the serial blocked fill.
	//
	// COLUMNS FIRST: a column split partitions the weights across workers
	// while a row split replicates them, and it is not capped by the row
	// count. Measured better at every trunk shape — see wantParallelCols.
	if wantParallelCols(M, K, N) {
		matmulBTColsInto(a, b, dst, M, K, N)
		return
	}
	// Fallback for outputs too narrow to shard by column.
	if wantParallelMatmul(M, K, N) {
		matmulBTBlockedIntoParallel(a, b, dst, M, K, N)
		return
	}
	matmulBTBlockedInto(a, b, dst, M, K, N)
}

func matmulBTNaiveInto(a, b, dst []float32, M, K, N int) {
	if len(dst) < M*N {
		panic("encoder: matmulBTNaiveInto dst too small")
	}
	for i := range M {
		aRow := a[i*K : (i+1)*K]
		for n := range N {
			bRow := b[n*K : (n+1)*K]
			var s float32
			for k := range K {
				s += aRow[k] * bRow[k]
			}
			dst[i*N+n] = s
		}
	}
}

func zeroF32Slice(s []float32) {
	for i := range s {
		s[i] = 0
	}
}

// matmulBTBlocked allocates dst and runs the shared linalg blocked GEMM into it. The
// tiling + register kernels (cache tiles + Dot8x4/Dot2x8) live in linalg now — one
// implementation behind the encoder, the public linalg.MatmulBT, and other kit
// consumers; see linalg/matmul_blocked.go.
func matmulBTBlocked(a, b []float32, M, K, N int) []float32 {
	dst := make([]float32, M*N)
	linalg.MatmulBTInto(dst, a, b, M, K, N)
	return dst
}

// matmulAdd is matmulBT + an additive bias broadcast across rows:
// dst[i, n] += bias[n]. No-op when bias is nil. Kept separate from
// matmulBT so the hot, no-bias path (all CodeRankEmbed weights have
// no bias) stays tight.
func addBias(dst []float32, bias []float32, M, N int) {
	if bias == nil {
		return
	}
	for i := range M {
		row := dst[i*N : (i+1)*N]
		for n := range N {
			row[n] += bias[n]
		}
	}
}

// softmaxRow normalizes row in-place along its last axis, using
// max-subtraction for stability and a float64 accumulator for the sum
// (plan §6: softmax accumulates in f64).
//
// The exponentials themselves are float32 (linalg.SoftmaxRowInto), NOT Go's
// float64 math.Exp. That is perf-campaign item 13, and it is the one change in
// this campaign that is deliberately NOT bit-identical: math.Exp is a branchy
// scalar f64 routine that cannot inline into this loop, and attention calls it
// O(L²) times per head per layer. linalg's kernel is accurate to 0.68 ULP of
// float32 (gated by TestExpF32_accuracy over 3.8M points) and softmax then
// normalizes, which cancels most of the residual — the encoders' HF parity
// goldens are the end-to-end check that this stays invisible.
func softmaxRow(row []float32) {
	linalg.SoftmaxRowContractInto(row, row)
}

// softmaxRowScaled is softmaxRow fused with the attention score scale
// (1/sqrt(headDim)) — see linalg.SoftmaxRowScaledInto for why this replaces
// a caller's own separate `scores[i] *= scale` pass bit-identically.
func softmaxRowScaled(row []float32, scale float32) {
	linalg.SoftmaxRowScaledContractInto(row, row, scale)
}

// silu returns x · sigmoid(x), the SwiGLU gate activation.
//
// linalg's float32 kernel (perf-campaign item 13), not f64 math.Exp. The
// saturation the old f64 comment worried about is handled by ExpF32's own
// overflow/underflow cases, and the accuracy contract is absolute error ≤1e-06
// — see linalg.SiLUF32.
func silu(x float32) float32 {
	return linalg.SiLUF32(x)
}
