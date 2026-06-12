package encoder

import (
	"math"

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
// M3 dispatch: small or unaligned shapes go straight through the
// i-n-k naive path (it's hard to beat for tiny tiles and avoids the
// blocked kernel's prologue overhead); large shapes route through
// matmulBTBlocked which is ~2-3× faster at the forward-pass shapes
// (verified by BenchmarkMatmulBT_*). The threshold is the FLOP count
// per call; tuned so the layer-7 attention QKᵀ at L=80 (~3 MFLOP) is
// naive while everything bigger is blocked.
func matmulBT(a, b []float32, M, K, N int) []float32 {
	if int64(M)*int64(K)*int64(N) < 4_000_000 {
		return matmulBTNaive(a, b, M, K, N)
	}
	return matmulBTBlocked(a, b, M, K, N)
}

// matmulBTNaive is the correctness-first baseline. Loop order i-n-k:
// inner reduction reads a[i,:] sequentially and b[n,:] sequentially —
// both contiguous in row-major.
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

// matmulBTInto dispatches by FLOP count, same as matmulBT, into a
// caller-provided dst.
func matmulBTInto(a, b, dst []float32, M, K, N int) {
	if int64(M)*int64(K)*int64(N) < 4_000_000 {
		matmulBTNaiveInto(a, b, dst, M, K, N)
		return
	}
	// Intra-op row parallelism kicks in only for a lone forward with a
	// large enough shape (see wantParallelMatmul). Under EncodeBatch's
	// per-worker parallelism this is false, so the batch path stays on
	// the serial blocked fill.
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
// max-subtraction for stability and float64 accumulators (plan §6:
// softmax accumulates in f64). row is modified in place.
func softmaxRow(row []float32) {
	if len(row) == 0 {
		return
	}
	// max for stability
	maxV := row[0]
	for _, v := range row[1:] {
		if v > maxV {
			maxV = v
		}
	}
	// sum of exp in f64
	var sum float64
	for i, v := range row {
		e := math.Exp(float64(v - maxV))
		row[i] = float32(e)
		sum += e
	}
	if sum == 0 {
		// degenerate — all-equal-to-zero output keeps the vector finite
		for i := range row {
			row[i] = 1.0 / float32(len(row))
		}
		return
	}
	inv := float32(1.0 / sum)
	for i := range row {
		row[i] *= inv
	}
}

// silu returns x · sigmoid(x), the SwiGLU gate activation. f64
// internally to avoid float32 saturation on large |x|.
func silu(x float32) float32 {
	xf := float64(x)
	return float32(xf / (1.0 + math.Exp(-xf)))
}
