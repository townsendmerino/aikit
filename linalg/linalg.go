// Package linalg holds aikit's shared SIMD compute kernels: the dot-product
// kernels (AVX2+FMA on amd64, NEON on arm64, scalar elsewhere — selected by
// build tag and runtime CPU detection) and a row-parallel float32 matmul built
// on them. It is the single home for the hand-written assembly so the encoder
// and goinfer's LLM decoder share one copy.
package linalg

import (
	"runtime"
	"sync"
)

// Dot returns Σ a[i]*b[i]. len(a) must equal len(b).
func Dot(a, b []float32) float32 { return dotF32(a, b) }

// Dot4x4 / Dot8x4 are the register-blocked micro-kernels: each computes 4 / 8
// dot products of the shared row `a` against consecutive b-rows, writing every
// row's full sum into the first lane of its 4-lane block in sums (the rest
// zero). n4 is len/4 (K is a multiple of 4 in the matmul hot path). Exposed
// for the encoder's cache-blocked matmul; the decoder uses MatmulBT/Dot.
func Dot4x4(a, b0, b1, b2, b3 *float32, n4 int, sums *[16]float32) {
	dotNEON4x4(a, b0, b1, b2, b3, n4, sums)
}

func Dot8x4(a, b0, b1, b2, b3, b4, b5, b6, b7 *float32, n4 int, sums *[32]float32) {
	dotNEON8x4(a, b0, b1, b2, b3, b4, b5, b6, b7, n4, sums)
}

// parThreshold is the MAC count (M*N*K) below which a matmul runs serially —
// under it the goroutine fork/join costs more than the parallelism saves. It is
// set high enough that the M=1 single-token decode projections (≤ ~9M MACs for
// even a fused gate+up on a 0.5B–1.5B model) stay serial: goinfer's profile
// showed those tiny per-call fan-outs spending ~70% of decode CPU in
// pthread_cond park/wake for no speedup. Prompt/prefill (large M) and the
// encoder (large M) clear it comfortably and still parallelize.
//
// SetParallelThreshold tunes it (see SetParallelThreshold).
var parThreshold = 1 << 24 // 16.78M MACs

// SetParallelThreshold sets the MAC count (M*N*K) at/above which matmuls
// parallelize across goroutines; below it they run serially. The default
// (16.78M) keeps M=1 single-token decode serial — the regime where per-call
// fork/join dominates — while prompt/prefill and the encoder still parallelize.
//
// It's a process-wide knob for callers tuning a specific workload + machine
// against an end-to-end benchmark (a microbenchmark of back-to-back matmuls
// can't reproduce the cold goroutine park/wake that makes serial win in a real
// decode loop). Set it once at startup, before concurrent matmuls run; it is
// not safe to change while matmuls are in flight. n ≤ 0 forces always-parallel.
func SetParallelThreshold(macs int) { parThreshold = macs }

// ParallelThreshold reports the current threshold (see SetParallelThreshold).
func ParallelThreshold() int { return parThreshold }

// parWidth caps the number of worker shards a parallelized matmul fans out to;
// 0 means GOMAXPROCS (the default). See SetParallelWidth.
var parWidth = 0

// SetParallelWidth caps how many worker shards a parallel matmul fans out to
// (0 = use GOMAXPROCS, the default). Orthogonal to SetParallelThreshold: the
// threshold decides *whether* to parallelize, the width decides into *how many*
// shards. Lower it to avoid slow-core stragglers at the fork/join barrier on
// heterogeneous CPUs (Apple big.LITTLE, Intel P/E): a barrier waits on its
// slowest shard, and an E-core handed an equal 1/N slice finishes well after a
// P-core. Fewer shards tighten the join — but pure Go can't pin goroutines to
// P vs E cores, so this only *lowers the odds* of an E-core straggler; it's a
// statistical win, not a guarantee.
//
// Numerically inert: parallel matmuls partition output COLUMNS, so each output
// is computed by one worker doing the full K-reduction — any width is
// bit-identical. Process-wide; set once at startup. The effective worker count
// is min(width-or-GOMAXPROCS, GOMAXPROCS, columns).
func SetParallelWidth(n int) { parWidth = n }

// ParallelWidth reports the current fan-out width cap (0 = GOMAXPROCS).
func ParallelWidth() int { return parWidth }

// effectiveWidth resolves the configured width against GOMAXPROCS.
func effectiveWidth() int {
	g := runtime.GOMAXPROCS(0)
	if parWidth <= 0 || parWidth > g {
		return g
	}
	return parWidth
}

// MatmulBT computes dst[M,N] = a[M,K] · b[N,K]ᵀ — the PyTorch [out,in] weight
// layout the safetensors checkpoints store, so no transpose copy is needed.
// Each output is a Dot of an a-row against a b-row; work is parallelized over
// the N output columns (always large in the transformer projections, and the
// only dimension with parallelism on the M=1 single-token decode path).
func MatmulBT(a, b, dst []float32, M, K, N int) {
	parallelCols(M*N*K, N, func(j0, j1 int) {
		for i := range M {
			arow := a[i*K : i*K+K]
			drow := dst[i*N : i*N+N]
			for j := j0; j < j1; j++ {
				drow[j] = dotF32(arow, b[j*K:j*K+K])
			}
		}
	})
}

// parallelCols runs fn over the [0,N) output columns, split into one chunk per
// worker. Serial below parThreshold MACs (work) where the goroutine fan-out
// would cost more than it saves. Both the f32 and int8 matmuls share it.
func parallelCols(work, N int, fn func(j0, j1 int)) {
	if work < parThreshold || N < 2 {
		fn(0, N)
		return
	}
	parallelSpawnCols(N, fn)
}

// parallelSpawnCols splits [0,N) into one chunk per worker and runs fn on each
// in its own goroutine. The caller has already decided parallelism is worth it
// (so the fn closure's heap escape here is paid only on the parallel path —
// callers that want a zero-alloc serial path call their span function directly
// when below threshold, rather than routing a closure through parallelCols).
func parallelSpawnCols(N int, fn func(j0, j1 int)) {
	workers := min(effectiveWidth(), N)
	chunk := (N + workers - 1) / workers
	var wg sync.WaitGroup
	for j0 := 0; j0 < N; j0 += chunk {
		j1 := min(j0+chunk, N)
		wg.Add(1)
		go func(j0, j1 int) {
			defer wg.Done()
			fn(j0, j1)
		}(j0, j1)
	}
	wg.Wait()
}
