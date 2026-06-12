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
// for a cache-blocked matmul; the decoder uses MatmulBT/Dot.
func Dot4x4(a, b0, b1, b2, b3 *float32, n4 int, sums *[16]float32) {
	dotNEON4x4(a, b0, b1, b2, b3, n4, sums)
}

// Dot8x4 reuses the shared row `a` across 8 b-rows, so it beats Dot4x4 at
// small-to-mid K (≈1.7× at K=768). But its 8 live accumulators plus the streamed
// b-rows outgrow the register/cache budget at large K, where it regresses BELOW
// Dot4x4 — measured on amd64 AVX2 at 40.5 vs 51.4 GB/s at K=3072. A cache-blocked
// caller should tile K into ≤~768-element strips and feed Dot8x4 the strips: that
// keeps it in its fast range and is the right cache-blocking move regardless.
// aikit's encoder does exactly this (kBlock=768), so it never hits the cliff; a
// caller that hands Dot8x4 a single large-K row instead should prefer Dot4x4.
func Dot8x4(a, b0, b1, b2, b3, b4, b5, b6, b7 *float32, n4 int, sums *[32]float32) {
	dotNEON8x4(a, b0, b1, b2, b3, b4, b5, b6, b7, n4, sums)
}

// Dot2x8 is the MR×NR (2 a-rows × 8 b-rows) register microkernel: it computes the 16
// dot products of two shared rows a0,a1 against eight consecutive b-rows, with 16 live
// 4-lane accumulators held across the K loop. Versus Dot8x4 (1×8), each b-load feeds 2
// FMLAs instead of 1 and the accumulator count rises from 8 to 16 — addressing the 1×8
// kernel's load- and latency-binding (see BenchmarkGEMMPeakFraction). sums holds 16
// 4-lane partial-dot blocks ([a0·b0 … a0·b7, a1·b0 … a1·b7]); the caller sums each
// block's 4 lanes and adds the K%4 scalar tail. arm64 has the NEON kernel; other arches
// get a portable reduction (the encoder wires this in only on arm64, keeping amd64 on
// the AVX2 Dot8x4 path).
func Dot2x8(a0, a1, b0, b1, b2, b3, b4, b5, b6, b7 *float32, n4 int, sums *[64]float32) {
	dotNEON2x8(a0, a1, b0, b1, b2, b3, b4, b5, b6, b7, n4, sums)
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
// It's a process-wide DEFAULT for callers tuning a specific workload + machine
// against an end-to-end benchmark (a microbenchmark of back-to-back matmuls
// can't reproduce the cold goroutine park/wake that makes serial win in a real
// decode loop). Set it once at startup, before concurrent matmuls run; it is
// not safe to change while matmuls are in flight. n ≤ 0 forces always-parallel.
//
// To override it for one workload WITHOUT mutating the global — e.g. independent
// decode streams on the same machine — use Workspace.SetThreshold (and
// Workspace.SetWorkers for width) and call the Workspace's matmul methods.
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
// bit-identical. Process-wide DEFAULT; set once at startup. The effective worker
// count is min(width-or-GOMAXPROCS, GOMAXPROCS, columns). For per-stream scoping
// without touching the global, give a Workspace a pool via Workspace.SetWorkers.
func SetParallelWidth(n int) { parWidth = n }

// ParallelWidth reports the current fan-out width cap (0 = GOMAXPROCS).
func ParallelWidth() int { return parWidth }

// resolveWidth resolves a fan-out width — 0 means the process-wide default
// (SetParallelWidth) — against GOMAXPROCS. Used by both the global spawn path
// (width 0) and per-Workspace overrides (Workspace.width).
func resolveWidth(width int) int {
	if width <= 0 {
		width = parWidth
	}
	g := runtime.GOMAXPROCS(0)
	if width <= 0 || width > g {
		return g
	}
	return width
}

// MatmulBT computes dst[M,N] = a[M,K] · b[N,K]ᵀ — the PyTorch [out,in] weight
// layout the safetensors checkpoints store, so no transpose copy is needed.
// Each output is a Dot of an a-row against a b-row; work is parallelized over
// the N output columns (always large in the transformer projections, and the
// only dimension with parallelism on the M=1 single-token decode path).
func MatmulBT(a, b, dst []float32, M, K, N int) {
	parallelCols(M*N*K, N, func(j0, j1 int) { matmulBTSpan(a, b, dst, M, K, N, j0, j1) })
}

// MatmulBT run through a Workspace uses that Workspace's scoped threshold and worker
// pool (SetThreshold / SetWorkers) instead of the process-wide globals — so an
// independent decode stream tunes its own parallelism. Same shape and numerics as
// the package-level MatmulBT.
func (w *Workspace) MatmulBT(a, b, dst []float32, M, K, N int) {
	w.parallelCols(M*N*K, N, func(j0, j1 int) { matmulBTSpan(a, b, dst, M, K, N, j0, j1) })
}

// matmulBTSpan computes output columns [j0,j1) of dst[M,N] = a[M,K]·b[N,K]ᵀ.
func matmulBTSpan(a, b, dst []float32, M, K, N, j0, j1 int) {
	for i := range M {
		arow := a[i*K : i*K+K]
		drow := dst[i*N : i*N+N]
		for j := j0; j < j1; j++ {
			drow[j] = dotF32(arow, b[j*K:j*K+K])
		}
	}
}

// MatmulBTAcc64 is MatmulBT (dst[M,N] = a[M,K] · b[N,K]ᵀ) but each output dot is
// accumulated in float64 before rounding to float32 — bit-identical to a
// sequential-order f64 reference. Inputs and output stay []float32; same shape
// contract as MatmulBT, only the reduction precision changes.
//
// Use it where the f32 reassociation error is amplified downstream. The motivating
// case: a transformer attention (QKᵀ / scores·V) feeding a discrete MoE top-k
// router. f32 reassociation differs from the scalar f64 reference by ~1e-6, which
// can flip an expert at a near-tie and cascade into different generated tokens; the
// f64 accumulate drops the error to ~1e-15, below any realistic router boundary,
// while keeping the parallelism over N. For dense models MatmulBT's f32 accumulate
// is fine — prefer it (this is slower).
func MatmulBTAcc64(a, b, dst []float32, M, K, N int) {
	parallelCols(M*N*K, N, func(j0, j1 int) { matmulBTAcc64Span(a, b, dst, M, K, N, j0, j1) })
}

// MatmulBTAcc64 run through a Workspace uses its scoped threshold + worker pool.
func (w *Workspace) MatmulBTAcc64(a, b, dst []float32, M, K, N int) {
	w.parallelCols(M*N*K, N, func(j0, j1 int) { matmulBTAcc64Span(a, b, dst, M, K, N, j0, j1) })
}

func matmulBTAcc64Span(a, b, dst []float32, M, K, N, j0, j1 int) {
	for i := range M {
		arow := a[i*K : i*K+K]
		drow := dst[i*N : i*N+N]
		for j := j0; j < j1; j++ {
			drow[j] = float32(dotF32Acc64(arow, b[j*K:j*K+K]))
		}
	}
}

// parallelCols runs fn over the [0,N) output columns, split into one chunk per
// worker. Serial below parThreshold MACs (work) where the goroutine fan-out
// would cost more than it saves. Both the f32 and int8 matmuls share it.
func parallelCols(work, N int, fn func(j0, j1 int)) {
	if work < parThreshold || N < 2 {
		fn(0, N)
		return
	}
	parallelSpawnCols(N, resolveWidth(0), fn)
}

// parallelSpawnCols splits [0,N) into one chunk per worker and runs fn on each
// in its own goroutine. The caller has already decided parallelism is worth it
// (so the fn closure's heap escape here is paid only on the parallel path —
// callers that want a zero-alloc serial path call their span function directly
// when below threshold, rather than routing a closure through parallelCols).
// workers is the resolved fan-out (see resolveWidth); it is clamped to [1, N].
func parallelSpawnCols(N, workers int, fn func(j0, j1 int)) {
	if workers > N {
		workers = N
	}
	if workers < 1 {
		workers = 1
	}
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
