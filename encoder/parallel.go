package encoder

import (
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/townsendmerino/aikit/linalg"
)

// Intra-op (within-one-forward) matmul parallelism for the standalone-
// Encode latency case.
//
// The hard constraint: EncodeBatch already saturates every core by
// running runtime.NumCPU() forward workers concurrently (model.go).
// Parallelizing the matmul *inside* a forward would oversubscribe in
// that mode. But a lone Encode() call — one text, nothing else running
// — leaves N-1 cores idle, and a single large forward's matmuls (fc11,
// fc2, wqkv at L≥~256) are big enough to split across them profitably.
//
// We distinguish the two modes WITHOUT threading a flag through the
// hot, shared selfAttention/swigluMLP code (which both forward and
// forwardBatch call): every forward entry point bumps inflightForwards
// on the way in. The matmul parallelizes its row dimension only when
// the count is ≤1 (this forward is the only one in flight) AND the
// shape clears parallelThreshold. Under EncodeBatch the count is
// NumCPU>1, so the matmul stays byte-for-byte on the serial blocked
// path — the batch path is behaviorally unchanged.
//
// Numerics: splitting the M (row) dimension is exact. Each output row
// depends only on its own a-row and the shared b; the row range a
// worker computes is tiled identically to the serial path, so results
// are bit-identical to matmulBTBlockedInto (no new f32 reduction-order
// noise — unlike changing the K-tiling would introduce).
var inflightForwards atomic.Int32

// enterForward/leaveForward bracket one forward pass for the in-flight
// accounting above. Every forward variant (f32/q8, single/batch) must
// pair these so the parallelism gate sees an accurate concurrent-
// forward count.
func enterForward() { inflightForwards.Add(1) }
func leaveForward() { inflightForwards.Add(-1) }

// parallelThreshold is the per-call FLOP count (M*K*N) at/above which
// row-splitting pays off. Tuned on an M1 Pro (8 core) against
// parallel_test's BenchmarkMatmul{Parallel,Serial}_*; goroutine spawn
// is cheap enough (~µs, vs sub-ms matmuls) that even small shapes win:
//
//	shape (M,K,N)        FLOP    serial     parallel   speedup
//	L80 outproj 80,768,768   47M  2.02 ms    0.76 ms    2.6×
//	L80 fc11 80,768,3072    188M  8.59 ms    2.92 ms    2.9×
//	L256 fc11 256,768,3072  604M  27.6 ms    6.37 ms    4.3×
//	L512 fc11 512,768,3072  1.2G  54.5 ms    12.1 ms    4.5×
//
// 32M sits above the per-head attention QKᵀ shape (L512: 512·64·512 ≈
// 17M), keeping those serial: they win in isolation too (~3.4×) but
// recur 12 heads × 12 layers = 144×/forward (1000+ goroutine spawns),
// whose NET effect needs an end-to-end forward benchmark on real
// weights to judge — a follow-up, not guessed at here. Every f32
// linear layer (wqkv/fc11/fc12/fc2/outproj at L≥80) clears 32M and
// parallelizes. Sits well above matmulBT's 4M naive/blocked cutoff.
const parallelThreshold = 32_000_000

// minRowsPerWorker keeps each goroutine's slice fat enough to be worth
// a spawn (and ≥ the linalg blocked-GEMM mBlock tile, so no worker gets a
// sub-tile sliver).
const minRowsPerWorker = 32

// wantParallelMatmul reports whether matmulBTInto should row-split this
// call across cores: only when no other forward is in flight and the
// shape is both large enough (FLOPs) and tall enough (rows to split).
func wantParallelMatmul(M, K, N int) bool {
	if inflightForwards.Load() > 1 {
		return false
	}
	if int64(M)*int64(K)*int64(N) < parallelThreshold {
		return false
	}
	return M >= 2*minRowsPerWorker
}

// minColsForSplit is the narrowest N worth fanning across output columns.
// linalg.MatmulBT shards 8-aligned and clamps its worker count to N, so this
// only has to rule out outputs too narrow to give each worker a shard. Every
// transformer linear is far above it (the narrowest here is N=768).
const minColsForSplit = 128

// wantParallelCols reports whether a matmul should fan across output COLUMNS.
// This is the PRIMARY intra-op axis; the row split below is the fallback for
// outputs too narrow to shard.
//
// That ordering is measured, and it reverses what this file originally did.
// BenchmarkParallelAxis sweeps serial/rows/cols over the real trunk shapes with
// a 12-deep weight bank (so no variant gets to stream b out of L3 repeatedly).
// Columns win at every shape, on a 16-thread 3700X:
//
//	shape (M,K,N)              serial     rows       cols      cols/rows
//	fc11  L22   22,768,3072     2.32 ms   2.29 ms    0.69 ms     3.33×
//	qkv   L91   91,768,2304     7.18 ms   2.87 ms    1.50 ms     1.91×
//	fc11  L91   91,768,3072     9.80 ms   3.81 ms    1.90 ms     2.01×
//	fc2   L91   91,3072,768     13.1 ms   4.89 ms    2.56 ms     1.91×
//	fc11  L357  357,768,3072    33.1 ms   7.12 ms    5.62 ms     1.27×
//	upgate L690 690,768,6144     140 ms   20.8 ms    20.1 ms     1.03×
//
// Two independent reasons, both structural rather than tuning:
//
//  1. A row split hands every worker the WHOLE of b, so the weights are streamed
//     `workers` times; a column split partitions b, so they are streamed once. In
//     a transformer linear a is [L,K] activations (small) and b is [N,K] weights
//     (9.4 MB for one fc11) — the row split multiplies the dominant memory
//     traffic by the worker count. This is why the gap closes as M grows: the
//     arithmetic per byte of b rises until bandwidth stops being the limit.
//  2. matmulBTBlockedIntoParallel caps workers at (M+31)/32, so M=91 gets THREE
//     workers on a 16-thread box, and M=22 gets one (i.e. serial).
//
// Neither depends on the chip, but the cost of (1) does: the original tuning
// table (parallelThreshold, below) was measured on an 8-core M1 Pro, where
// replicating b across a unified, large shared cache is much cheaper than on a
// desktop part with split-CCX L3.
//
// The in-flight guard is the contract wantParallelMatmul documents: under
// EncodeBatch every core is already busy with sibling forwards, so this declines
// and the batch path is behaviorally unchanged.
//
// One effect this makes it easy to misread, recorded because it cost real time
// to diagnose (perf-campaign §7.12): mixing an all-core burst into an otherwise
// serial forward taxes the serial part. A memory-free all-core burst alone slowed
// a following serial trunk 12.8% (boost clock), and a real parallel matmul slowed
// it 32% (boost plus cache). So parallelizing ONE stage of a forward can lose end
// to end even when that stage gets 5× faster — the answer is to parallelize the
// rest, not to revert.
//
// The crossover back to rows is where the row split stops being WORKER-STARVED.
// matmulBTBlockedIntoParallel gives each worker at least minRowsPerWorker rows,
// so it reaches full width only at M ≥ minRowsPerWorker·NumCPU (512 on a
// 16-thread box; 256 on 8). Below that it is running short-handed AND paying the
// b-replication cost, which is why columns win by 3.33× at M=22 and 1.91× at
// M=91. At and above it, the two converge (1.03× at M=690 in the micro) and the
// end-to-end measurement tips the other way — GTE.Encode at L=690 is 2.3% faster
// on rows (p=0.008), where the micro predicted a 3% column win. Near a crossover
// the per-kernel benchmark stops being predictive, so the boundary is taken from
// the end-to-end number.
//
// Deriving it from NumCPU rather than pinning a constant keeps it honest on other
// core counts: the quantity that matters is whether the row split can fill the
// machine, not any particular M.
func wantParallelCols(M, K, N int) bool {
	if inflightForwards.Load() > 1 {
		return false
	}
	if int64(M)*int64(K)*int64(N) < parallelThreshold {
		return false
	}
	if N < minColsForSplit {
		return false
	}
	return M < minRowsPerWorker*numCPU
}

// numCPU caches runtime.NumCPU (it is not free, and this is on the matmul
// dispatch path). GOMAXPROCS changes after init do not retune the crossover;
// that is acceptable for a heuristic and avoids a syscall per matmul.
var numCPU = runtime.NumCPU()

// softmaxRows applies softmaxRow to each of `rows` consecutive `cols`-wide rows
// of scores, in parallel when it is worth it.
//
// This is the other half of the attention cost, and it is the half that grows
// fastest. The linear projections are O(L) work; the attention score matrix is
// O(L²), and every element of it goes through math.Exp. Profiling GTE.Encode at
// L=690 on a 16-thread 3700X: softmaxRow was 2.1 s of a 4.19 s call — HALF the
// wall clock — running entirely on one core, of which 1.63 s was math.archExp.
// Meanwhile the linears, which the parallel matmul does cover, were ~1.4 s.
//
// Rows are independent: each is max-subtracted, exponentiated and normalized
// using only its own elements. Splitting them changes no arithmetic and no
// order within a row, so the result is bit-identical — unlike parallelizing a
// reduction, which would not be.
//
// The in-flight guard is the same contract the matmul gates use: under
// EncodeBatch every core is already busy with sibling forwards.
//
// This does NOT make the exp itself cheaper — that is perf-campaign item 13's
// SIMD expF32, which is orthogonal and multiplies with this.
func softmaxRows(scores []float32, rows, cols int) {
	parallelRows(rows, rows*cols, func(start, end int) {
		for i := start; i < end; i++ {
			softmaxRow(scores[i*cols : (i+1)*cols])
		}
	})
}

// softmaxRowsScaled is softmaxRows fused with the attention score scale —
// replaces a caller's own `for i := range scores { scores[i] *= scale }`
// pass immediately before calling softmaxRows, bit-identically in both
// builds (see elem.SoftmaxRowScaledInto), and genuinely eliminates that
// separate O(L²) pass under GOEXPERIMENT=simd (dead-ends §4.4).
func softmaxRowsScaled(scores []float32, scale float32, rows, cols int) {
	parallelRows(rows, rows*cols, func(start, end int) {
		for i := start; i < end; i++ {
			softmaxRowScaled(scores[i*cols:(i+1)*cols], scale)
		}
	})
}

// parallelRows splits [0,rows) into contiguous ranges across cores and runs fn
// on each, when `work` (the total element count) justifies the spawn and no
// other forward is in flight. Otherwise it runs fn(0, rows) inline on this
// goroutine — so the caller writes one loop body and gets both paths.
//
// fn MUST treat its range as exclusively owned; the callers here write only
// row-local state, which is what makes the split numerically inert.
//
// This exists because parallelizing the matmuls promotes whatever elementwise
// stage was second-largest into first place. Profiling GTE.Encode at L=690
// after the softmax split, gelu was 0.5 s of a 1.88 s call — 27% — on one core.
func parallelRows(rows, work int, fn func(start, end int)) {
	if rows < 2 || work < parallelRowsThreshold || inflightForwards.Load() > 1 {
		fn(0, rows)
		return
	}
	workers := min(numCPU, rows)
	rowsPer := (rows + workers - 1) / workers
	var wg sync.WaitGroup
	for w := range workers {
		start := w * rowsPer
		if start >= rows {
			break
		}
		end := min(start+rowsPer, rows)
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			fn(start, end)
		}(start, end)
	}
	wg.Wait()
}

// parallelRowsThreshold is the element count at/above which splitting an
// elementwise pass pays for the goroutine spawn (~µs against a pass that is
// transcendental-bound at ~20 ns/element).
const parallelRowsThreshold = 1 << 16

// matmulBTColsInto computes dst[M,N] = a[M,K]·b[N,K]ᵀ, fanning across the N
// (output-column) dimension instead of the M (row) dimension.
//
// This is the counterpart to matmulBTInto for the shape where a row split
// cannot help. The SPLADE vocabulary projection is [L,768]·[30522,768]ᵀ: N is
// the 30k vocabulary, but M is a *query* length — often ~20 tokens. That fails
// wantParallelMatmul's M ≥ 2*minRowsPerWorker test, which is the right test for
// a row split and exactly wrong here, and left a ~500 MFLOP matmul on one core
// for every short query (perf-campaign item 7).
//
// The in-flight guard is the same contract wantParallelMatmul documents: under
// EncodeBatch every core is already busy with sibling forwards, so this stays on
// the serial fill and the batch path is behaviorally unchanged. No FLOP gate is
// needed — linalg.MatmulBT applies its own (parallelCols) and runs the span
// inline below it.
//
// Numerics: linalg.MatmulBT shards output columns 8-aligned and runs the same
// blockedFill per shard as the serial path, so each dst[i,j] is bit-identical to
// matmulBTBlockedInto at any fan-out width — linalg's documented M-invariance
// plus TestParallelWidth_bitIdentical; re-gated end-to-end here by
// TestSpladeVocabProj_colParallelIsBitIdentical.
func matmulBTColsInto(a, b, dst []float32, M, K, N int) {
	if inflightForwards.Load() > 1 {
		matmulBTBlockedInto(a, b, dst, M, K, N)
		return
	}
	linalg.MatmulBT(a, b, dst, M, K, N)
}

// matmulBTBlockedIntoParallel splits the M (row) dimension across
// NumCPU goroutines, each computing a disjoint, contiguous block of
// output rows via the serial blocked fill. dst MUST have len ≥ M*N.
//
// Each worker writes dst[iStart*N:iEnd*N] and reads a[iStart*K:iEnd*K]
// + the shared read-only b — disjoint writes, no locking. dst is
// zeroed once up front (the blocked fill accumulates into a zeroed
// region per its k-tile contract).
func matmulBTBlockedIntoParallel(a, b, dst []float32, M, K, N int) {
	if len(dst) < M*N {
		panic("encoder: matmulBTBlockedIntoParallel dst too small")
	}
	workers := runtime.NumCPU()
	maxByRows := (M + minRowsPerWorker - 1) / minRowsPerWorker
	if workers > maxByRows {
		workers = maxByRows
	}
	if workers <= 1 {
		matmulBTBlockedInto(a, b, dst, M, K, N)
		return
	}

	rowsPer := (M + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		iStart := w * rowsPer
		if iStart >= M {
			break
		}
		iEnd := min(iStart+rowsPer, M)
		wg.Add(1)
		go func(iStart, iEnd int) {
			defer wg.Done()
			m := iEnd - iStart
			// Each worker owns a disjoint row block; MatmulBTInto zeroes and
			// fills its own dst sub-slice, so no shared up-front zeroing.
			linalg.MatmulBTInto(dst[iStart*N:iEnd*N], a[iStart*K:iEnd*K], b, m, K, N)
		}(iStart, iEnd)
	}
	wg.Wait()
}
