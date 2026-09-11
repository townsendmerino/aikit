//go:build darwin

// Package annmetal is the native-Metal ann.Backend — int8 corpus GEMV and GEMM plus
// on-device top-k on Apple GPUs, the seam ann.FlatI8.EnableGPU() plugs into.
package annmetal

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/townsendmerino/aikit/ann"
	gpu "github.com/townsendmerino/aikit/gpu"
)

// annbackend.go registers a native-Metal implementation of ann.Backend, so an
// ann.FlatI8 that calls EnableGPU scores its int8 corpus GEMV on the GPU. This is
// the Phase-1 "one proving consumer": it exercises the whole device path — the
// lifted device layer + a minimal W8A8 GEMV kernel — on aikit's highest-fit
// workload (queries × the whole index), parity-gated against the CPU
// linalg.MatmulBTW8A8 (gemv_test.go). The kernels here are intentionally minimal
// and correctness-only; the tuned decode kernel is Phase 1b/2.
//
// Two kernels: gemv_w8a8 scores ONE query against the whole index (FlatI8.Query),
// gemm_w8a8_tiled scores M queries in one dispatch (FlatI8.QueryBatch) — the batched
// int8 GEMM that is the GPU's sweet spot (Phase 2). Both compute the exact int32 dot
// of the host-quantized int8 query against each row, then the query/row rescale — the
// same value w8a8Span produces on the CPU, so GPU and CPU rank identically.
//
// gemm_w8a8_tiled stages a TILE×TILE block of the query tile (A) and the corpus tile (B)
// through threadgroup memory per K-chunk, so each int8 code is read from global memory
// ONCE per tile rather than once per output — the same tiling that took the f32 GEMM from
// ~350 to ~1080 GFLOP/s. It stays BIT-IDENTICAL to the naive one-thread-per-output kernel
// it replaced: the accumulator is int32 and integer addition is associative, so chunking
// K cannot change the sum (a much sharper property than a tolerance, and the batch parity
// test asserts it). Launch with dispatchThreadgroups over a 2-D (ceil(N/TILE), ceil(M/TILE))
// grid of TILE×TILE threads. The first Apple ANN slice showed the naive kernel (~8 GOP/s)
// LOSING ~5× to the SIMD CPU (docs/BENCH-gpu-results.md); this is the Phase-2 fix.
const w8a8Src = `
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
#define TILE 16
// gemv_w8a8 scores ONE query against every row: one SIMD-GROUP per output row, not one
// thread. The naive one-thread-per-row form (lanes j and j+1 read addresses K bytes apart)
// touched 32 cache lines per warp and used one byte from each — measured 4.7% of the M1 Pro's
// 182 GB/s streaming ceiling, and LOST to the CPU SIMD path. Here the 32 lanes of a SIMD-group
// walk the SAME row: lane L reads the L-th char4 chunk, striding by the group width, so the 32
// lanes cover a contiguous 128-byte span (fully-used cache lines / coalesced). simd_sum folds
// the lanes' int32 partials — exact, not approximate, because integer addition is associative
// (this kernel is bit-identity-exempt for that reason; the f32 kernels are NOT).
//
// Launch is N × threadExecutionWidth threads with the threadgroup a multiple of the width, so
// j = gid/W is the row (0..N-1 exactly — Metal's dispatchThreads launches EXACTLY that many,
// no round-up) and lane = gid%W is the SIMD lane; no bound check is needed because every
// thread maps to a valid row. W is read from the hardware, not assumed 32.
kernel void gemv_w8a8(
    device const char*  codes  [[buffer(0)]],   // [N*K] int8, row-major
    device const char*  qi8    [[buffer(1)]],   // [K]   int8 query (host-quantized)
    device const float* scales [[buffer(2)]],   // [N]   per-row scale
    constant uint&      K      [[buffer(3)]],
    constant float&     qscale [[buffer(4)]],   // query scale (0 ⇒ all-zero query)
    device float*       out    [[buffer(5)]],   // [N]   scores
    uint gid [[thread_position_in_grid]],
    uint W   [[threads_per_simdgroup]]) {
    uint j    = gid / W;                 // output row this SIMD-group owns
    uint lane = gid % W;                 // == thread_index_in_simdgroup (tg is a multiple of W)
    device const char* row = codes + (uint)j * K;
    int acc = 0;
    if ((K & 3u) == 0u) {
        // Coalesced char4 path (K%4==0 ⇒ row and qi8 are 4-byte aligned): no __dp4a on Metal,
        // so load char4 per lane and do the four products explicitly.
        device const char4* r4 = (device const char4*)row;
        device const char4* q4 = (device const char4*)qi8;
        for (uint c = lane; c < (K >> 2); c += W) {
            char4 a = q4[c], b = r4[c];
            acc += int(a.x)*int(b.x) + int(a.y)*int(b.y) + int(a.z)*int(b.z) + int(a.w)*int(b.w);
        }
    } else {
        // K not a multiple of 4: scalar strided fallback (still lane-coalesced byte reads).
        for (uint k = lane; k < K; k += W) acc += int(qi8[k]) * int(row[k]);
    }
    int total = simd_sum(acc);           // fold the W lane partials (int add is associative)
    if (lane == 0) out[j] = float(total) * qscale * scales[j];
}
// gemm_w8a8_qtile — the batched GEMM for the shape ANN actually has (audit M-16).
//
// gemm_w8a8_tiled below is the 16x16 byte-tiled form: one output per thread,
// byte-granular threadgroup staging. That is the same shape CUDA's sweep ranked
// WORST and that the roofline campaign measured at ~3% of either M1 Pro roof —
// and the July dead end that concluded "GPU loses ~5x to the SIMD CPU" measured
// that kernel, not the one CUDA's sweep found best. So the dead end's premise is
// what is being retested here, not its arithmetic.
//
// The shape is skinny-M: a handful of QUERIES against a huge corpus. A 64x64
// output tile is the wrong geometry for that — at M=8 it wastes seven eighths of
// every tile. What batching should buy is reading each corpus row ONCE and
// amortizing it across the queries, and that is exactly what this does, reusing
// the structure gemv_w8a8 above already proved: one SIMD-GROUP per corpus row,
// its 32 lanes walking the row in char4 chunks so each warp covers a contiguous
// 128-byte span, and simd_sum folding the lane partials at the end.
//
// The difference from gemv is QTILE accumulators instead of one: a lane loads the
// corpus chunk once and MACs it against QTILE query chunks. The corpus is the
// streamed operand (N*K bytes); the queries are tiny and stay in cache. So the
// corpus traffic per output falls by QTILE.
//
// BIT-IDENTICAL to gemm_w8a8_tiled and to the CPU: the accumulator is int32 and
// integer addition is associative, so neither the lane split nor the simd_sum
// tree can change the sum, and the epilogue applies the same
// qscale[m]*scales[j] in the same order.
#define QTILE 8
kernel void gemm_w8a8_qtile(
    device const char*  codes  [[buffer(0)]],   // [N*K] int8 corpus rows
    device const char*  qi8    [[buffer(1)]],   // [M*K] int8 queries
    device const float* scales [[buffer(2)]],   // [N]
    constant uint&      K      [[buffer(3)]],
    constant uint&      N      [[buffer(4)]],
    device const float* qscale [[buffer(5)]],   // [M]
    device float*       out    [[buffer(6)]],   // [M*N] scores, row-major
    constant uint&      M      [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint W   [[threads_per_simdgroup]]) {
    uint group = gid / W;
    uint lane  = gid % W;
    uint j     = group % N;            // corpus row this SIMD-group owns
    uint mBase = (group / N) * QTILE;  // this pass's query block
    if (j >= N || mBase >= M) return;

    device const char* row = codes + (uint)j * K;
    int acc[QTILE];
    for (uint t = 0; t < QTILE; t++) acc[t] = 0;

    if ((K & 3u) == 0u) {
        device const char4* r4 = (device const char4*)row;
        for (uint c = lane; c < (K >> 2); c += W) {
            char4 b = r4[c];           // read the corpus chunk ONCE for all QTILE queries
            for (uint t = 0; t < QTILE; t++) {
                // Clamp rather than branch: a past-the-end query recomputes the
                // last row's product and its store is skipped below, which keeps
                // the loop bound a compile-time QTILE so acc[] stays in registers.
                uint m = mBase + t; if (m >= M) m = M - 1;
                char4 a = ((device const char4*)(qi8 + (uint)m * K))[c];
                acc[t] += int(a.x)*int(b.x) + int(a.y)*int(b.y) + int(a.z)*int(b.z) + int(a.w)*int(b.w);
            }
        }
    } else {
        for (uint k = lane; k < K; k += W) {
            int b = int(row[k]);
            for (uint t = 0; t < QTILE; t++) {
                uint m = mBase + t; if (m >= M) m = M - 1;
                acc[t] += int(qi8[(uint)m * K + k]) * b;
            }
        }
    }
    for (uint t = 0; t < QTILE; t++) {
        int total = simd_sum(acc[t]);
        uint m = mBase + t;
        if (lane == 0 && m < M) out[(uint)m * N + j] = float(total) * qscale[m] * scales[j];
    }
}
kernel void gemm_w8a8_tiled(
    device const char*  codes  [[buffer(0)]],   // [N*K] int8 (corpus rows = B)
    device const char*  qi8    [[buffer(1)]],   // [M*K] int8 queries (= A)
    device const float* scales [[buffer(2)]],   // [N]
    constant uint&      K      [[buffer(3)]],
    constant uint&      N      [[buffer(4)]],
    device const float* qscale [[buffer(5)]],   // [M] per-query scale
    device float*       out    [[buffer(6)]],   // [M*N] scores, row-major
    constant uint&      M      [[buffer(7)]],
    uint2 tgpos [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]]) {
    threadgroup char As[TILE][TILE]; // query tile  [m-local][k]
    threadgroup char Bs[TILE][TILE]; // corpus tile [n-local][k]
    int tx = (int)tid.x, ty = (int)tid.y;
    int m = (int)tgpos.y * TILE + ty;    // query row
    int n = (int)tgpos.x * TILE + tx;    // corpus row (output column)
    int bRow = (int)tgpos.x * TILE + ty; // the corpus row this thread STAGES (not the one it uses)
    int acc = 0;
    for (uint k0 = 0; k0 < K; k0 += TILE) {
        uint k = k0 + (uint)tx;
        As[ty][tx] = (m < (int)M && k < K) ? qi8[(uint)m * K + k] : (char)0;
        Bs[ty][tx] = (bRow < (int)N && k < K) ? codes[(uint)bRow * K + k] : (char)0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < TILE; kk++) acc += int(As[ty][kk]) * int(Bs[tx][kk]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (m < (int)M && n < (int)N) out[(uint)m * (uint)N + (uint)n] = float(acc) * qscale[m] * scales[n];
}
// topk_rows selects each query's top-k directly on the device, so QueryBatch never copies
// (or allocates on the host) the full M*N score matrix — only M*k hits cross back. One
// threadgroup per query, TOPK_TG threads: each thread strip-scans the row keeping a local
// top-k and sorts it, then the TOPK_TG partial lists are folded by a PARALLEL TREE MERGE
// (log2(TOPK_TG) steps, each an O(k) two-list merge) leaving the group's top-k in slot 0.
// The prior form had thread 0 merge all TOPK_TG*k candidates alone while the rest idled; it
// measured at 1.5–8% of streaming bandwidth (docs/internal/roofline-2026-08.md §6). The order
// is (score DESC, index ASC) — byte-for-byte FlatI8.topHits's tie-break — so the device top-k
// is the SAME set the CPU picks, independent of merge strategy. k is capped at TOPK_MAXK (else
// QueryBatch falls back). TOPK_TG must be a power of two (the reduction halves the stride) and
// the dispatch launches exactly TOPK_TG threads per group, so every group is full.
#define TOPK_TG   128
#define TOPK_MAXK 16
// tkBetter: is (sa,ia) strictly ahead of (sb,ib) in topHits order — score desc, then index
// asc? Filler slots carry (-INF, -1), so any real entry outranks them.
static inline bool tkBetter(float sa, int ia, float sb, int ib) {
    return sa > sb || (sa == sb && ia < ib);
}
kernel void topk_rows(
    device const float* scores   [[buffer(0)]], // [M*N]
    constant uint&      N        [[buffer(1)]],
    constant uint&      k        [[buffer(2)]],
    device int*         outIdx   [[buffer(3)]], // [M*k]
    device float*       outScore [[buffer(4)]], // [M*k]
    uint m [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint tgsz [[threads_per_threadgroup]]) {
    threadgroup float mScore[TOPK_TG * TOPK_MAXK];
    threadgroup int   mIdx  [TOPK_TG * TOPK_MAXK];
    device const float* row = scores + (uint)m * N;
    // 1. strip-scan: this thread's local top-k over its stride of the row (unsorted, worst-tracked).
    float ls[TOPK_MAXK];
    int   li[TOPK_MAXK];
    // worstS/worstI mirror ls[worst]/li[worst] in scalars so the common case — an element that
    // does not beat the current worst — is a register-only compare that never touches the
    // dynamically-indexed (i.e. local-memory-backed) ls/li arrays. Those are read only on an
    // actual eviction, ~k·ln(N/k) times over the strip rather than once per element.
    int cnt = 0, worst = 0;
    float worstS = INFINITY;
    int   worstI = 0;
    for (uint j = tid; j < N; j += tgsz) {
        float s = row[j]; int idx = (int)j;
        if (cnt < (int)k) {
            ls[cnt] = s; li[cnt] = idx; cnt++;
            if (cnt == (int)k) {
                worst = 0;
                for (int t = 1; t < (int)k; t++)
                    if (tkBetter(ls[worst], li[worst], ls[t], li[t])) worst = t;
                worstS = ls[worst]; worstI = li[worst];
            }
        } else if (tkBetter(s, idx, worstS, worstI)) {
            ls[worst] = s; li[worst] = idx;
            worst = 0;
            for (int t = 1; t < (int)k; t++)
                if (tkBetter(ls[worst], li[worst], ls[t], li[t])) worst = t;
            worstS = ls[worst]; worstI = li[worst];
        }
    }
    for (int t = cnt; t < (int)k; t++) { ls[t] = -INFINITY; li[t] = -1; }
    // 2. sort this thread's k entries into topHits order (selection sort, k ≤ TOPK_MAXK).
    for (int a = 0; a < (int)k; a++) {
        int best = a;
        for (int b = a + 1; b < (int)k; b++)
            if (tkBetter(ls[b], li[b], ls[best], li[best])) best = b;
        float ts = ls[a]; ls[a] = ls[best]; ls[best] = ts;
        int   ti = li[a]; li[a] = li[best]; li[best] = ti;
    }
    for (int t = 0; t < (int)k; t++) { mScore[tid * (int)k + t] = ls[t]; mIdx[tid * (int)k + t] = li[t]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 3. parallel tree merge: fold tgsz sorted lists pairwise. Active thread tid merges its
    //    sorted-k list with tid+stride's, keeping the top-k of the two (linear two-pointer, O(k)).
    //    tid+stride is inactive this round, so its list is stable; A is written only after it is
    //    fully read. Leaves the group's top-k in slot 0.
    for (uint stride = tgsz >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            threadgroup float* A  = &mScore[tid * (int)k];
            threadgroup int*   AI = &mIdx[tid * (int)k];
            threadgroup float* B  = &mScore[(tid + stride) * (int)k];
            threadgroup int*   BI = &mIdx[(tid + stride) * (int)k];
            float rs[TOPK_MAXK];
            int   ri[TOPK_MAXK];
            int ia = 0, ib = 0;
            for (int r = 0; r < (int)k; r++) {
                if (ib >= (int)k || (ia < (int)k && tkBetter(A[ia], AI[ia], B[ib], BI[ib]))) {
                    rs[r] = A[ia]; ri[r] = AI[ia]; ia++;
                } else {
                    rs[r] = B[ib]; ri[r] = BI[ib]; ib++;
                }
            }
            for (int r = 0; r < (int)k; r++) { A[r] = rs[r]; AI[r] = ri[r]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // 4. slot 0 holds the merged top-k already in topHits order.
    if (tid == 0)
        for (int r = 0; r < (int)k; r++) {
            outScore[(uint)m * k + (uint)r] = mScore[r];
            outIdx[(uint)m * k + (uint)r]   = mIdx[r];
        }
}`

type metalBackend struct {
	dev    *gpu.Device
	q      gpu.Queue
	gemv   gpu.Pipeline // one query  × N rows (FlatI8.Query), SIMD-group per row
	gemvW  int          // gemv's SIMD-group width (threadExecutionWidth); launch is N×this
	gemmqW int          // gemm_w8a8_qtile's SIMD-group width, same role
	gemm   gpu.Pipeline // M queries × N rows (FlatI8.QueryBatch), 16x16 byte-tiled (fallback)
	gemmq  gpu.Pipeline // M queries × N rows, SIMD-group-per-row QTILE form (audit M-16)
	topk   gpu.Pipeline // per-query top-k over the device score matrix
}

// topkTG / topkMaxK mirror the topk_rows kernel's TOPK_TG / TOPK_MAXK; k above topkMaxK
// falls back to the ScoreBatch (full-matrix) path.
//
// topkMinN is the corpus size below which on-device top-k is DECLINED (→ ScoreBatch + host
// top-k). Measured on M1 Pro (docs/BENCH-gpu-results.md): the second dispatch + the top-k
// reduction pass cost more than the M*N readback they save at small N (device top-k *lost* at
// N=1e4), but win at N≥1e5 — 1.29→1.73× / 1.25→1.98× at batch 64/256 — and win larger still as
// the M*N host allocation the ScoreBatch path makes (~1 GB at N=1e6, batch=256) becomes
// prohibitive. So it is a LARGE-N optimization, gated here rather than applied blindly.
const (
	topkTG   = 128
	topkMaxK = 16
	topkMinN = 100_000
)

// init reaches Metal and registers the backend. If there is no Metal GPU (or the
// kernel fails to compile), it registers nothing — ann.FlatI8 then stays on the
// CPU and EnableGPU returns "no backend". Building this package is the opt-in.
func init() {
	dev, err := gpu.CreateSystemDefaultDevice()
	if err != nil {
		return
	}
	lib, err := dev.CompileLibrary(w8a8Src, gpu.MSL3_1)
	if err != nil {
		dev.ReleaseObjects()
		return
	}
	gemv, err := dev.NewComputePipeline(lib, "gemv_w8a8")
	if err != nil {
		dev.ReleaseObjects()
		return
	}
	gemm, err := dev.NewComputePipeline(lib, "gemm_w8a8_tiled")
	if err != nil {
		dev.ReleaseObjects()
		return
	}
	gemmq, err := dev.NewComputePipeline(lib, "gemm_w8a8_qtile")
	if err != nil {
		dev.ReleaseObjects()
		return
	}
	topk, err := dev.NewComputePipeline(lib, "topk_rows")
	if err != nil {
		dev.ReleaseObjects()
		return
	}
	w := gemv.ThreadExecutionWidth()
	if w <= 0 {
		w = 32 // every Apple GPU is 32; guard a bad query rather than divide by zero on launch
	}
	wq := gemmq.ThreadExecutionWidth()
	if wq <= 0 {
		wq = 32
	}
	ann.RegisterBackend(&metalBackend{dev: dev, q: dev.NewCommandQueue(), gemv: gemv, gemvW: w, gemmqW: wq, gemm: gemm,
		gemmq: gemmq, topk: topk})
}

func (b *metalBackend) Name() string { return "metal" }

// runLocked runs a 1-D dispatch pinned to one OS thread. Run1D allocates and
// drains an NSAutoreleasePool, and objc requires the drain on the SAME thread as
// the alloc — but a Go goroutine can migrate between the two, draining on the
// wrong thread and crashing (an intermittent SIGSEGV). Pinning the thread across
// the whole dispatch keeps them together. (goinfer's decode avoids this with a
// dedicated LockOSThread executor; the per-call proving path locks here instead.)
func (b *metalBackend) runLocked(p gpu.Pipeline, n, tg int, bufs ...gpu.Buffer) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	b.q.Run1D(p, n, tg, bufs...)
}

// qtileGEMM dispatches the batched score GEMM (audit M-16).
//
// The QTILE kernel is a 1-D dispatch of one SIMD-GROUP per (corpus row, query
// block): ceil(M/QTILE) blocks x N rows x W lanes. It replaces the 16x16
// byte-tiled 2-D dispatch, whose geometry wastes most of every tile at the
// skinny-M shape ANN actually runs and which measured ~3% of roof.
//
// Both kernels take the same buffers in the same order and produce identical
// bits (int32 accumulator, same epilogue), so this is a dispatch swap.
const qtileQ = 8 // must match QTILE in the MSL source

func (b *metalBackend) runGEMM(M, N int, bufs ...gpu.Buffer) {
	blocks := (M + qtileQ - 1) / qtileQ
	W := b.gemmqW
	tg := W * 8 // whole SIMD-groups per threadgroup
	b.runLocked(b.gemmq, blocks*N*W, tg, bufs...)
}

// runLocked2D is runLocked for a 2-D dispatchThreadgroups grid — the tiled GEMM's shape
// (gx×gy whole threadgroups of tgx×tgy threads), same OS-thread pinning for the autorelease
// pool.
func (b *metalBackend) runLocked2D(p gpu.Pipeline, gx, gy, tgx, tgy int, bufs ...gpu.Buffer) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	b.q.Run2D(p, gx, gy, tgx, tgy, bufs...)
}

// NewI8Index uploads the int8 codes + scales resident on the device and allocates
// the reusable per-query scratch (quantized query, query scale, output).
func (b *metalBackend) NewI8Index(bq []int8, scales []float32, n, dim int) (idx ann.I8Index, err error) {
	if n <= 0 || dim <= 0 {
		return nil, fmt.Errorf("gpu: empty int8 index (n=%d dim=%d)", n, dim)
	}
	if len(bq) != n*dim || len(scales) != n {
		return nil, fmt.Errorf("gpu: int8 index shape mismatch (bq %d, scales %d, n=%d dim=%d)", len(bq), len(scales), n, dim)
	}
	// A corpus upload is the one allocation here big enough to exhaust device
	// memory, and the device layer reports OOM as a loud panic (MustBuf) rather
	// than a silently unusable buffer. Recover it into an error so EnableGPU
	// declines and the index keeps scoring on the CPU — which is what
	// ann.Backend promises ("falls through to the tiers below on any error").
	// anncuda has had this at all three sites; annmetal did not (audit C-04).
	//
	// The buffers are declared BEFORE the defer so a partial allocation is
	// released rather than leaked. anncuda registers its recover before the
	// allocations but its release defer after them, so a panic part-way through
	// leaks whatever already succeeded — on the one path where device memory is
	// by definition short. ReleaseBuf ignores the zero Buffer, so the loop is
	// safe over the ones that were never assigned.
	var codes, scl, qi8, qscale, kbuf, out gpu.Buffer
	defer func() {
		if r := recover(); r != nil {
			for _, buf := range []gpu.Buffer{codes, scl, qi8, qscale, kbuf, out} {
				b.dev.ReleaseBuf(buf)
			}
			idx, err = nil, fmt.Errorf("gpu: device upload failed: %v", r)
		}
	}()
	codes = gpu.NewBufferOf(b.dev, bq)
	scl = gpu.NewBufferOf(b.dev, scales)
	qi8 = gpu.NewBufferOf(b.dev, make([]int8, dim))
	qscale = gpu.NewBufferOf(b.dev, []float32{0})
	kbuf = gpu.NewBufferOf(b.dev, []uint32{uint32(dim)})
	out = b.dev.NewBufferLen(n)
	return &metalI8Index{
		b: b, n: n, dim: dim,
		codes: codes, scales: scl, qi8: qi8, qscale: qscale, kbuf: kbuf, out: out,
	}, nil
}

type metalI8Index struct {
	b      *metalBackend
	n, dim int
	codes  gpu.Buffer // [n*dim] int8 (resident)
	scales gpu.Buffer // [n] f32
	qi8    gpu.Buffer // [dim] int8 (reused per query)
	qscale gpu.Buffer // [1] f32 (reused)
	kbuf   gpu.Buffer // [1] u32
	out    gpu.Buffer // [n] f32 (reused)
	mu     sync.Mutex
}

// Score dynamically quantizes q on the host (byte-identical to linalg's
// quantizeRowInt8), dispatches the GEMV, and copies the scores back. Serialized:
// the reused scratch buffers are per-index, and one command queue runs serially.
func (x *metalI8Index) Score(q []float32, dst []float32) error {
	if len(q) != x.dim {
		return fmt.Errorf("gpu: query dim %d != index dim %d", len(q), x.dim)
	}
	if len(dst) != x.n {
		return fmt.Errorf("gpu: dst len %d != index n %d", len(dst), x.n)
	}
	x.mu.Lock()
	defer x.mu.Unlock()
	qscale := quantizeRowInt8(q, x.qi8.Int8s())
	x.qscale.Floats()[0] = qscale
	// One SIMD-group per row: launch N×W threads, threadgroup a multiple of W so SIMD-groups
	// never straddle a threadgroup boundary (j = gid/W is then the row, exactly, no bound check).
	W := x.b.gemvW
	tg := 8 * W // 256 at W=32; ≤ maxTotalThreadsPerThreadgroup
	x.b.runLocked(x.b.gemv, x.n*W, tg, x.codes, x.qi8, x.scales, x.kbuf, x.qscale, x.out)
	copy(dst, x.out.Floats())
	return nil
}

// ScoreBatch scores M = len(queries) queries against all N rows in a single
// dispatch of the batched GEMM — the throughput win over calling Score in a loop.
// dst is [M*N] row-major (dst[m*N+j]). Per-call scratch (host quantization + three
// device buffers) is allocated and released each call: batch queries are the
// infrequent path, so this stays simple rather than caching an M-sized arena.
func (x *metalI8Index) ScoreBatch(queries [][]float32, dst []float32) (err error) {
	M := len(queries)
	if M == 0 {
		return nil
	}
	N, K := x.n, x.dim
	if len(dst) != M*N {
		return fmt.Errorf("gpu: batch dst len %d != M*N %d", len(dst), M*N)
	}
	qi8 := make([]int8, M*K)
	qscale := make([]float32, M)
	for m, q := range queries {
		if len(q) != K {
			return fmt.Errorf("gpu: batch query %d dim %d != index dim %d", m, len(q), K)
		}
		qscale[m] = quantizeRowInt8(q, qi8[m*K:(m+1)*K])
	}
	x.mu.Lock()
	defer x.mu.Unlock()
	// Same OOM-to-error contract as NewI8Index (audit C-04): an M*N output can be
	// large — 1 GB at N=1e6, M=256, the advertised shape. Declared before the
	// defer so the release covers a partial allocation too.
	var qi8Buf, qscaleBuf, nBuf, mBuf, outBuf gpu.Buffer
	defer func() {
		for _, b := range []gpu.Buffer{qi8Buf, qscaleBuf, nBuf, mBuf, outBuf} {
			x.b.dev.ReleaseBuf(b)
		}
		if r := recover(); r != nil {
			err = fmt.Errorf("gpu: batch scratch allocation failed: %v", r)
		}
	}()
	qi8Buf = gpu.NewBufferOf(x.b.dev, qi8)
	qscaleBuf = gpu.NewBufferOf(x.b.dev, qscale)
	nBuf = gpu.NewBufferOf(x.b.dev, []uint32{uint32(N)})
	mBuf = gpu.NewBufferOf(x.b.dev, []uint32{uint32(M)})
	outBuf = x.b.dev.NewBufferLen(M * N)
	x.b.runGEMM(M, N, x.codes, qi8Buf, x.scales, x.kbuf, nBuf, qscaleBuf, outBuf, mBuf)
	copy(dst, outBuf.Floats())
	return nil
}

// TopKBatch scores the batch with the same tiled GEMM, then selects each query's top-k ON
// THE DEVICE (topk_rows), so only M*k hits are read back — not the full M*N score matrix,
// which is neither copied to nor allocated on the host (at N=1e6, batch=256 that matrix is
// ~1 GB). The scores are the same int32 dot ScoreBatch produces, and the device selection
// uses topHits's exact (score-desc, index-asc) order, so the result is the SAME top-k set.
// k above topkMaxK returns an error and QueryBatch falls back to the ScoreBatch path.
func (x *metalI8Index) TopKBatch(queries [][]float32, k int) (hits [][]ann.Hit, err error) {
	M := len(queries)
	if M == 0 {
		return [][]ann.Hit{}, nil
	}
	N, K := x.n, x.dim
	if k <= 0 || k > topkMaxK || k >= N || N < topkMinN {
		return nil, fmt.Errorf("gpu: topk declined (k=%d max %d, n=%d min %d) — fall back", k, topkMaxK, N, topkMinN)
	}
	qi8 := make([]int8, M*K)
	qscale := make([]float32, M)
	for m, q := range queries {
		if len(q) != K {
			return nil, fmt.Errorf("gpu: batch query %d dim %d != index dim %d", m, len(q), K)
		}
		qscale[m] = quantizeRowInt8(q, qi8[m*K:(m+1)*K])
	}
	x.mu.Lock()
	defer x.mu.Unlock()
	dev := x.b.dev
	// Same OOM-to-error contract as NewI8Index (audit C-04); scoreBuf alone is
	// M*N floats. Declared before the defer so a partial allocation is released.
	var qi8Buf, qscaleBuf, nBuf, mBuf, kBuf, scoreBuf, idxOut, scoreOut gpu.Buffer
	defer func() {
		for _, b := range []gpu.Buffer{qi8Buf, qscaleBuf, nBuf, mBuf, kBuf, scoreBuf, idxOut, scoreOut} {
			dev.ReleaseBuf(b)
		}
		if r := recover(); r != nil {
			hits, err = nil, fmt.Errorf("gpu: topk scratch allocation failed: %v", r)
		}
	}()
	qi8Buf = gpu.NewBufferOf(dev, qi8)
	qscaleBuf = gpu.NewBufferOf(dev, qscale)
	nBuf = gpu.NewBufferOf(dev, []uint32{uint32(N)})
	mBuf = gpu.NewBufferOf(dev, []uint32{uint32(M)})
	kBuf = gpu.NewBufferOf(dev, []uint32{uint32(k)})
	scoreBuf = dev.NewBufferLen(M * N)                 // device-only; never crosses to the host
	idxOut = gpu.NewBufferOf(dev, make([]uint32, M*k)) // device int* (u32 storage, same bytes)
	scoreOut = dev.NewBufferLen(M * k)                 // only M*k floats come back
	x.b.runGEMM(M, N, x.codes, qi8Buf, x.scales, x.kbuf, nBuf, qscaleBuf, scoreBuf, mBuf)
	// one threadgroup per query (M groups of topkTG threads), reducing scoreBuf → M*k.
	x.b.runLocked(x.b.topk, M*topkTG, topkTG, scoreBuf, nBuf, kBuf, idxOut, scoreOut)

	idxs := idxOut.U32s()
	scs := scoreOut.Floats()
	hits = make([][]ann.Hit, M)
	for m := range M {
		h := make([]ann.Hit, k)
		for r := range k {
			h[r] = ann.Hit{Index: int(int32(idxs[m*k+r])), Score: float64(scs[m*k+r])}
		}
		hits[m] = h
	}
	return hits, nil
}

// Close releases this index's device buffers (the shared device stays alive for
// other indexes; the process holds one Metal device).
func (x *metalI8Index) Close() error {
	for _, b := range []gpu.Buffer{x.codes, x.scales, x.qi8, x.qscale, x.kbuf, x.out} {
		x.b.dev.ReleaseBuf(b)
	}
	return nil
}

// quantizeRowInt8 replicates linalg's (unexported) per-row int8 quantizer exactly:
// scale = maxAbs/127, dst[k] = round(q[k]/scale) clamped to ±127; all-zero input →
// scale 0. Kept byte-identical so the GPU int8 dot equals the CPU one.
func quantizeRowInt8(a []float32, dst []int8) (scale float32) {
	var maxAbs float32
	for _, v := range a {
		if v < 0 {
			v = -v
		}
		if v > maxAbs {
			maxAbs = v
		}
	}
	if maxAbs == 0 {
		for k := range dst {
			dst[k] = 0
		}
		return 0
	}
	scale = maxAbs / 127
	inv := 1.0 / scale
	for k, v := range a {
		x := math.Round(float64(v * inv))
		if x > 127 {
			x = 127
		} else if x < -127 {
			x = -127
		}
		dst[k] = int8(x)
	}
	return scale
}
