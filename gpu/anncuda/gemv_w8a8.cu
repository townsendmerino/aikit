// gemv_w8a8.cu — the W8A8 scoring kernels behind the CUDA ann.Backend
// (docs/task-native-gpu.md, Phase 1b). `gemv_w8a8` scores ONE query against the whole
// index (FlatI8.Query); `gemv_w8a8_batch8`/`gemv_w8a8_batch16` score M queries in one
// launch (FlatI8.QueryBatch, FlatI8.TopKBatch); `topk_rows` selects on the device.
//
// This no longer mirrors gpu/annmetal/backend.go's MSL kernel-for-kernel. Both
// backends started from the same two-kernel sketch, and both have since been
// measured against their own device's ceilings, which pushed them apart — the batch
// path here is a lane-group GEMV where Metal's is a threadgroup-tiled GEMM. The
// PARITY contract is what the two share, not the kernel shape.
//
// BIT-IDENTITY-EXEMPT: structurally immune rather than merely uncontracted. The dot
// products accumulate into an `int` — integer addition is associative, so the reduction
// is exact and order-independent — and the float tail is `(float)acc * qscale * scale`,
// two multiplies with NO add. There is no multiply-accumulate here for a compiler to
// contract, which the shipped PTX confirms: 0 fma.rn.f32 against 50 mul.f32 (the count
// rose with the batch kernels' unrolled epilogues; what matters is that the fma count
// is ZERO, since there is nothing to fuse). Adding a float accumulate or a bias term
// would end that, so re-classify if this kernel grows one.
//
// All compute the EXACT int32 dot of the host-quantized int8 query against each
// row, then apply the query/row rescale — the same value linalg.MatmulBTW8A8
// produces on the CPU. Integer accumulation is exact, so GPU and CPU rank
// identically (backend_test.go gates that, break-it-first).
//
// BOTH SCORING KERNELS WERE ONCE DELIBERATELY UNTUNED — "no __dp4a, no warp-per-row
// reduction, no vectorized loads", on the reasoning that the tuned blob is goinfer's
// and this is only the proving kernel. Measurement retired that twice, for the same
// underlying reason: a proving kernel is fine, but a proving kernel wired to a
// production entry point is a pessimization.
//
//   - gemv_w8a8 (FlatI8.Query) was thread-per-row, so adjacent threads read addresses
//     K bytes apart and a warp's load touched 32 cache lines to use one byte of each.
//     25-28 GB/s against a measured 412 GB/s streaming ceiling. Now a coalesced lane
//     group per row with __dp4a: 3.3-6.0x (be90aec), and a further 15-34% from sweeping
//     the group width rather than assuming a full warp.
//   - The batch path (FlatI8.QueryBatch) ran gpu/vit.cu's 16x16 gemm_w8a8_tiled, whose
//     cost depends on ceil(M/16) rather than M — 6.79 ms at N=200k for every M from 1
//     to 16. Now the lane-group batch GEMV below: 6.4-12.7x, and the tiled kernel is
//     no longer reachable from this backend at any M.
//
// The lesson both share is in docs/internal/roofline-2026-08.md: the denominator has
// to be measured before the kernel can be called fast or slow.
//
// Two shape conventions carried over from the Metal side:
//   - Scalars arrive as 1-element BUFFERS, not by-value kernel params, mirroring
//     MSL's `constant uint&` binds. That is what lets gpu.Queue.Run1D keep Metal's
//     `bufs ...Buffer` signature on both platforms.
//   - Each kernel takes its own thread count as a trailing param and guards on it:
//     a CUDA launch rounds up to whole blocks, where Metal's dispatchThreads
//     launches exactly n (gpu/cuda.go, divergence 2).

// GEMV_LANES is how many threads cooperate on one corpus row. It was 32 (one full
// warp), which is the obvious choice and not the best one: a warp of 32 lanes reducing
// one row pays a 5-deep shuffle tree per row, and at K=768 each lane only gets 6 loads
// to amortize it against.
//
// SWEPT, and the answer is not the fastest single number. ms per launch at N=200k:
//
//            K=768   K=256   K=255 (byte path)
//   L4       0.418   0.151   0.495
//   L8       0.398   0.144   0.325
//   L16      0.400   0.150   0.260
//   L32      0.491   0.229   0.305
//
// L8 wins the two k%4==0 columns — but only by 0-4%, and it LOSES the byte path by 25%,
// because there each lane loads a single byte and 8 lanes cover only 8 of a 32-byte
// sector. L16 is within 4% of the best on every shape measured, including the one that
// is not the common case. Robustness across shapes is worth more here than 4% on the
// shapes we happen to benchmark, and it avoids a K-dependent second kernel.
//
// Against the old 32: 19% at K=768, 34% at K=256, 15% at K=255.
#define GEMV_LANES 16

// gemv_w8a8: out[j] = dot(qi8, codes[j]) * qscale * scales[j], one LANE GROUP per row.
extern "C" __global__ void gemv_w8a8(
    const signed char* __restrict__ codes,  // [N*K] int8, row-major
    const signed char* __restrict__ qi8,    // [K]   int8 query (host-quantized)
    const float* __restrict__ scales,       // [N]   per-row scale
    const unsigned int* __restrict__ K,
    const float* __restrict__ qscale,       // query scale (0 => all-zero query)
    float* __restrict__ out,                // [N]   scores
    const unsigned int* __restrict__ N)     // row count (the launch bound)
{
    // GEMV_LANES threads walk the SAME row, so their loads coalesce — one fully-used
    // transaction against the 32 partly-used ones the original thread-per-row form
    // issued. The launch dispatches N*GEMV_LANES threads (see cudaI8Index.Score); j is
    // the global LANE-GROUP index, not a thread index.
    unsigned int groupsPerBlock = blockDim.x / GEMV_LANES;
    unsigned int j = blockIdx.x * groupsPerBlock + (threadIdx.x / GEMV_LANES);
    if (j >= *N) return;
    unsigned int lane = threadIdx.x & (unsigned int)(GEMV_LANES - 1);
    unsigned int k = *K;
    const signed char* row = codes + (unsigned long long)j * k;

    int acc = 0;
    if ((k & 3u) == 0) {
        // 4 int8 MACs per instruction via __dp4a, and 4 bytes per lane per load, so a
        // group moves GEMV_LANES*4 bytes at a time. Safe because k%4==0 makes every row
        // start 4-byte aligned (the base allocation is 256-byte aligned).
        const int* row4 = (const int*)row;
        const int* q4 = (const int*)qi8;
        unsigned int k4 = k >> 2;
        for (unsigned int i = lane; i < k4; i += GEMV_LANES) {
            acc = __dp4a(row4[i], q4[i], acc);
        }
    } else {
        // Byte path for k%4 != 0: still coalesced, just one MAC per lane per step
        // instead of four. This is the column GEMV_LANES=16 is chosen to protect.
        for (unsigned int i = lane; i < k; i += GEMV_LANES) {
            acc += (int)qi8[i] * (int)row[i];
        }
    }

    // Reduction over the group. The shuffles are warp-wide but the tree is only
    // log2(GEMV_LANES) deep, so the other groups sharing this warp reduce their own rows
    // in the same instructions; lanes above the group boundary read across it and their
    // results are discarded, since only lane 0 stores.
    //
    // Integer addition is associative, so this is EXACTLY the sequential sum — the
    // reason the group width is a free parameter at all, and why this kernel stays
    // BIT-IDENTITY-EXEMPT rather than needing a contract. TestCUDAGEMV_parityWithCPU
    // holds that to exact equality.
    for (int off = GEMV_LANES / 2; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        out[j] = (float)acc * (*qscale) * scales[j];
    }
}

// ---------------------------------------------------------------------------
// gemv_w8a8_batch{8,16}: the BATCHED form — out[m*N+j] for M queries — as one
// group of lanes per corpus row, with the query tile staged in shared memory so
// every loaded corpus row is reused across all QTILE queries in the tile.
//
// WHY THIS REPLACED THE TILED GEMM. This path used to run gpu/vit.cu's
// gemm_w8a8_tiled, which stages a 16x16 output tile, so its wall time depends on
// ceil(M/16) rather than on M: measured on nvidia-rtx2070s at N=200k K=768 it took
// 6.79 ms for EVERY M from 1 to 16, and 109 ms at M=256. Looping the single-query
// gemv_w8a8 costs 0.47 ms per query, so the tiled kernel did not break even until
// M ~ 15 and never beat that loop by more than 9%. Both sat far below both roofs
// (360 GMAC/s against a measured 4876 GMAC/s int32-MAD ceiling; 45 GB/s against a
// measured 412 GB/s streaming ceiling) because a 16x16 byte-wise tile spends its
// time on shared-memory traffic and __syncthreads, not on the dot. The constant
// that selected it (gemmTileMinM = 2) was calibrated when the single-query GEMV was
// still thread-per-row and 5.6x slower — the tiled kernel only ever looked good
// against a broken baseline. See docs/internal/roofline-2026-08.md.
//
// THE TWO PARAMETERS WERE SWEPT, NOT REASONED. LANES (lanes cooperating on one row)
// and QTILE (queries per tile) trade off against each other: the shuffle reduction
// costs QTILE * log2(LANES) per row, so a wider query tile wants FEWER lanes, while
// coalescing wants more (8 lanes * 4 B = 32 B, exactly one sector). Measured at
// N=200k K=768, ms per launch, best in each column marked *:
//
//        M=1     M=2     M=4     M=8    M=16    M=64   M=256
//  L8Q8   0.571*  0.547*  0.587*  0.664*  1.267   4.858  19.450
//  L4Q16  0.908   0.914   0.922   0.956   1.082*  3.935*  15.513*
//  L8Q16  0.919   0.921   0.944   1.026   1.206   4.465  17.360
//  L2Q8   0.978   0.981   0.982   0.993   1.940   7.531  24.970
//  L32Q16 1.893   —       —       —       2.402   9.622  —
//
// Hence L8Q8 for M <= 8 and L4Q16 above it. The all-32-lane form at the bottom is
// what a straight port of the single-query kernel would have given: 2.2x off.
//
// QTILE ACCUMULATORS ARE ALWAYS COMPUTED, EVEN WHEN M < QTILE. The alternative — a
// runtime-bounded loop over mc — indexes acc[] dynamically, which spills the array
// from registers to local memory and costs far more than the padded MACs. The
// staging loop zero-fills the unused query rows, so the padded accumulators compute
// a defined zero and are simply not stored. That is what the M=1..8 column above is
// paying for, and it is still an order of magnitude better than the tiled kernel.
//
// BIT-IDENTITY: same structural exemption as gemv_w8a8 — int accumulation is
// associative, so the reduction over LANES equals the sequential sum, and the float
// tail is two multiplies with no add. out[m*N+j] is exactly the value gemv_w8a8
// writes for query m, which TestCUDAGEMM_batchMatchesQuery gates as exact equality.
// ---------------------------------------------------------------------------

template <int LANES, int QTILE>
__device__ __forceinline__ void batch_body(
    const signed char* __restrict__ codes, const signed char* __restrict__ qi8,
    const float* __restrict__ scales, const unsigned int* __restrict__ K,
    const float* __restrict__ qscale, float* __restrict__ out,
    const unsigned int* __restrict__ N, const unsigned int* __restrict__ M)
{
    // Dynamic shared memory, declared as int so the 4-byte reads below are aligned;
    // the launch sizes it at QTILE*K bytes (gpu.Buffer-free, so it costs no upload).
    extern __shared__ int qsmem[];
    signed char* qs = (signed char*)qsmem;

    unsigned int k = *K, n = *N, m = *M;
    unsigned int m0 = blockIdx.y * QTILE;
    unsigned int mc = (m - m0 < QTILE) ? (m - m0) : QTILE;

    // Stage this tile's queries, zero-filling rows past mc. Every thread in the block
    // runs this loop and reaches the barrier; with the grid-stride loop below, no thread
    // returns early at all, so the barrier cannot strand anyone.
    for (unsigned int i = threadIdx.x; i < (unsigned int)QTILE * k; i += blockDim.x) {
        unsigned int t = i / k;
        qs[i] = (t < mc) ? qi8[(unsigned long long)(m0 + t) * k + (i - t * k)] : (signed char)0;
    }
    __syncthreads();

    unsigned int groupsPerBlock = blockDim.x / LANES;
    unsigned int lane = threadIdx.x & (unsigned int)(LANES - 1);

    // GRID-STRIDE OVER ROWS, so the staging above is paid once per BLOCK rather than
    // once per row-group. It used to be one block per group of rows, which meant every
    // block re-staged the whole QTILE*K query tile and hit the barrier for 64 rows of
    // work. Ablation put that at 14% of the kernel at M=64 — the second largest line
    // item after the corpus stream itself — and launching 8x fewer blocks recovered
    // 10-24% (M=16 1.230 -> 1.111 ms, M=64 4.659 -> 3.552, M=256 15.49 -> 13.87).
    //
    // Same arithmetic per row, so results are unchanged; only how many rows amortize
    // one staging. The launch chooses the block count (see launchBatch).
    unsigned int stride = gridDim.x * groupsPerBlock;
    for (unsigned int j = blockIdx.x * groupsPerBlock + (threadIdx.x / LANES);
         j < n; j += stride) {
    const signed char* row = codes + (unsigned long long)j * k;

    int acc[QTILE];
    #pragma unroll
    for (int t = 0; t < QTILE; t++) acc[t] = 0;

    if ((k & 3u) == 0) {
        // 4 int8 MACs per instruction via __dp4a, 4 bytes per lane per load. k%4==0
        // makes every row start 4-byte aligned (the base allocation is 256-byte
        // aligned), and the shared tile inherits qsmem's alignment.
        const int* row4 = (const int*)row;
        const int* qs4 = (const int*)qs;
        unsigned int k4 = k >> 2;
        for (unsigned int i = lane; i < k4; i += LANES) {
            int rv = row4[i]; // ONE global load feeds all QTILE dp4a's — the whole point
            #pragma unroll
            for (int t = 0; t < QTILE; t++) {
                acc[t] = __dp4a(rv, qs4[(unsigned int)t * k4 + i], acc[t]);
            }
        }
    } else {
        // Byte path for k%4 != 0: still coalesced, one MAC per lane per step.
        for (unsigned int i = lane; i < k; i += LANES) {
            int rv = (int)row[i];
            #pragma unroll
            for (int t = 0; t < QTILE; t++) {
                acc[t] += rv * (int)qs[(unsigned int)t * k + i];
            }
        }
    }

    // One reduction per query. The shuffles are warp-wide but the tree is only
    // log2(LANES) deep, so the lanes of the other row-groups sharing this warp
    // reduce their own rows in the same instructions.
    #pragma unroll
    for (int t = 0; t < QTILE; t++) {
        int a = acc[t];
        for (int off = LANES / 2; off > 0; off >>= 1) {
            a += __shfl_down_sync(0xffffffffu, a, off);
        }
        if (lane == 0 && (unsigned int)t < mc) {
            out[(unsigned long long)(m0 + t) * n + j] = (float)a * qscale[m0 + t] * scales[j];
        }
    }
    } // grid-stride row loop
}

// gemv_w8a8_batch8 — 8 lanes per row, 8 queries per tile. For M <= 8.
extern "C" __global__ void gemv_w8a8_batch8(
    const signed char* __restrict__ codes,  // [N*K] int8, row-major
    const signed char* __restrict__ qi8,    // [M*K] int8 queries (host-quantized)
    const float* __restrict__ scales,       // [N]   per-row scale
    const unsigned int* __restrict__ K,
    const float* __restrict__ qscale,       // [M]   per-query scale
    float* __restrict__ out,                // [M*N] scores, row-major
    const unsigned int* __restrict__ N,
    const unsigned int* __restrict__ M)     // query count
{
    batch_body<8, 8>(codes, qi8, scales, K, qscale, out, N, M);
}

// gemv_w8a8_batch16 — 4 lanes per row, 16 queries per tile. For M > 8.
extern "C" __global__ void gemv_w8a8_batch16(
    const signed char* __restrict__ codes,
    const signed char* __restrict__ qi8,
    const float* __restrict__ scales,
    const unsigned int* __restrict__ K,
    const float* __restrict__ qscale,
    float* __restrict__ out,
    const unsigned int* __restrict__ N,
    const unsigned int* __restrict__ M)
{
    batch_body<4, 16>(codes, qi8, scales, K, qscale, out, N, M);
}

// topk_rows — per-query top-k selection ON THE DEVICE, so a batch returns M*k hits
// instead of the whole M*N score matrix. On a discrete GPU that readback is a real
// PCIe copy (at N=1e6, batch=256 it is ~1 GB), which is the cost this exists to avoid.
//
// THE ORDER MUST MATCH ann.FlatI8.topHits EXACTLY, or the GPU returns a different
// (still plausible) set and parity silently degrades: score DESCENDING, and on equal
// scores the LOWER INDEX wins. That tie-break is not decoration — an all-equal-scores
// index must yield [0..k-1], and the test fixture asserts precisely that.
//
// TWO KERNELS, and which one runs is a function of k alone.
//
// topk_rows_reg (the default) makes ONE pass over the row. Each thread keeps its own
// best TKREG candidates in registers, and the block then merges 256 sorted lists by
// popping heads k times — a merge over 256 elements per pass rather than over N.
//
// topk_rows makes k passes over the row, each a block-wide argmax that writes -FLT_MAX
// into the winner's slot so the next pass cannot re-select it. It is kept for k > TKREG,
// where the register form's per-thread array would spill.
//
// "k*N/threads work per query is cheap next to the GEMM that produced the row" is what
// the k-pass form used to claim, and it was true when the GEMM was the 16x16 tiled one.
// Measured against the batch GEMV that replaced it, at N=200k on nvidia-rtx2070s:
//
//              gemv    topk (k-pass)
//   M=8  k=10  0.88 ms   2.57 ms      topk is 2.9x the GEMV
//   M=64 k=10  4.75      2.98         0.6x
//   M=256 k=10 18.2      7.46         0.4x
//   M=256 k=50 18.2     36.0          2.0x
//
// The k-dependence is the whole problem: the GEMV's cost does not move with k and this
// one's is linear in it. Same lesson as the tiled GEMM above — a cost that was genuinely
// negligible became dominant when the thing it was measured against got faster.
//
// THE ORDER IS THE CONTRACT, in both kernels. score DESCENDING, and on equal scores the
// LOWER INDEX wins, matching ann.FlatI8.topHits exactly — an all-equal-scores index must
// yield [0..k-1], and the test fixture asserts precisely that. In topk_rows_reg that rule
// lives in one place (tkBetter) and is used by the per-thread insert, the block merge and
// the pop, so the three cannot drift apart.
//
// Real dot-product scores do not reach -FLT_MAX, which both kernels rely on: it is the
// k-pass form's "already consumed" marker and the register form's empty-slot sentinel.
#define TKBLOCK 256

// TKREG is the per-thread candidate count, and therefore the largest k a given
// instantiation can answer: a thread's own stride may contain the entire global top-k,
// so it must be able to hold k of them.
//
// THREE INSTANTIATIONS, BECAUSE THE COST IS LINEAR IN TKREG AND FLAT IN k. That is the
// opposite of the k-pass kernel and worth stating plainly. The scan rejects almost
// every element with one compare, but an ACCEPTED candidate costs a TKREG-long bubble,
// and accepts are not rare — a thread scanning S elements accepts about TKREG*ln(S) of
// them, which at S=781 is ~6.7 per slot. So TKREG is the price, and k only decides
// which price you have to pay. Measured at N=200k, ms per launch:
//
//              k<=8   k<=16   k<=32   k<=64      bytes/s at M=256
//   TKREG=8   0.878    —       —       —         233 GB/s  (57% of the 412 roof)
//   TKREG=16  1.115   2.17     —       —          95 GB/s
//   TKREG=32  1.107   2.15    4.89     —          42 GB/s
//   TKREG=64    —      —       —     19.3         11 GB/s
//   k-pass    1.11*   2.16*   ~23    36.0         (*extrapolated from k=10: 7.46)
//
// (M=256; the M=8 and M=64 rows agree on the ordering.) A single TKREG=32 kernel would
// have been 2.3x slower at k=10 — the repo's own default — than picking by k.
//
// float4 loads are used ONLY at TKREG=8, and that is measured too: they cut it 1.107 ->
// 0.878 where the scan is close to memory-bound, and cost 15-25% at TKREG=16 and 32
// where it is not, because the four staged floats add register pressure to a kernel
// already limited by it.
#define TKREG_SMALL 8
#define TKREG_MID 16
#define TKREG_BIG 32
#define TKREG_HUGE 64
#define TKREG_GIANT 128

// tkBetter is THE tie-break, in one place: score descending, lower index wins ties, and
// an empty slot (idx < 0) loses to anything real. Every comparison in the register
// kernels — the insert, the block merge and the pop — goes through it, so the three
// cannot drift apart.
__device__ __forceinline__ bool tkBetter(float av, int ai, float bv, int bi) {
    if (bi < 0) return ai >= 0;
    if (ai < 0) return false;
    return av > bv || (av == bv && ai < bi);
}

// tkInsert: offer one candidate to a sorted best-first register array. The array was
// sorted and only its last element changes, so ONE bubble-up pass restores order. All
// indices are constant so the array stays in registers; a runtime-indexed version
// spills to local memory and loses more than the pass it saves.
template <int TKREG>
__device__ __forceinline__ void tkInsert(float* kv, int* ki, float v, int idx) {
    if (!tkBetter(v, idx, kv[TKREG - 1], ki[TKREG - 1])) return;
    kv[TKREG - 1] = v;
    ki[TKREG - 1] = idx;
    #pragma unroll
    for (int j = TKREG - 1; j >= 1; j--) {
        if (tkBetter(kv[j], ki[j], kv[j - 1], ki[j - 1])) {
            float tv = kv[j]; kv[j] = kv[j - 1]; kv[j - 1] = tv;
            int ti = ki[j]; ki[j] = ki[j - 1]; ki[j - 1] = ti;
        }
    }
}

// tkInit seeds a per-thread array with empty slots.
template <int TKREG>
__device__ __forceinline__ void tkInit(float* kv, int* ki) {
    #pragma unroll
    for (int i = 0; i < TKREG; i++) { kv[i] = -3.402823466e+38f; ki[i] = -1; }
}

// tkScan folds row[lo:hi) into a per-thread array, one pass, threads striding by TKBLOCK.
//
// THE ALIGNMENT GUARD IS LOAD-BEARING, not defensive. row is scores + m*N, so a float4
// read of it is only legal when that pointer is 16-byte aligned — which the 256-byte
// aligned base allocation gives only when m*N is a multiple of 4. At N=60001 every odd
// row is off by one float and the kernel takes CUDA_ERROR_MISALIGNED_ADDRESS, which
// kills the whole context, not just the launch. Checking the pointer rather than
// reasoning about N keeps that self-evident. lo is always a multiple of 4 (the caller
// rounds chunk boundaries), so it cannot reintroduce a misalignment.
template <int TKREG, bool VEC4>
__device__ __forceinline__ void tkScan(const float* __restrict__ row,
                                       unsigned int lo, unsigned int hi,
                                       float* kv, int* ki)
{
    if (VEC4 && ((((unsigned long long)row) & 15ull) == 0)) {
        unsigned int e4 = hi >> 2;
        const float4* r4 = (const float4*)row;
        for (unsigned int i = (lo >> 2) + threadIdx.x; i < e4; i += TKBLOCK) {
            float4 v = r4[i];
            int b = (int)(i << 2);
            tkInsert<TKREG>(kv, ki, v.x, b);
            tkInsert<TKREG>(kv, ki, v.y, b + 1);
            tkInsert<TKREG>(kv, ki, v.z, b + 2);
            tkInsert<TKREG>(kv, ki, v.w, b + 3);
        }
        for (unsigned int i = (e4 << 2) + threadIdx.x; i < hi; i += TKBLOCK) {
            tkInsert<TKREG>(kv, ki, row[i], (int)i);
        }
    } else {
        for (unsigned int i = lo + threadIdx.x; i < hi; i += TKBLOCK) {
            tkInsert<TKREG>(kv, ki, row[i], (int)i);
        }
    }
}

// tkEmit drains the block's per-thread arrays into k sorted winners: k rounds of
// "block-wide argmax over the TKBLOCK heads, winner pops". Each round reduces over 256
// candidates rather than over N, which is why k costs so little here.
//
// EVERY THREAD IN THE BLOCK MUST REACH THIS, including threads whose scan found nothing
// — the barriers are block-wide. Callers that skip the scan (a split block past the end
// of the row) still call tkEmit, and correctly emit sentinels.
template <int TKREG>
__device__ __forceinline__ void tkEmit(float* kv, int* ki, unsigned int k,
                                       int* __restrict__ outIdx, float* __restrict__ outVal)
{
    __shared__ float sVal[TKBLOCK];
    __shared__ int sIdx[TKBLOCK];
    for (unsigned int j = 0; j < k; j++) {
        sVal[threadIdx.x] = kv[0];
        sIdx[threadIdx.x] = ki[0];
        __syncthreads();
        for (unsigned int off = TKBLOCK / 2; off > 0; off >>= 1) {
            if (threadIdx.x < off) {
                if (tkBetter(sVal[threadIdx.x + off], sIdx[threadIdx.x + off],
                             sVal[threadIdx.x], sIdx[threadIdx.x])) {
                    sVal[threadIdx.x] = sVal[threadIdx.x + off];
                    sIdx[threadIdx.x] = sIdx[threadIdx.x + off];
                }
            }
            __syncthreads();
        }
        float wv = sVal[0];
        int wi = sIdx[0];
        if (threadIdx.x == 0) { outIdx[j] = wi; outVal[j] = wv; }
        // The winner pops its head. Candidate indices are unique within a query, so
        // exactly one thread matches — no need to reduce a thread id alongside the value.
        if (wi >= 0 && ki[0] == wi) {
            #pragma unroll
            for (int i = 0; i < TKREG - 1; i++) { kv[i] = kv[i + 1]; ki[i] = ki[i + 1]; }
            kv[TKREG - 1] = -3.402823466e+38f;
            ki[TKREG - 1] = -1;
        }
        __syncthreads();
    }
}

// topk_reg_body: ONE block per query over the whole row.
template <int TKREG, bool VEC4>
__device__ __forceinline__ void topk_reg_body(
    const float* __restrict__ scores, int* __restrict__ outIdx,
    float* __restrict__ outScore, const unsigned int* __restrict__ Mp,
    const unsigned int* __restrict__ Np, const unsigned int* __restrict__ kp)
{
    unsigned int M = *Mp, N = *Np, k = *kp;
    unsigned int m = blockIdx.x;
    if (m >= M) return;
    float kv[TKREG];
    int ki[TKREG];
    tkInit<TKREG>(kv, ki);
    tkScan<TKREG, VEC4>(scores + (unsigned long long)m * N, 0, N, kv, ki);
    tkEmit<TKREG>(kv, ki, k, outIdx + (unsigned long long)m * k,
                  outScore + (unsigned long long)m * k);
}

// ---------------------------------------------------------------------------
// SPLIT + MERGE — the same selection across MANY blocks per query.
//
// WHY. topk_reg_body puts one block on a query, so a batch of M occupies M of this
// device's 40 SMs: at M=8 that is 20% of the machine and at M=1 it is 2.5%. Measured at
// N=200k k=10, the one-block form costs 0.77 ms at M=8 and 2.16 ms at M=256 — barely 3x
// for 32x the work, which is the signature of a launch that was starved and then filled
// up. The batch GEMV in front of it has no such limit.
//
// Splitting also unblocks the single-query path. ann.FlatI8.Query currently reads all N
// scores back and selects on the host (readback 0.178 ms + host top-k 0.147 at N=200k,
// together 43% of the call). A device selection could replace both, but only if it beats
// 0.325 ms — which one block over the whole corpus does not.
//
// Phase 1 (topk_split_*): grid (parts, M). Block (c, m) selects the top k of its own
// chunk of row m and writes them to partials[m][c][0:k].
// Phase 2 (topk_merge_*): grid (M). Block m selects the top k of its parts*k partials.
//
// Chunk boundaries are rounded UP to a multiple of 4 so tkScan's float4 path stays
// aligned inside the row; the last blocks may therefore get nothing, which tkEmit
// handles by emitting sentinels that lose to everything in phase 2.
//
// EXACTNESS SURVIVES THE SPLIT because selection is decomposable in a way summation is
// not: the global top k is contained in the union of the per-chunk top k, since a
// candidate excluded from its own chunk's top k has k better candidates in that chunk
// alone. Both phases use the same tkBetter, so the tie-break is the same rule applied
// twice rather than two rules that must agree.
// ---------------------------------------------------------------------------

template <int TKREG, bool VEC4>
__device__ __forceinline__ void topk_split_body(
    const float* __restrict__ scores, int* __restrict__ partIdx,
    float* __restrict__ partVal, const unsigned int* __restrict__ Mp,
    const unsigned int* __restrict__ Np, const unsigned int* __restrict__ kp)
{
    unsigned int M = *Mp, N = *Np, k = *kp;
    unsigned int c = blockIdx.x, m = blockIdx.y;
    if (m >= M) return;
    unsigned int parts = gridDim.x;
    unsigned int chunk = (((N + parts - 1) / parts) + 3u) & ~3u;
    unsigned int lo = c * chunk;
    unsigned int hi = lo + chunk;
    if (hi > N) hi = N;

    float kv[TKREG];
    int ki[TKREG];
    tkInit<TKREG>(kv, ki);
    // Block-uniform: every thread here takes the same branch, so skipping the scan
    // cannot strand anyone at tkEmit's barriers.
    if (lo < hi) {
        tkScan<TKREG, VEC4>(scores + (unsigned long long)m * N, lo, hi, kv, ki);
    }
    unsigned long long slot = ((unsigned long long)m * parts + c) * k;
    tkEmit<TKREG>(kv, ki, k, partIdx + slot, partVal + slot);
}

template <int TKREG>
__device__ __forceinline__ void topk_merge_body(
    const int* __restrict__ partIdx, const float* __restrict__ partVal,
    int* __restrict__ outIdx, float* __restrict__ outScore,
    const unsigned int* __restrict__ Mp, const unsigned int* __restrict__ Pp,
    const unsigned int* __restrict__ kp)
{
    unsigned int M = *Mp, P = *Pp, k = *kp; // P = parts*k candidates per query
    unsigned int m = blockIdx.x;
    if (m >= M) return;
    const int* pi = partIdx + (unsigned long long)m * P;
    const float* pv = partVal + (unsigned long long)m * P;
    float kv[TKREG];
    int ki[TKREG];
    tkInit<TKREG>(kv, ki);
    for (unsigned int i = threadIdx.x; i < P; i += TKBLOCK) {
        tkInsert<TKREG>(kv, ki, pv[i], pi[i]);
    }
    tkEmit<TKREG>(kv, ki, k, outIdx + (unsigned long long)m * k,
                  outScore + (unsigned long long)m * k);
}

#define TOPK_REG_ENTRY(NAME, TKREG, VEC4)                                      \
extern "C" __global__ void NAME(                                               \
    const float* __restrict__ scores, /* [M*N] — read-only, NOT consumed */    \
    int* __restrict__ outIdx,         /* [M*k] */                              \
    float* __restrict__ outScore,     /* [M*k] */                              \
    const unsigned int* __restrict__ Mp,                                       \
    const unsigned int* __restrict__ Np,                                       \
    const unsigned int* __restrict__ kp)                                       \
{                                                                              \
    topk_reg_body<TKREG, VEC4>(scores, outIdx, outScore, Mp, Np, kp);          \
}

#define TOPK_SPLIT_ENTRY(NAME, TKREG, VEC4)                                    \
extern "C" __global__ void NAME(                                               \
    const float* __restrict__ scores, int* __restrict__ partIdx,               \
    float* __restrict__ partVal, const unsigned int* __restrict__ Mp,          \
    const unsigned int* __restrict__ Np, const unsigned int* __restrict__ kp)  \
{                                                                              \
    topk_split_body<TKREG, VEC4>(scores, partIdx, partVal, Mp, Np, kp);        \
}

#define TOPK_MERGE_ENTRY(NAME, TKREG)                                          \
extern "C" __global__ void NAME(                                               \
    const int* __restrict__ partIdx, const float* __restrict__ partVal,        \
    int* __restrict__ outIdx, float* __restrict__ outScore,                    \
    const unsigned int* __restrict__ Mp, const unsigned int* __restrict__ Pp,  \
    const unsigned int* __restrict__ kp)                                       \
{                                                                              \
    topk_merge_body<TKREG>(partIdx, partVal, outIdx, outScore, Mp, Pp, kp);    \
}

TOPK_REG_ENTRY(topk_rows_r8, TKREG_SMALL, true)
TOPK_REG_ENTRY(topk_rows_r16, TKREG_MID, false)
TOPK_REG_ENTRY(topk_rows_r32, TKREG_BIG, false)
TOPK_REG_ENTRY(topk_rows_r64, TKREG_HUGE, false)
TOPK_REG_ENTRY(topk_rows_r128, TKREG_GIANT, false)

TOPK_SPLIT_ENTRY(topk_split_r8, TKREG_SMALL, true)
TOPK_SPLIT_ENTRY(topk_split_r16, TKREG_MID, false)
TOPK_SPLIT_ENTRY(topk_split_r32, TKREG_BIG, false)
TOPK_SPLIT_ENTRY(topk_split_r64, TKREG_HUGE, false)
TOPK_SPLIT_ENTRY(topk_split_r128, TKREG_GIANT, false)

TOPK_MERGE_ENTRY(topk_merge_r8, TKREG_SMALL)
TOPK_MERGE_ENTRY(topk_merge_r16, TKREG_MID)
TOPK_MERGE_ENTRY(topk_merge_r32, TKREG_BIG)
TOPK_MERGE_ENTRY(topk_merge_r64, TKREG_HUGE)
TOPK_MERGE_ENTRY(topk_merge_r128, TKREG_GIANT)

extern "C" __global__ void topk_rows(
    float* __restrict__ scores,      // [M*N] — MUTATED (winners consumed)
    int* __restrict__ outIdx,        // [M*k]
    float* __restrict__ outScore,    // [M*k]
    const unsigned int* __restrict__ Mp,
    const unsigned int* __restrict__ Np,
    const unsigned int* __restrict__ kp)
{
    unsigned int M = *Mp, N = *Np, k = *kp;
    unsigned int m = blockIdx.x;
    if (m >= M) return;
    float* row = scores + (long)m * N;
    __shared__ float sVal[TKBLOCK];
    __shared__ int sIdx[TKBLOCK];

    for (unsigned int j = 0; j < k; j++) {
        float bv = -3.402823466e+38f;
        int bi = -1;
        for (unsigned int i = threadIdx.x; i < N; i += blockDim.x) {
            float v = row[i];
            // score DESC, then index ASC — the topHits order.
            if (bi < 0 || v > bv || (v == bv && (int)i < bi)) {
                if (v != -3.402823466e+38f || bi < 0) { bv = v; bi = (int)i; }
            }
        }
        sVal[threadIdx.x] = bv;
        sIdx[threadIdx.x] = bi;
        __syncthreads();
        for (unsigned int off = blockDim.x / 2; off > 0; off >>= 1) {
            if (threadIdx.x < off) {
                float ov = sVal[threadIdx.x + off];
                int oi = sIdx[threadIdx.x + off];
                float cv = sVal[threadIdx.x];
                int ci = sIdx[threadIdx.x];
                if (oi >= 0 && (ci < 0 || ov > cv || (ov == cv && oi < ci))) {
                    sVal[threadIdx.x] = ov;
                    sIdx[threadIdx.x] = oi;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            outIdx[m * k + j] = sIdx[0];
            outScore[m * k + j] = sVal[0];
            if (sIdx[0] >= 0) row[sIdx[0]] = -3.402823466e+38f; // consume
        }
        __syncthreads();
    }
}
