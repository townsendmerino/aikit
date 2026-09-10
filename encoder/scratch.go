package encoder

import "sync"

// scratchPool gives every forward-pass goroutine a private reusable
// scratch arena. EncodeBatch's static-partition design (M3 + M7)
// runs one forward at a time per worker, so a Get/Put pair around
// each forward keeps the scratch private to that goroutine for the
// duration of the pass. Subsequent forwards on the same worker
// (typical for warm-cache rerank workloads) reuse the same buffers,
// avoiding the per-layer / per-head allocations that M3+M7 profiles
// showed dominating GC time.
var scratchPool = sync.Pool{
	New: func() any { return &scratch{} },
}

func getScratch() *scratch { return scratchPool.Get().(*scratch) }

// putScratch returns a scratch to the pool. It CLEARS the backend first: scratches are
// pooled and shared across models, so a backend left on one would silently leak into the
// next forward — including a forward on a model that never opted in.
func putScratch(s *scratch) {
	s.be = nil
	scratchPool.Put(s)
}

// scratch is a per-forward scratchpad of reusable float32 buffers. The
// alternative (allocating inside every selfAttention call) burns ~432
// small mallocs per single-sequence forward (12 layers × 12 heads × 3
// buffers: qH, kH, vH) and a handful of larger ones (qkv, Q, K, V,
// ctx, out, scores). At rerankN=50 that's ~20k mallocs per query —
// real GC pressure visible in M3/M7 traces.
//
// One scratch is held per goroutine. EncodeBatch's static-partition
// design (M3 + M7) means each worker runs one forward at a time, so
// a scratch on the goroutine's stack-allocated *scratch is private.
// Per-call we ENSURE each buffer is at least the size we need, then
// reuse. The buffer slices grow monotonically across the 12 layers
// (within a single forward they stay the same size); the next
// forward on the same worker reuses them.
//
// Buffers are sized for the maximum L and D the forward will see.
// For single-sequence: cap=L; for batched: cap=B*Lmax. Caller passes
// the right cap via ensure*.
type scratch struct {
	// be is the compute backend for this forward's f32 matmuls — nil means the
	// pure-Go path, which is the default and is BIT-IDENTICAL to not having this
	// field at all (mm dispatches to exactly the function the call sites used to
	// call). Set from the model at each getScratch site; cleared by putScratch.
	be Backend

	// Linear-layer scratch (sized to L*3*D = QKV output, or L*D for
	// projections). qkv is reused across the 12 selfAttention calls;
	// Q/K/V/ctx are split-out per-call buffers.
	qkv []float32 // [L*3*D]
	Q   []float32 // [L*D]
	K   []float32 // [L*D]
	V   []float32 // [L*D]
	ctx []float32 // [L*D]
	out []float32 // [L*D] — attention out_proj output
	// MLP scratch.
	val  []float32 // [L*intermediate]
	gate []float32 // [L*intermediate]
	mid  []float32 // [L*D] — MLP fc2 output
	// upGate is the fused up/gate projection output [L, 2*intermediate] for
	// models (GTE) whose MLP does ONE 2I-wide matmul instead of two I-wide ones.
	// It cannot be served by val+gate: those are two separate [L,I] buffers,
	// while this is one row-major block whose rows interleave up and gate —
	// which is exactly what the single fused matmul writes. Sized only on
	// request (ensureFusedMLP), so the BERT/Nomic scratches never carry it.
	upGate []float32
	// Per-head extracts (sized perHeadLen*headDim each). vH holds V
	// TRANSPOSED to [headDim, perHeadLen] so the scores·V context step can run
	// through the SIMD A·Bᵀ matmul instead of a scalar triple-loop.
	qH []float32
	kH []float32
	vH []float32
	// ctxHead is the per-head scores·V output [perHeadLen, headDim], scattered
	// into the interleaved ctx[L, D] afterward.
	ctxHead []float32
	// Attention scores [L, L].
	scores []float32
	// MoE scratch (moeMLP only): moeScores holds the per-token router logits
	// [numExperts]; moeOut is the per-token expert-combination accumulator [D].
	// x1 (the W1 output) and contrib (the per-expert W2 output) reuse val/mid,
	// exactly as gelu/swigluMLP reuse them. All are pooled so the MoE layers of a
	// forward allocate nothing after warmup (audit #10).
	moeScores []float32
	moeOut    []float32
	// Expert-grouping scratch (perf-campaign item 33). moeMLP used to run both
	// expert projections at M=1, once per (token, rank) — 2048 single-row GEMM
	// calls per MoE layer at L=512, top-k 2. These let it batch the tokens
	// routed to the same expert into one call instead.
	moeAllScores []float32 // [L*numExperts] router logits for every token
	moeExpert    []int32   // [L*topK] chosen expert per (token, rank)
	moeWeight    []float32 // [L*topK] router weight per (token, rank)
	moeTokens    []int32   // [L] token ids gathered for the expert in hand
	moeGathered  []float32 // [L*D] their rows, made contiguous for the GEMM
	moeAcc       []float32 // [L*D] combined expert output per token
	// deqW holds one int8 weight matrix dequantized to f32 for the q8 linear path:
	// matmulBTQ8Into widens the int8 weights here ONCE per matmul (N*K) and then runs
	// the SIMD f32 matmulBTInto, instead of the scalar inline-widen (which redid the
	// widen M times). Pooled, so the 60 matmuls of a forward reuse it. The stored
	// weights stay int8 (¼ memory); this is transient runtime scratch.
	deqW []float32
}

// ensureDeqW sizes the q8 weight-dequant buffer to the largest weight matrix the
// forward will dequantize: N*K over {Wqkv 3D×D, OutProj D×D, fc1 inter×D, fc2 D×inter}
// → D·max(3D, intermediate). Only the q8 path calls this, so f32 forwards don't pay it.
func (s *scratch) ensureDeqW(D, intermediate int) {
	n := max(intermediate, 3*D)
	s.deqW = ensureF32(s.deqW, n*D)
}

// ensureF32 grows b to capacity n (returning a slice of length n).
// Reuses the underlying array when n ≤ cap(b); allocates a new one
// 25% bigger than n otherwise so subsequent calls with similar
// sizes don't reallocate.
func ensureF32(b []float32, n int) []float32 {
	if cap(b) >= n {
		return b[:n]
	}
	return make([]float32, n, n+n/4)
}

// ensureI32 is ensureF32 for the MoE routing tables (item 33).
func ensureI32(b []int32, n int) []int32 {
	if cap(b) >= n {
		return b[:n]
	}
	return make([]int32, n, n+n/4)
}

// ensureLayer sizes the scratch buffers for one forward pass.
//
//   - L is the per-row count for the BATCHED scratch slices
//     (qkv/Q/K/V/ctx/out/val/gate/mid) — pass B*Lmax in the batched
//     path, L in the single-seq path.
//   - perHeadLen is the per-(b, head) inner-loop bound, used to size
//     qH/kH/vH/scores. Pass Lmax in the batched path (the longest
//     sequence in the batch) and L in the single-seq path. Pre-M11
//     this was conflated with L which over-sized scores to (B*Lmax)²
//     in the batched path — wasted memory the M11 attention alloc
//     fix surfaced.
func (s *scratch) ensureLayer(L, D, intermediate, heads, headDim, perHeadLen int) {
	s.qkv = ensureF32(s.qkv, L*3*D)
	s.Q = ensureF32(s.Q, L*D)
	s.K = ensureF32(s.K, L*D)
	s.V = ensureF32(s.V, L*D)
	s.ctx = ensureF32(s.ctx, L*D)
	s.out = ensureF32(s.out, L*D)
	s.val = ensureF32(s.val, L*intermediate)
	s.gate = ensureF32(s.gate, L*intermediate)
	s.mid = ensureF32(s.mid, L*D)
	s.qH = ensureF32(s.qH, perHeadLen*headDim)
	s.kH = ensureF32(s.kH, perHeadLen*headDim)
	s.vH = ensureF32(s.vH, perHeadLen*headDim)
	s.ctxHead = ensureF32(s.ctxHead, perHeadLen*headDim)
	s.scores = ensureF32(s.scores, perHeadLen*perHeadLen)
}

// ensureFusedMLP sizes the fused up/gate buffer. Kept out of ensureLayer so the
// models that never use it (BERT, Nomic) do not carry an extra L*2*intermediate
// span in every pooled scratch — at L=690/I=3072 that is 17 MB, the bulk of
// perf-campaign item 8.
func (s *scratch) ensureFusedMLP(L, intermediate int) {
	s.upGate = ensureF32(s.upGate, L*2*intermediate)
}

// mm is the encoder's f32 matmul dispatch point: dst[M,N] = a[M,K] · b[N,K]ᵀ.
//
// This is the seam Phase 4 needed. Before it, encoder.Backend existed, NewBackend
// resolved it and goinfer/gpu registered "webgpu", but nothing ever CALLED
// Backend.MatmulBT — the hot path went straight to matmulBTInto, so every registered
// backend was inert. Routing through here is what makes a backend do anything.
//
// With no backend (the default) this is a direct call to matmulBTInto, so the pure-Go
// numerics are unchanged — not "equivalent", the same function.
//
// A backend sees EVERY f32 matmul in the forward, including the small per-head QKᵀ and
// scores·V. Deciding which shapes are worth a device round-trip is the BACKEND's job,
// not this layer's: matmulBT's own 4-MFLOP naive/blocked split is the precedent for
// where that policy belongs. A backend that offloads unconditionally will be slower on
// short sequences.
func (s *scratch) mm(a, b, dst []float32, M, K, N int) {
	if s.be != nil {
		s.be.MatmulBT(a, b, dst, M, K, N)
		return
	}
	matmulBTInto(a, b, dst, M, K, N)
}

// mmq8 is the encoder's INT8 matmul dispatch point: dst[M,N] = a[M,K] · dequant(wq)[N,K]ᵀ.
//
// It is the int8 twin of mm, and it exists because forward_q8.go called matmulBTQ8Into
// directly, so int8 encoders got no GPU acceleration at all even with a backend
// attached — the f32 seam simply did not reach them.
//
// WHAT THIS PATH IS, AND WHAT IT IS NOT. The encoder's int8 is WEIGHT-ONLY: the
// weights are stored int8 and widened to f32, then multiplied against f32 activations
// (matmulBTQ8Into). It is NOT W8A8 — the activations are never quantized. That is a
// deliberate, measured choice: full W8A8 was tried and fell below the 0.97 reranker
// bar that TestModelQ8_cosineMatchesF32 still enforces, while weight-only holds cosine
// 0.997 vs f32. A backend that "accelerates int8" by quantizing activations would
// silently reintroduce that regression, so the Q8Backend contract below is explicitly
// weight-only.
//
// A backend that declines (false) — too small, no device, an error — leaves the CPU
// path to run exactly as before.
func (s *scratch) mmq8(dst, a []float32, wq []int8, wscales []float32, M, K, N int) {
	if q8, ok := s.be.(Q8Backend); ok && q8 != nil {
		if q8.MatmulBTQ8(dst, a, wq, wscales, M, K, N) {
			return
		}
	}
	matmulBTQ8Into(dst, a, wq, wscales, M, K, N, s.deqW)
}

// headScratch is one attention head's private working set, for the head-parallel
// path (audit M-05). The per-forward scratch holds exactly one of each of these,
// which is all the serial loop needs; a fan-out needs one set per worker.
//
// Pooled rather than sized into scratch as [workers]x: the serial path then pays
// nothing at all, and the parallel path allocates only on its first few calls.
// scores alone is mOut*L floats — 1 MB at L=512 — so multiplying the per-forward
// arena by numCPU would cost real memory on every forward to serve the case that
// fans out.
type headScratch struct {
	qH, kH, vH, ctxHead, scores []float32
}

var headScratchPool = sync.Pool{New: func() any { return new(headScratch) }}

func getHeadScratch(mOut, headDim, L int) *headScratch {
	hs := headScratchPool.Get().(*headScratch)
	hs.qH = ensureF32(hs.qH, mOut*headDim)
	hs.kH = ensureF32(hs.kH, L*headDim)
	hs.vH = ensureF32(hs.vH, headDim*L)
	hs.ctxHead = ensureF32(hs.ctxHead, mOut*headDim)
	hs.scores = ensureF32(hs.scores, mOut*L)
	// Reslice to the exact lengths attendOneHead indexes; ensureF32 only
	// guarantees capacity.
	hs.qH, hs.kH = hs.qH[:mOut*headDim], hs.kH[:L*headDim]
	hs.vH, hs.ctxHead = hs.vH[:headDim*L], hs.ctxHead[:mOut*headDim]
	hs.scores = hs.scores[:mOut*L]
	return hs
}

func putHeadScratch(hs *headScratch) { headScratchPool.Put(hs) }
