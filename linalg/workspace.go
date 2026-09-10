package linalg

// Workspace holds reusable scratch buffers so steady-state matmul calls — the
// W8A8 decode path especially — allocate nothing after warm-up. A decode loop
// calls ~168 matmuls per token; without reuse each allocates its quantized
// activation (and, parallelized, one per worker), which dominated alloc_space
// in goinfer's profile. Thread one Workspace through the hot path instead.
//
// NOT safe for concurrent use: the buffers are shared mutable scratch. Use one
// Workspace per goroutine / decode stream (goinfer owns one per stream, next to
// its KV cache). The zero value is ready to use; buffers grow on demand and are
// never shrunk.
type Workspace struct {
	i8  []int8
	f32 []float32
	// deq is the weight-only Q8/Q4 spans' dequantized-row scratch, kept SEPARATE
	// from f32 on purpose (audit C-05). f32 holds activation scales, whose
	// lifetime QuantizeActivations documents to the caller; the dequant scratch
	// is per-call and K-wide. Sharing one buffer meant a weight-only matmul
	// silently overwrote a caller's held scales with dequantized weight values
	// whenever K >= M — without growing the buffer, so the documented
	// "valid until the next call that grows the scratch" rule did not catch it.
	deq          []float32
	width        int  // per-Workspace fan-out cap; 0 ⇒ inherit SetParallelWidth
	threshold    int  // per-Workspace parallelization threshold (when thresholdSet)
	thresholdSet bool // false ⇒ inherit the process-wide SetParallelThreshold default
}

// SetThreshold overrides the parallelization threshold (see SetParallelThreshold)
// for matmuls run through THIS Workspace, leaving the process-wide default untouched
// — so independent decode streams can tune the whether-to-parallelize decision
// without racing on a global. macs is the MAC count (M*N*K) at/above which a matmul
// parallelizes; macs ≤ 0 forces always-parallel. The zero-value Workspace inherits
// the process default. Pairs with SetWorkers, which scopes the fan-out width.
func (w *Workspace) SetThreshold(macs int) {
	w.threshold = macs
	w.thresholdSet = true
}

// threshold resolves this Workspace's parallelization threshold: its own override if
// set, otherwise the process-wide default.
func (w *Workspace) thr() int {
	if w.thresholdSet {
		return w.threshold
	}
	return parThreshold
}

// parallelCols is the Workspace-scoped sibling of the package parallelCols: it uses
// this Workspace's threshold and width instead of the globals.
func (w *Workspace) parallelCols(work, N int, fn func(j0, j1 int)) {
	if work < w.thr() || N < 2 {
		fn(0, N)
		return
	}
	w.parallel(N, fn)
}

// SetWorkers caps how many worker shards a parallel matmul run through THIS
// Workspace fans out to (0 ⇒ inherit the process-wide SetParallelWidth default).
// Lower it to the P-core count on a heterogeneous CPU (Apple big.LITTLE, Intel P/E)
// to cut E-core stragglers at the fork/join barrier. Per-Workspace, so independent
// decode streams pick their own width without racing on a global, and the zero-value
// Workspace inherits the default. Numerically inert: parallel matmuls partition
// output columns, so any width is bit-identical. Pairs with SetThreshold.
func (w *Workspace) SetWorkers(n int) { w.width = n }

// parallel runs fn over [0,N) as a per-call goroutine fan-out capped at this
// Workspace's width. Caller has already decided the work clears the threshold.
func (w *Workspace) parallel(N int, fn func(j0, j1 int)) {
	parallelSpawnCols(N, resolveWidth(w.width), fn)
}

// int8Buf returns a length-n int8 scratch slice backed by reusable storage.
func (w *Workspace) int8Buf(n int) []int8 {
	if cap(w.i8) < n {
		w.i8 = make([]int8, n)
	}
	return w.i8[:n]
}

// f32Buf returns a length-n float32 scratch slice backed by reusable storage.
// This is the ACTIVATION-SCALE buffer: QuantizeActivations hands views of it to
// callers and documents how long they stay valid, so a per-call scratch must
// not be taken from here. Use deqBuf for that.
func (w *Workspace) f32Buf(n int) []float32 {
	if cap(w.f32) < n {
		w.f32 = make([]float32, n)
	}
	return w.f32[:n]
}

// deqBuf returns a length-n float32 scratch slice for dequantized weight rows,
// backed by storage distinct from f32Buf's. See the deq field comment: the
// weight-only Q8/Q4 spans are the only users, and keeping them off f32 is what
// makes QuantizeActivations' documented lifetime true.
func (w *Workspace) deqBuf(n int) []float32 {
	if cap(w.deq) < n {
		w.deq = make([]float32, n)
	}
	return w.deq[:n]
}

// QuantizeActivations dynamically quantizes a's M×K rows to int8 into this
// Workspace's own reusable int8/scale scratch and returns views of them, for a
// caller that reuses one activation (a query) across many weight blocks and wants
// to quantize it ONCE (lens §3.5). The returned slices are valid until the next
// call that writes the Workspace's int8/scale scratch — any further
// QuantizeActivations, or a W8A8/W4A8 matmul through the same Workspace, which
// takes its own aScales from f32Buf. MatmulBTW8A8Pre does NOT touch them, so the
// paged-scan pattern (quantize once, Pre per block) is safe, and neither do the
// weight-only Q8/Q4 matmuls, whose dequant scratch is deqBuf (audit C-05 — they
// used to share f32Buf and would clobber these scales for K >= M without
// growing the buffer). Not goroutine-safe: one Workspace per user, as elsewhere.
func (w *Workspace) QuantizeActivations(a []float32, M, K int) (aq []int8, scales []float32) {
	aq = w.int8Buf(M * K)
	scales = w.f32Buf(M)
	QuantizeActivationsInto(aq, scales, a, M, K)
	return aq, scales
}
