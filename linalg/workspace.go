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
	i8   []int8
	f32  []float32
	pool *pool // nil ⇒ parallel matmuls spawn goroutines per call (the default)
}

// SetWorkers gives this Workspace a persistent pool of n worker goroutines that
// spin briefly before parking, so the back-to-back matmuls of a decode step
// reuse hot workers instead of spawning + parking goroutines per call. n is the
// total degree of parallelism (the dispatcher counts as one), so n = number of
// cores to use; n ≤ 1 disables the pool (and stops any existing one). Pass the
// P-core count to avoid E-core load-imbalance stalls.
//
// The pool is owned by this Workspace and driven by a SINGLE dispatcher, so —
// like the rest of Workspace — it is NOT safe for concurrent use; keep one per
// decode stream. Call Close to stop the workers when the stream ends, or they
// leak. The zero-value Workspace has no pool, so the allocating matmul wrappers
// (which use a transient Workspace) never start one.
func (w *Workspace) SetWorkers(n int) {
	if w.pool != nil {
		w.pool.close()
		w.pool = nil
	}
	if n > 1 {
		w.pool = newPool(n)
	}
}

// Close stops the Workspace's worker pool, if any. Idempotent. After Close the
// Workspace still works (scratch + spawn-per-call fallback); only the persistent
// pool is gone.
func (w *Workspace) Close() {
	if w.pool != nil {
		w.pool.close()
		w.pool = nil
	}
}

// parallel runs fn over [0,N) using the pool when present, else a per-call
// goroutine fan-out. Caller has already decided the work clears parThreshold.
func (w *Workspace) parallel(N int, fn func(j0, j1 int)) {
	if w.pool != nil {
		w.pool.run(N, fn)
		return
	}
	parallelSpawnCols(N, fn)
}

// int8Buf returns a length-n int8 scratch slice backed by reusable storage.
func (w *Workspace) int8Buf(n int) []int8 {
	if cap(w.i8) < n {
		w.i8 = make([]int8, n)
	}
	return w.i8[:n]
}

// f32Buf returns a length-n float32 scratch slice backed by reusable storage.
func (w *Workspace) f32Buf(n int) []float32 {
	if cap(w.f32) < n {
		w.f32 = make([]float32, n)
	}
	return w.f32[:n]
}
