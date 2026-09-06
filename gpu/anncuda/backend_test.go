//go:build linux

package anncuda

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/ann"
	gpu "github.com/townsendmerino/aikit/gpu"
)

// randUnit builds n random L2-normalized dim-vectors (the FlatI8 invariant).
func randUnit(rng *rand.Rand, n, dim int) [][]float32 {
	out := make([][]float32, n)
	for i := range out {
		v := make([]float32, dim)
		var s float64
		for d := range v {
			v[d] = float32(rng.NormFloat64())
			s += float64(v[d]) * float64(v[d])
		}
		inv := float32(1 / math.Sqrt(s))
		for d := range v {
			v[d] *= inv
		}
		out[i] = v
	}
	return out
}

// TestCUDAGEMV_parityWithCPU is the Phase-1b gate — the CUDA twin of annmetal's
// TestMetalGEMV_parityWithCPU: an ann.FlatI8 scored on the GPU (EnableGPU) must
// return exactly the CPU FlatI8's top-k — same indices in the same order, scores
// within the int8/float tolerance — over many random queries. The int8 dot is
// exact integer arithmetic, so the rankings are identical, not merely close.
// Break-it-first below proves the equality check is not vacuous.
func TestCUDAGEMV_parityWithCPU(t *testing.T) {
	if !ann.HasBackend() {
		t.Skip("no CUDA backend registered (no GPU?)")
	}
	rng := rand.New(rand.NewSource(1))
	const n, dim, k = 2000, 96, 10
	vecs := randUnit(rng, n, dim)

	cpu := ann.NewFlatI8(vecs)
	gpu := ann.NewFlatI8(vecs)
	if err := gpu.EnableGPU(); err != nil {
		t.Fatalf("EnableGPU: %v", err)
	}
	defer gpu.Close()
	if !gpu.GPUEnabled() {
		t.Fatal("GPUEnabled false after EnableGPU")
	}
	t.Logf("backend=%q n=%d dim=%d k=%d", ann.BackendName(), n, dim, k)

	queries := randUnit(rng, 200, dim)
	worstScoreΔ := 0.0
	for qi, q := range queries {
		want := cpu.Query(q, k)
		got := gpu.Query(q, k)
		if len(got) != len(want) {
			t.Fatalf("q%d: got %d hits, want %d", qi, len(got), len(want))
		}
		for i := range want {
			if got[i].Index != want[i].Index {
				t.Fatalf("q%d rank %d: GPU index %d != CPU index %d (top-k diverged)", qi, i, got[i].Index, want[i].Index)
			}
			if d := math.Abs(got[i].Score - want[i].Score); d > worstScoreΔ {
				worstScoreΔ = d
			}
		}
	}
	t.Logf("GPU≡CPU top-%d over %d queries: worst score Δ %.3e", k, len(queries), worstScoreΔ)
	if worstScoreΔ > 1e-3 {
		t.Errorf("worst score Δ %.3e too large — GPU rescale diverges from CPU", worstScoreΔ)
	}

	// Break-it-first: the equality gate must be able to FAIL. Score the SAME GPU
	// index but compare against a DIFFERENT index's top-k — it must diverge, proving
	// the per-query comparison is discriminating (not "any two rankings match").
	other := ann.NewFlatI8(randUnit(rng, n, dim))
	diverged := 0
	for _, q := range queries {
		g := gpu.Query(q, k)
		o := other.Query(q, k)
		for i := range g {
			if g[i].Index != o[i].Index {
				diverged++
				break
			}
		}
	}
	t.Logf("break-it-first: %d/%d queries' top-k differ from an unrelated index", diverged, len(queries))
	if diverged == 0 {
		t.Error("break-it-first vacuous: GPU top-k matched an unrelated index on every query")
	}
}

// Compile-time proof that the CUDA index is a batch index — so FlatI8.QueryBatch
// takes the batched-GEMM path (ScoreBatch), never the per-query fallback.
var _ ann.I8BatchIndex = (*cudaI8Index)(nil)

// TestCUDAGEMM_batchParityWithCPU is the batched gate (Phase 2's shape, on CUDA):
// scoring a batch of queries as one int8 GEMM on the GPU must return exactly the
// CPU FlatI8's per-query top-k for every query in the batch — same indices, scores
// within tolerance. The int8 dot is exact, so the rankings are identical.
func TestCUDAGEMM_batchParityWithCPU(t *testing.T) {
	if !ann.HasBackend() {
		t.Skip("no CUDA backend registered (no GPU?)")
	}
	rng := rand.New(rand.NewSource(7))
	const n, dim, k, M = 3000, 128, 10, 16
	vecs := randUnit(rng, n, dim)

	cpu := ann.NewFlatI8(vecs)
	gpu := ann.NewFlatI8(vecs)
	if err := gpu.EnableGPU(); err != nil {
		t.Fatalf("EnableGPU: %v", err)
	}
	defer gpu.Close()

	queries := randUnit(rng, M, dim)
	batch := gpu.QueryBatch(queries, k)
	if len(batch) != M {
		t.Fatalf("QueryBatch returned %d rows, want %d", len(batch), M)
	}
	worst := 0.0
	for m, q := range queries {
		want := cpu.Query(q, k)
		got := batch[m]
		if len(got) != len(want) {
			t.Fatalf("q%d: %d hits, want %d", m, len(got), len(want))
		}
		for i := range want {
			if got[i].Index != want[i].Index {
				t.Fatalf("q%d rank %d: GPU-batch index %d != CPU index %d (GEMM diverged)", m, i, got[i].Index, want[i].Index)
			}
			if d := math.Abs(got[i].Score - want[i].Score); d > worst {
				worst = d
			}
		}
	}
	t.Logf("GPU-batch(M=%d) ≡ CPU per-query top-%d: worst score Δ %.3e", M, k, worst)
	if worst > 1e-3 {
		t.Errorf("worst score Δ %.3e too large", worst)
	}

	// Vacuity: distinct queries must give distinct top hits (the batch is not
	// collapsing every row to the same answer).
	distinct := false
	for m := 1; m < M; m++ {
		if len(batch[m]) > 0 && len(batch[0]) > 0 && batch[m][0].Index != batch[0][0].Index {
			distinct = true
			break
		}
	}
	if !distinct {
		t.Error("break-it-first vacuous: every batch row shared the same top hit")
	}
}

// TestCUDA_shapesOffBlockBoundary pushes shapes that are NOT multiples of the
// 256-thread block through both kernels — the CUDA-specific geometry the Metal
// path never sees, since dispatchThreads launches exactly n where a CUDA launch
// rounds up to whole blocks. It gates that no row is dropped or misindexed when
// the grid overhangs N (or M*N).
//
// It does NOT gate the kernels' bounds checks, and mutation testing is how we
// know: deleting `if (j >= *N) return;` from gemv_w8a8 leaves every test here
// PASSING. The overhanging threads write just past out[N], which lands in the
// allocation's alignment slack — silent corruption, not a wrong answer. The gate
// that does catch it is TestCUDA_tailBlockGuard in the device layer, which
// allocates a sentinel canary past the output and checks it survives (deleting
// the guard there fails 3 assertions). Guard changes belong to that test.
func TestCUDA_shapesOffBlockBoundary(t *testing.T) {
	if !ann.HasBackend() {
		t.Skip("no CUDA backend registered (no GPU?)")
	}
	rng := rand.New(rand.NewSource(11))
	const k = 5
	// n values straddling the 256-block boundary; dims that are not multiples of 4
	// (so the K loop's tail is exercised too).
	for _, shape := range []struct{ n, dim, m int }{
		{1, 7, 1},      // single row, single query
		{255, 33, 3},   // one short of a block
		{256, 64, 1},   // exactly one block
		{257, 17, 2},   // one past a block
		{1023, 31, 7},  // large, prime-ish
		{97, 23, 9},    // M just past batchSmallMaxM — the wide kernel, partial tile
		{513, 12, 16},  // wide kernel, tile exactly full
		{31, 5, 17},    // n below one lane-group's worth; M spills to a second tile
		{2049, 40, 33}, // three tiles, and n well past a block
	} {
		vecs := randUnit(rng, shape.n, shape.dim)
		cpu := ann.NewFlatI8(vecs)
		gpu := ann.NewFlatI8(vecs)
		if err := gpu.EnableGPU(); err != nil {
			t.Fatalf("n=%d dim=%d EnableGPU: %v", shape.n, shape.dim, err)
		}
		queries := randUnit(rng, shape.m, shape.dim)

		// Single-query path.
		for qi, q := range queries {
			want, got := cpu.Query(q, k), gpu.Query(q, k)
			if len(got) != len(want) {
				t.Fatalf("n=%d dim=%d q%d: %d hits, want %d", shape.n, shape.dim, qi, len(got), len(want))
			}
			for i := range want {
				if got[i].Index != want[i].Index {
					t.Fatalf("n=%d dim=%d q%d rank %d: GEMV index %d != CPU %d", shape.n, shape.dim, qi, i, got[i].Index, want[i].Index)
				}
			}
		}
		// Batched path (M*N is the launch bound there).
		batch := gpu.QueryBatch(queries, k)
		for m, q := range queries {
			want := cpu.Query(q, k)
			if len(batch[m]) != len(want) {
				t.Fatalf("n=%d dim=%d M=%d q%d: %d hits, want %d", shape.n, shape.dim, shape.m, m, len(batch[m]), len(want))
			}
			for i := range want {
				if batch[m][i].Index != want[i].Index {
					t.Fatalf("n=%d dim=%d M=%d q%d rank %d: GEMM index %d != CPU %d", shape.n, shape.dim, shape.m, m, i, batch[m][i].Index, want[i].Index)
				}
			}
		}
		gpu.Close()
	}
	t.Log("GEMV+batch ≡ CPU across 9 shapes straddling the block, lane-group and query-tile boundaries")
}

// TestCUDA_concurrentScore runs many goroutines against one GPU index. A CUDA
// context is thread-affine, and a per-call dispatch from a migrating goroutine is
// the exact crash class that hit the Metal path through NSAutoreleasePool. Nothing
// here pins a thread because gocudrv's Context owns a LockOSThread'd executor and
// funnels every driver call through it (gpu/cuda.go, "Thread affinity") — this
// test is what holds that claim honest, and also covers the per-index mutex that
// guards the shared scratch buffers.
func TestCUDA_concurrentScore(t *testing.T) {
	if !ann.HasBackend() {
		t.Skip("no CUDA backend registered (no GPU?)")
	}
	rng := rand.New(rand.NewSource(23))
	const n, dim, k, workers, iters = 512, 48, 5, 8, 25
	vecs := randUnit(rng, n, dim)
	cpu := ann.NewFlatI8(vecs)
	gpu := ann.NewFlatI8(vecs)
	if err := gpu.EnableGPU(); err != nil {
		t.Fatalf("EnableGPU: %v", err)
	}
	defer gpu.Close()

	queries := randUnit(rng, workers, dim)
	want := make([][]ann.Hit, workers)
	for i, q := range queries {
		want[i] = cpu.Query(q, k)
	}

	errs := make(chan string, workers)
	done := make(chan struct{})
	for w := range workers {
		go func() {
			defer func() { done <- struct{}{} }()
			for range iters {
				got := gpu.Query(queries[w], k)
				if len(got) != len(want[w]) {
					errs <- "hit count changed under concurrency"
					return
				}
				for i := range want[w] {
					if got[i].Index != want[w][i].Index {
						errs <- "top-k diverged under concurrency"
						return
					}
				}
			}
		}()
	}
	for range workers {
		<-done
	}
	close(errs)
	for e := range errs {
		t.Error(e)
	}
	t.Logf("%d goroutines × %d queries: top-%d stable and CPU-identical", workers, iters, k)
}

// --- on-device top-k (ann.I8TopKIndex) ---

// Compile-time proof the CUDA index offers the top-k capability, so
// FlatI8.QueryBatch prefers it over score-then-host-select.
var _ ann.I8TopKIndex = (*cudaI8Index)(nil)

// TestCUDATopK_matchesCPU gates the device selection against the CPU's own top-k over
// an index large enough to clear topkMinN. The hits must be IDENTICAL — same indices in
// the same order — not merely a plausible set: the kernel reimplements topHits's
// ordering, so anything else means the two disagree about what "top-k" is.
func TestCUDATopK_matchesCPU(t *testing.T) {
	if !ann.HasBackend() {
		t.Skip("no CUDA backend registered (no GPU?)")
	}
	rng := rand.New(rand.NewSource(101))
	const n, dim, k, M = 60_000, 64, 10, 8 // n > topkMinN so the device path engages
	vecs := randUnit(rng, n, dim)
	cpu := ann.NewFlatI8(vecs)
	g := ann.NewFlatI8(vecs)
	if err := g.EnableGPU(); err != nil {
		t.Fatalf("EnableGPU: %v", err)
	}
	defer g.Close()

	queries := randUnit(rng, M, dim)
	got := g.QueryBatch(queries, k)
	for m, q := range queries {
		want := cpu.Query(q, k)
		if len(got[m]) != len(want) {
			t.Fatalf("q%d: %d hits, want %d", m, len(got[m]), len(want))
		}
		for i := range want {
			if got[m][i].Index != want[i].Index {
				t.Fatalf("q%d rank %d: device top-k index %d != CPU %d", m, i, got[m][i].Index, want[i].Index)
			}
			if math.Abs(got[m][i].Score-want[i].Score) > 1e-3 {
				t.Fatalf("q%d rank %d: score %v vs CPU %v", m, i, got[m][i].Score, want[i].Score)
			}
		}
	}
	t.Logf("device top-k ≡ CPU top-%d over %d queries at n=%d", k, M, n)
}

// TestCUDATopK_tieBreak is the sharp one. With EVERY score equal, the ONLY thing
// deciding the answer is the tie-break, so the result must be exactly [0..k-1] —
// index ASC. A kernel that reduced on score alone would return an arbitrary k here and
// still look correct on random data, which is precisely how a wrong tie-break survives.
func TestCUDATopK_tieBreak(t *testing.T) {
	if !ann.HasBackend() {
		t.Skip("no CUDA backend registered")
	}
	const n, dim, k, M = 60_000, 32, 12, 3
	// Every vector identical ⇒ every score identical for any query.
	vecs := make([][]float32, n)
	base := make([]float32, dim)
	for d := range base {
		base[d] = float32(1.0 / math.Sqrt(float64(dim)))
	}
	for i := range vecs {
		vecs[i] = base
	}
	g := ann.NewFlatI8(vecs)
	if err := g.EnableGPU(); err != nil {
		t.Fatalf("EnableGPU: %v", err)
	}
	defer g.Close()

	queries := make([][]float32, M)
	for m := range queries {
		queries[m] = base
	}
	got := g.QueryBatch(queries, k)
	for m := range M {
		if len(got[m]) != k {
			t.Fatalf("q%d: %d hits, want %d", m, len(got[m]), k)
		}
		for i := range k {
			if got[m][i].Index != i {
				t.Fatalf("q%d rank %d: index %d, want %d — all scores are equal, so the tie-break (index ASC) is the whole answer",
					m, i, got[m][i].Index, i)
			}
		}
	}
	t.Log("all-equal scores ⇒ top-k is [0..k-1]: tie-break matches topHits")
}

// newTestIndex builds a backend + index directly, so the decline paths can be
// exercised on the concrete type. Reaching them through FlatI8 is not possible — the
// index is unexported there — and asserting nothing would make this test vacuous.
func newTestIndex(t *testing.T, vecs [][]float32) *cudaI8Index {
	t.Helper()
	dev, err := gpu.CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	lib, err := dev.CompileLibrary(w8a8PTX)
	if err != nil {
		dev.ReleaseObjects()
		t.Fatalf("CompileLibrary: %v", err)
	}
	gemv, err := dev.NewComputePipeline(lib, "gemv_w8a8")
	if err != nil {
		dev.ReleaseObjects()
		t.Fatalf("gemv: %v", err)
	}
	topk, err := dev.NewComputePipeline(lib, "topk_rows")
	if err != nil {
		dev.ReleaseObjects()
		t.Fatalf("topk: %v", err)
	}
	batchSmall, err := dev.NewComputePipeline(lib, "gemv_w8a8_batch8")
	if err != nil {
		dev.ReleaseObjects()
		t.Fatalf("batch8: %v", err)
	}
	batchWide, err := dev.NewComputePipeline(lib, "gemv_w8a8_batch16")
	if err != nil {
		dev.ReleaseObjects()
		t.Fatalf("batch16: %v", err)
	}
	b := &cudaBackend{dev: dev, q: dev.NewCommandQueue(), gemv: gemv,
		batchSmall: batchSmall, batchWide: batchWide, topk: topk}
	b.sm = dev.SMCount()
	for i, w := range topkWidths {
		for _, load := range []struct {
			name string
			dst  *gpu.Pipeline
		}{
			{topkKernelName("rows", w), &b.topkReg[i]},
			{topkKernelName("split", w), &b.topkSplit[i]},
			{topkKernelName("merge", w), &b.topkMerge[i]},
		} {
			p, err := dev.NewComputePipeline(lib, load.name)
			if err != nil {
				dev.ReleaseObjects()
				t.Fatalf("%s: %v", load.name, err)
			}
			*load.dst = p
		}
	}
	t.Cleanup(dev.ReleaseObjects)

	// Quantize with the package's own quantizeRowInt8 — byte-identical to what
	// FlatI8 stores, so the index under test holds the same codes it would in
	// production.
	n, dim := len(vecs), len(vecs[0])
	bq := make([]int8, n*dim)
	scales := make([]float32, n)
	for i, v := range vecs {
		scales[i] = quantizeRowInt8(v, bq[i*dim:(i+1)*dim])
	}
	idx, err := b.NewI8Index(bq, scales, n, dim)
	if err != nil {
		t.Fatalf("NewI8Index: %v", err)
	}
	return idx.(*cudaI8Index)
}

// TestCUDATopK_declines pins the fallback contract: outside the range the kernel
// implements, TopKBatch must decline so QueryBatch takes the scoring path — and the
// answer must still be right.
func TestCUDATopK_declines(t *testing.T) {
	if !ann.HasBackend() {
		t.Skip("no CUDA backend registered")
	}
	rng := rand.New(rand.NewSource(102))
	const dim, k, small = 48, 5, 2_000
	vecs := randUnit(rng, small, dim)
	idx := newTestIndex(t, vecs)
	defer func() { _ = idx.Close() }()

	q := [][]float32{randUnit(rng, 1, dim)[0]}
	if _, err := idx.TopKBatch(q, k); err == nil {
		t.Errorf("TopKBatch should decline at n=%d < topkMinN=%d", small, topkMinN)
	}
	if _, err := idx.TopKBatch(q, small); err == nil {
		t.Error("TopKBatch should decline for k >= n (topHits's full-sort path)")
	}
	if _, err := idx.TopKBatch(q, 0); err == nil {
		t.Error("TopKBatch should decline for k <= 0")
	}

	// A declining index must leave QueryBatch's answer identical to the CPU's.
	cpu := ann.NewFlatI8(vecs)
	g := ann.NewFlatI8(vecs)
	if err := g.EnableGPU(); err != nil {
		t.Fatalf("EnableGPU: %v", err)
	}
	defer g.Close()
	queries := randUnit(rng, 4, dim)
	got := g.QueryBatch(queries, k)
	for m, qq := range queries {
		want := cpu.Query(qq, k)
		for i := range want {
			if got[m][i].Index != want[i].Index {
				t.Fatalf("q%d rank %d after decline: %d != CPU %d", m, i, got[m][i].Index, want[i].Index)
			}
		}
	}
	t.Log("declines below topkMinN, for k>=n and k<=0; the fallback answer is unchanged")
}

// TestCUDAGEMM_batchMatchesQuery gates the batched kernels against the single-query
// one at every width where the routing or the tiling changes behaviour.
//
// The widths are chosen, not sampled. batchSmallMaxM (8) selects between the two
// instantiations, and each pads its query tile to QTILE (8 and 16), so M=8/9 crosses
// the kernel choice, M=9/16/17 crosses the wide kernel's tile boundary, and M=1 takes
// the GEMV reroute. The failure this catches is a mis-sized launch: GridY is
// ceil(M/QTILE) and shared memory is QTILE*K, both computed host-side from constants
// that are duplicated from the .cu, so a wrong one drops a whole tile of queries or
// truncates the staged tile — and either way the surviving rows still look plausible.
//
// Exact equality is the right bar rather than a tolerance: the batch kernels compute
// the same int32 dot in a different lane order, and integer addition is associative,
// so any difference at all is a bug rather than rounding.
func TestCUDAGEMM_batchMatchesQuery(t *testing.T) {
	if !ann.HasBackend() {
		t.Skip("no CUDA backend registered (no GPU?)")
	}
	rng := rand.New(rand.NewSource(7))
	const n, dim, k = 5000, 128, 10
	vecs := randUnit(rng, n, dim)

	idx := ann.NewFlatI8(vecs)
	if err := idx.EnableGPU(); err != nil {
		t.Fatalf("EnableGPU: %v", err)
	}
	defer func() { _ = idx.Close() }()

	for _, M := range []int{1, 2, 7, 8, 9, 16, 17, 33} {
		queries := randUnit(rng, M, dim)
		got := idx.QueryBatch(queries, k)
		if len(got) != M {
			t.Fatalf("M=%d: QueryBatch returned %d result sets", M, len(got))
		}
		for qi, q := range queries {
			want := idx.Query(q, k)
			if len(got[qi]) != len(want) {
				t.Fatalf("M=%d q%d: batch returned %d hits, Query returned %d", M, qi, len(got[qi]), len(want))
			}
			for i := range want {
				if got[qi][i].Index != want[i].Index || got[qi][i].Score != want[i].Score {
					t.Fatalf("M=%d q%d hit %d: batch {%d, %v} != Query {%d, %v}",
						M, qi, i, got[qi][i].Index, got[qi][i].Score, want[i].Index, want[i].Score)
				}
			}
		}
		// Vacuity: a batch that collapsed every row to the same query would still
		// pass the loop above if Query were consulted per row, so check the rows differ.
		if M > 1 {
			distinct := false
			for m := 1; m < M; m++ {
				if len(got[m]) > 0 && len(got[0]) > 0 && got[m][0].Index != got[0][0].Index {
					distinct = true
					break
				}
			}
			if !distinct {
				t.Errorf("M=%d: every batch row shared the same top hit", M)
			}
		}
	}
	t.Logf("batch ≡ Query exactly at M ∈ {1,2,7,8,9,16,17,33} (n=%d dim=%d k=%d)", n, dim, k)
}

// TestCUDATopK_kernelsAgree is the gate the top-k routing rests on: topkPipeline picks
// among five kernels by k alone, on the claim that they are interchangeable.
//
// IT LAUNCHES THE KERNELS DIRECTLY, on a synthetic score matrix, because going through
// TopKBatch cannot construct the input that discriminates. The four register kernels
// keep TKREG candidates PER THREAD, and thread t sees only indices t, t+256, t+512, …
// A kernel is only wrong if one thread's stride holds MORE than TKREG of the global
// top-k — and on random scores that essentially never happens. Mutation-checked: with
// a random fixture, routing k≤16 to the 8-wide kernel passed. The clustered row below
// puts the top 64 values at indices 0, 256, 512, … so thread 0 holds ALL of them, and
// the same mutation then fails at every k > 8.
//
// Three rows, each targeting a different failure:
//
//	clustered — the top values all in one thread's stride (per-thread capacity)
//	tied      — scores from a tiny integer range, so the tie-break decides the order
//	            and a kernel returning the right SET in the wrong order is still wrong
//	random    — the ordinary case, which none of the above would notice breaking
//
// Comparison is index-for-index against a CPU selection using the same rule the kernels
// must implement: score descending, lower index wins ties.
func TestCUDATopK_kernelsAgree(t *testing.T) {
	dev, err := gpu.CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	t.Cleanup(dev.ReleaseObjects)
	lib, err := dev.CompileLibrary(w8a8PTX)
	if err != nil {
		t.Fatalf("CompileLibrary: %v", err)
	}
	q := dev.NewCommandQueue()

	// N values straddling every residue mod 4 — see the alignment note above.
	for _, N := range []int{60_000, 60_001, 60_002, 60_003} {
		topkKernelsAgreeAt(t, dev, lib, q, N)
	}
	t.Logf("5 kernels ≡ CPU selection over clustered/tied/random rows, 14 values of k, N ≡ 0..3 mod 4")
}

func topkKernelsAgreeAt(t *testing.T, dev *gpu.Device, lib gpu.Library, q gpu.Queue, N int) {
	t.Helper()
	const M = 3
	rng := rand.New(rand.NewSource(23))
	scores := make([]float32, M*N)
	for i := range scores[0:N] { // row 0: clustered into thread 0's stride
		scores[i] = float32(rng.Intn(50))
	}
	for j := range 96 {
		if j*tkBlockThreads < N {
			scores[j*tkBlockThreads] = float32(10_000 - j) // strictly above the rest
		}
	}
	for i := N; i < 2*N; i++ { // row 1: tie-dense
		scores[i] = float32(rng.Intn(5))
	}
	for i := 2 * N; i < 3*N; i++ { // row 2: ordinary
		scores[i] = rng.Float32()
	}

	kernels := []string{"topk_rows_r8", "topk_rows_r16", "topk_rows_r32", "topk_rows_r64", "topk_rows"}
	maxK := map[string]int{"topk_rows_r8": 8, "topk_rows_r16": 16, "topk_rows_r32": 32,
		"topk_rows_r64": 64, "topk_rows": 1 << 30}

	sb := gpu.NewBufferOf(dev, scores)
	mB := gpu.NewBufferOf(dev, []uint32{uint32(M)})
	nB := gpu.NewBufferOf(dev, []uint32{uint32(N)})
	for _, k := range []int{1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100} {
		want := cpuTopK(scores, M, N, k)
		idxB := gpu.NewBufferLenOf[int32](dev, M*k)
		valB := dev.NewBufferLen(M * k)
		kB := gpu.NewBufferOf(dev, []uint32{uint32(k)})
		for _, name := range kernels {
			if k > maxK[name] {
				continue
			}
			p, err := dev.NewComputePipeline(lib, name)
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			// topk_rows consumes winners in place, so every kernel gets a fresh row.
			if err := gpu.Upload(sb, scores); err != nil {
				t.Fatalf("upload: %v", err)
			}
			if err := q.Launch(p, gpu.LaunchConfig{
				GridX: M, GridY: 1, GridZ: 1, BlockX: tkBlockThreads, BlockY: 1, BlockZ: 1,
			}, gpu.Arg(sb), gpu.Arg(idxB), gpu.Arg(valB),
				gpu.Arg(mB), gpu.Arg(nB), gpu.Arg(kB)); err != nil {
				t.Fatalf("%s launch: %v", name, err)
			}
			if err := q.Sync(); err != nil {
				t.Fatalf("%s sync: %v", name, err)
			}
			got := make([]int32, M*k)
			if err := gpu.Download(idxB, got); err != nil {
				t.Fatalf("%s download: %v", name, err)
			}
			rowName := [...]string{"clustered", "tied", "random"}
			for m := range M {
				for i := range k {
					if got[m*k+i] != want[m*k+i] {
						t.Fatalf("%s N=%d (N%%4=%d) k=%d row %q rank %d: index %d, want %d",
							name, N, N%4, k, rowName[m], i, got[m*k+i], want[m*k+i])
					}
				}
			}
		}
		for _, b := range []gpu.Buffer{idxB, valB, kB} {
			dev.ReleaseBuf(b)
		}
	}
}

// tkBlockThreads is TKBLOCK in gemv_w8a8.cu — the block width every top-k kernel
// launches with, and therefore the stride that decides which indices a thread sees.
// The clustered fixture above is built from it.
const tkBlockThreads = 256

// cpuTopK is the reference selection: score descending, lower index wins ties. Written
// out rather than reusing ann.topHits so the gate does not depend on the thing it is
// checking the device against.
func cpuTopK(scores []float32, M, N, k int) []int32 {
	out := make([]int32, M*k)
	for m := range M {
		row := scores[m*N : (m+1)*N]
		ord := make([]int32, N)
		for i := range ord {
			ord[i] = int32(i)
		}
		sort.SliceStable(ord, func(a, b int) bool {
			ia, ib := ord[a], ord[b]
			if row[ia] != row[ib] {
				return row[ia] > row[ib]
			}
			return ia < ib
		})
		copy(out[m*k:(m+1)*k], ord[:k])
	}
	return out
}

// TestCUDATopK_splitMatchesOneBlock gates the split/merge pair against the one-block
// kernel it replaces, launching both directly so the comparison does not depend on
// whichever route topkSplitParts happens to choose.
//
// The split is exact for a structural reason — the global top k is contained in the
// union of the per-chunk top k, since a candidate excluded from its own chunk's top k
// has k better candidates in that chunk alone — so equality is the right bar, not a
// tolerance. What can still break is the plumbing: chunk boundaries that skip or double
// an element, a partials stride that overlaps between queries, and blocks that fall
// entirely past the end of the row.
//
// PARTS IS SWEPT PAST THE USEFUL RANGE ON PURPOSE. parts=1000 over N=60000 gives every
// block a 64-element chunk and leaves most of them with nothing at all, which is the
// case where a block must still reach tkEmit's barriers and emit sentinels rather than
// return early. topkSplitParts would never choose it; that is exactly why the gate has
// to.
func TestCUDATopK_splitMatchesOneBlock(t *testing.T) {
	dev, err := gpu.CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	t.Cleanup(dev.ReleaseObjects)
	lib, err := dev.CompileLibrary(w8a8PTX)
	if err != nil {
		t.Fatalf("CompileLibrary: %v", err)
	}
	q := dev.NewCommandQueue()
	rng := rand.New(rand.NewSource(31))

	for _, N := range []int{60_000, 60_001, 60_003, 4_099} {
		for _, M := range []int{1, 3} {
			scores := make([]float32, M*N)
			for i := range scores {
				// Tie-dense: the tie-break has to survive being applied twice, once
				// per chunk and once across chunks.
				scores[i] = float32(rng.Intn(64))
			}
			sb := gpu.NewBufferOf(dev, scores)
			mB := gpu.NewBufferOf(dev, []uint32{uint32(M)})
			nB := gpu.NewBufferOf(dev, []uint32{uint32(N)})
			for wi, w := range topkWidths {
				for _, k := range []int{1, w} {
					want := cpuTopK(scores, M, N, k)
					kB := gpu.NewBufferOf(dev, []uint32{uint32(k)})
					ib := gpu.NewBufferLenOf[int32](dev, M*k)
					vb := dev.NewBufferLen(M * k)
					for _, parts := range []int{1, 2, 3, 7, 48, 1000} {
						split, err := dev.NewComputePipeline(lib, topkKernelName("split", w))
						if err != nil {
							t.Fatal(err)
						}
						merge, err := dev.NewComputePipeline(lib, topkKernelName("merge", w))
						if err != nil {
							t.Fatal(err)
						}
						pi := gpu.NewBufferLenOf[int32](dev, M*parts*k)
						pv := dev.NewBufferLen(M * parts * k)
						pB := gpu.NewBufferOf(dev, []uint32{uint32(parts * k)})
						if err := q.Launch(split, gpu.LaunchConfig{
							GridX: uint32(parts), GridY: uint32(M), GridZ: 1,
							BlockX: topkBlockThreads, BlockY: 1, BlockZ: 1,
						}, gpu.Arg(sb), gpu.Arg(pi), gpu.Arg(pv),
							gpu.Arg(mB), gpu.Arg(nB), gpu.Arg(kB)); err != nil {
							t.Fatalf("split N=%d M=%d w=%d k=%d parts=%d: %v", N, M, w, k, parts, err)
						}
						if err := q.Launch(merge, gpu.LaunchConfig{
							GridX: uint32(M), GridY: 1, GridZ: 1,
							BlockX: topkBlockThreads, BlockY: 1, BlockZ: 1,
						}, gpu.Arg(pi), gpu.Arg(pv), gpu.Arg(ib), gpu.Arg(vb),
							gpu.Arg(mB), gpu.Arg(pB), gpu.Arg(kB)); err != nil {
							t.Fatalf("merge N=%d M=%d w=%d k=%d parts=%d: %v", N, M, w, k, parts, err)
						}
						if err := q.Sync(); err != nil {
							t.Fatalf("sync N=%d parts=%d: %v", N, parts, err)
						}
						got := make([]int32, M*k)
						if err := gpu.Download(ib, got); err != nil {
							t.Fatal(err)
						}
						for i := range want {
							if got[i] != want[i] {
								t.Fatalf("split N=%d (N%%4=%d) M=%d width=%d k=%d parts=%d: hit %d index %d, want %d",
									N, N%4, M, w, k, parts, i, got[i], want[i])
							}
						}
						for _, b := range []gpu.Buffer{pi, pv, pB} {
							dev.ReleaseBuf(b)
						}
					}
					for _, b := range []gpu.Buffer{ib, vb, kB} {
						dev.ReleaseBuf(b)
					}
				}
				_ = wi
			}
			for _, b := range []gpu.Buffer{sb, mB, nB} {
				dev.ReleaseBuf(b)
			}
		}
	}
	t.Log("split+merge ≡ CPU selection over 4 N (incl. N%4≠0 and N<one chunk), 4 widths, 6 part counts incl. 1000")
}

// TestTopKSplitParts pins the split plan's two invariants without a GPU: never split
// when one block per query already fills the device, and never leave a block with less
// than a useful chunk of the row.
func TestTopKSplitParts(t *testing.T) {
	const sm = 40
	for _, M := range []int{sm, sm + 1, 128, 4096} {
		if p := topkSplitParts(M, 1<<20, sm); p != 1 {
			t.Errorf("topkSplitParts(M=%d) = %d, want 1 — M >= sm already fills the device", M, p)
		}
	}
	for _, tc := range []struct{ M, N int }{{1, 200_000}, {2, 200_000}, {8, 500_000}, {32, 500_000}} {
		p := topkSplitParts(tc.M, tc.N, sm)
		if p < 1 {
			t.Fatalf("topkSplitParts(%d, %d) = %d", tc.M, tc.N, p)
		}
		if chunk := tc.N / p; chunk < topkSplitChunk {
			t.Errorf("topkSplitParts(M=%d, N=%d) = %d leaves %d elements per block, below topkSplitChunk %d",
				tc.M, tc.N, p, chunk, topkSplitChunk)
		}
		if blocks := tc.M * p; blocks > topkSplitMaxBlocks {
			t.Errorf("topkSplitParts(M=%d, N=%d) = %d gives %d blocks, above topkSplitMaxBlocks %d",
				tc.M, tc.N, p, blocks, topkSplitMaxBlocks)
		}
	}
	// A tiny corpus must not be split into slivers.
	if p := topkSplitParts(1, 100, sm); p != 1 {
		t.Errorf("topkSplitParts(M=1, N=100) = %d, want 1", p)
	}
	// An unknown SM count must still produce a legal plan.
	if p := topkSplitParts(1, 200_000, 0); p < 1 {
		t.Errorf("topkSplitParts with sm=0 returned %d", p)
	}
}

// TestCUDATopK_scratchReuseIsInert gates the per-index scratch TopKBatch reuses for
// small batches. Reuse trades a fresh allocation for the risk of reading something the
// previous call left behind, and the shapes vary between calls: M and k change the
// score matrix, the hit arrays and the split partials independently.
//
// The order below is deliberate. Each call is preceded by a WIDER or DEEPER one, so
// every buffer is larger than the call needs and any stale tail is live memory rather
// than a fresh zero page — which is the only arrangement where reading past the current
// call's extent produces a plausible wrong answer instead of an obvious one.
func TestCUDATopK_scratchReuseIsInert(t *testing.T) {
	if !ann.HasBackend() {
		t.Skip("no CUDA backend registered (no GPU?)")
	}
	rng := rand.New(rand.NewSource(77))
	const n, dim = 60_000, 32
	vecs := randUnit(rng, n, dim)
	cpu := ann.NewFlatI8(vecs)
	x := newTestIndex(t, vecs)
	pool := randUnit(rng, 8, dim)

	for _, step := range []struct{ M, k int }{
		{8, 64}, {1, 8}, // wide+deep, then the narrowest and shallowest
		{4, 32}, {2, 4},
		{8, 1}, {1, 64}, // deep after shallow, so k grows back
		{1, 8}, {8, 8},
	} {
		queries := pool[:step.M]
		hits, err := x.TopKBatch(queries, step.k)
		if err != nil {
			t.Fatalf("M=%d k=%d: %v", step.M, step.k, err)
		}
		for m, q := range queries {
			want := cpu.Query(q, step.k)
			if len(hits[m]) != len(want) {
				t.Fatalf("M=%d k=%d q%d: %d hits, want %d", step.M, step.k, m, len(hits[m]), len(want))
			}
			for i := range want {
				if hits[m][i].Index != want[i].Index || hits[m][i].Score != want[i].Score {
					t.Fatalf("M=%d k=%d q%d rank %d: {%d, %v} != CPU {%d, %v} — stale scratch?",
						step.M, step.k, m, i, hits[m][i].Index, hits[m][i].Score, want[i].Index, want[i].Score)
				}
			}
		}
	}
	t.Log("8 TopKBatch calls with varying (M, k) over reused scratch, each ≡ CPU exactly")
}

// TestCUDAQuery_deviceSelectPathAndItsGuards covers the branch FlatI8.Query grew to
// route a single unfiltered query to the device's own top-k instead of copying all n
// scores back and scanning them here.
//
// The guards are the interesting part, and they are NOT equally load-bearing — which
// mutation testing is how we know, rather than assuming three conditions in one `if`
// carry equal weight:
//
//   - The keep filter guard IS load-bearing. Removing it fails this test: the device
//     selects before any filtering, so it hands back the global top k and the filter
//     then drops most of them, silently returning fewer hits than the caller asked for.
//   - The k guards (k > 0, k < n) are NOT observable through this backend. Removing
//     either leaves the test green, because cudaI8Index.TopKBatch declines exactly the
//     same k values and Query falls through anyway. They stay because ann must not
//     depend on a backend's decline policy for correctness — k >= n is topHits's
//     full-sort contract, which no device kernel here implements — but this test cannot
//     claim to prove them, and saying otherwise would make a redundant condition look
//     verified.
//
// Each case is checked against the CPU index computing the same thing.
func TestCUDAQuery_deviceSelectPathAndItsGuards(t *testing.T) {
	if !ann.HasBackend() {
		t.Skip("no CUDA backend registered (no GPU?)")
	}
	rng := rand.New(rand.NewSource(101))
	// n above topkMinN so the device path is actually reachable — below it TopKBatch
	// declines and this test would pass by never exercising the branch.
	const n, dim = 60_000, 32
	vecs := randUnit(rng, n, dim)
	cpu := ann.NewFlatI8(vecs)
	gpuIdx := ann.NewFlatI8(vecs)
	if err := gpuIdx.EnableGPU(); err != nil {
		t.Fatalf("EnableGPU: %v", err)
	}
	defer gpuIdx.Close()

	same := func(t *testing.T, what string, got, want []ann.Hit) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s: %d hits, want %d", what, len(got), len(want))
		}
		for i := range want {
			if got[i].Index != want[i].Index || got[i].Score != want[i].Score {
				t.Fatalf("%s rank %d: {%d, %v} != CPU {%d, %v}",
					what, i, got[i].Index, got[i].Score, want[i].Index, want[i].Score)
			}
		}
	}

	for qi, q := range randUnit(rng, 20, dim) {
		// The path itself.
		same(t, fmt.Sprintf("Query q%d", qi), gpuIdx.Query(q, 10), cpu.Query(q, 10))

		// Guard 1: a filter must fall through to score-then-select-here. Keeping only
		// even indices halves the corpus, so a device selection of the global top 10
		// would come back with roughly half the hits a caller asked for.
		keep := func(id int) bool { return id%2 == 0 }
		same(t, fmt.Sprintf("QueryFilter q%d", qi),
			gpuIdx.QueryFilter(q, 10, keep), cpu.QueryFilter(q, 10, keep))

		// Guard 2 and 3: k outside (0, n) is topHits's full-sort contract.
		same(t, fmt.Sprintf("Query k=n q%d", qi), gpuIdx.Query(q, n), cpu.Query(q, n))
		same(t, fmt.Sprintf("Query k=0 q%d", qi), gpuIdx.Query(q, 0), cpu.Query(q, 0))
	}
	t.Log("device-select Query ≡ CPU over 20 queries; filter, k=n and k=0 all fall through correctly")
}

// TestBatchGridX pins the row-grid narrowing. The kernel strides over rows, so ANY
// grid width is correct — which is exactly why this needs its own test: a wrong value
// costs throughput and nothing else, and every parity gate in this file stays green.
//
// Two directions to get wrong. Too wide gives back the staging saving the stride loop
// exists for; too narrow starves the device, and at small N the full grid can already
// be smaller than the machine, so dividing it further is pure loss.
func TestBatchGridX(t *testing.T) {
	const sm = 40
	// A large grid gets divided.
	if got := batchGridX(3125, 1, sm); got != 3125/batchRowsPerBlock {
		t.Errorf("batchGridX(3125, 1) = %d, want %d", got, 3125/batchRowsPerBlock)
	}
	// The floor holds across planes: with more query-tile planes, fewer blocks per
	// plane are needed to reach the same total.
	for _, planes := range []int{1, 4, 16} {
		got := int(batchGridX(8, planes, sm))
		if total := got * planes; total < batchMinBlocks*sm && got != 8 {
			t.Errorf("batchGridX(8, planes=%d) = %d gives %d blocks, below %d and not the full grid",
				planes, got, total, batchMinBlocks*sm)
		}
	}
	// Never wider than the grid it was given, and never zero.
	for _, full := range []int{1, 2, 7, 8, 100, 100_000} {
		for _, planes := range []int{1, 16} {
			got := int(batchGridX(full, planes, sm))
			if got < 1 || got > full {
				t.Errorf("batchGridX(%d, %d) = %d, out of [1, %d]", full, planes, got, full)
			}
		}
	}
	// An unknown SM count must still produce a legal grid.
	if got := batchGridX(3125, 1, 0); got < 1 {
		t.Errorf("batchGridX with sm=0 returned %d", got)
	}
}

// TestTopKPipelineWidths ties the three things that must agree — the kernel names, the
// widths the routing uses, and the widths those kernels were actually compiled with —
// and checks that topkPlan picks the narrowest kernel able to hold k.
//
// Routing k to a kernel NARROWER than k is the failure this exists for, and it is
// invisible to a parity test on ordinary data: a too-narrow kernel still returns a
// full, sorted, plausible list, missing only entries that happened to share a thread's
// stride. Mutation-checked — with the earlier hand-written switch, routing k≤16 to the
// 8-wide kernel passed every parity gate in this file.
//
// It needs no GPU, which is the point: on a machine without CUDA every device test here
// skips, and a skipping test is a passing test.
func TestTopKPipelineWidths(t *testing.T) {
	src, err := os.ReadFile("gemv_w8a8.cu")
	if err != nil {
		t.Fatalf("read kernel source: %v", err)
	}
	for i, w := range topkWidths {
		// Every kernel the routing can reach must exist in the source it is compiled
		// from. The names are derived from the widths, so this is what ties the two.
		for _, kind := range []string{"rows", "split", "merge"} {
			if decl := fmt.Sprintf("TOPK_%s_ENTRY(%s,", strings.ToUpper(map[string]string{
				"rows": "reg", "split": "split", "merge": "merge",
			}[kind]), topkKernelName(kind, w)); !bytes.Contains(src, []byte(decl)) {
				t.Errorf("gemv_w8a8.cu declares no %q", decl)
			}
		}
		if i > 0 && topkWidths[i-1] >= w {
			t.Errorf("topkWidths is not ascending at %d: %v", i, topkWidths)
		}
	}
	// The .cu's #defines must be the numbers the Go side thinks they are.
	for _, d := range []string{"#define TKREG_SMALL 8", "#define TKREG_MID 16",
		"#define TKREG_BIG 32", "#define TKREG_HUGE 64"} {
		if !bytes.Contains(src, []byte(d)) {
			t.Errorf("gemv_w8a8.cu has no %q", d)
		}
	}
	if !strings.Contains(fmt.Sprint(topkWidths), "[8 16 32 64]") {
		t.Errorf("topkWidths %v no longer matches the four TKREG defines above", topkWidths)
	}

	// The routing invariant: the narrowest kernel that can hold k, and -1 only above
	// the widest.
	widest := topkWidths[len(topkWidths)-1]
	for k := 1; k <= widest+8; k++ {
		i := topkPlan(k)
		if k > widest {
			if i != -1 {
				t.Errorf("topkPlan(%d) = %d, want -1 (k-pass) — widest register kernel holds %d", k, i, widest)
			}
			continue
		}
		if i < 0 {
			t.Errorf("topkPlan(%d) = -1, but the %d-wide kernel can hold it", k, widest)
			continue
		}
		if topkWidths[i] < k {
			t.Errorf("topkPlan(%d) picked the %d-wide kernel, which CANNOT hold %d candidates", k, topkWidths[i], k)
		}
		if i > 0 && topkWidths[i-1] >= k {
			t.Errorf("topkPlan(%d) picked the %d-wide kernel when the %d-wide one suffices", k, topkWidths[i], topkWidths[i-1])
		}
	}
}

// TestBatchPlan gates the pairing of kernel to geometry. See batchPlan's comment for
// why this cannot be left to the parity tests: the under-launch direction drops the
// tail of the corpus and still returns a complete, plausible top-k from the rows it
// did reach, and the over-launch direction is invisible to correctness entirely.
func TestBatchPlan(t *testing.T) {
	for _, tc := range []struct {
		M            int
		wantWide     bool
		lanes, qtile int
	}{
		{1, false, 8, 8},
		{8, false, 8, 8},
		{9, true, 4, 16},
		{256, true, 4, 16},
	} {
		wide, lanes, qtile := batchPlan(tc.M)
		if wide != tc.wantWide || lanes != tc.lanes || qtile != tc.qtile {
			t.Errorf("batchPlan(%d) = (wide=%v, lanes=%d, qtile=%d), want (%v, %d, %d)",
				tc.M, wide, lanes, qtile, tc.wantWide, tc.lanes, tc.qtile)
		}
	}
	// The geometry must belong to the kernel that was picked, not merely be one of
	// the two valid pairs: a swapped pairing satisfies "lanes is 4 or 8" and would
	// under-launch by 2x.
	if _, lanes, qtile := batchPlan(batchSmallMaxM + 1); lanes*qtile != batchLanesWide*batchQTileWide {
		t.Errorf("wide plan (%d, %d) is not the wide kernel's pair", lanes, qtile)
	}
}

// TestBatchKernelConstantsMatchSource ties the host-side launch geometry to the
// kernel source it must agree with. batchLanes*/batchQTile* are duplicated from
// gemv_w8a8.cu's template arguments because PTX carries no way to ask a kernel what
// it was compiled with, and a silent divergence there mis-sizes GridY and the shared
// query tile.
//
// This is the one gate in this file that needs no GPU, which is the point: on a
// machine without CUDA every parity test above skips, and a skipping test is a
// passing test.
func TestBatchKernelConstantsMatchSource(t *testing.T) {
	src, err := os.ReadFile("gemv_w8a8.cu")
	if err != nil {
		t.Fatalf("read kernel source: %v", err)
	}
	// The single-query kernel's group width, which sets its launch multiplier.
	if decl := fmt.Sprintf("#define GEMV_LANES %d", gemvLanes); !bytes.Contains(src, []byte(decl)) {
		t.Errorf("gemv_w8a8.cu has no %q — Score dispatches n*%d threads on that assumption, "+
			"and a kernel compiled for fewer would leave the tail of the corpus unscored",
			decl, gemvLanes)
	}
	for _, want := range []struct {
		call         string
		lanes, qtile int
	}{
		{"batch_body<8, 8>", batchLanesSmall, batchQTileSmall},
		{"batch_body<4, 16>", batchLanesWide, batchQTileWide},
	} {
		if !bytes.Contains(src, []byte(want.call)) {
			t.Errorf("gemv_w8a8.cu has no %q — the Go constants say (%d, %d)",
				want.call, want.lanes, want.qtile)
			continue
		}
		if got := fmt.Sprintf("batch_body<%d, %d>", want.lanes, want.qtile); got != want.call {
			t.Errorf("Go constants render %q, source has %q", got, want.call)
		}
	}
}
