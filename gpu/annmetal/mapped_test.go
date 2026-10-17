//go:build darwin

package annmetal

import (
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"testing"

	"github.com/townsendmerino/aikit/ann"
)

// Item 10 (docs/internal/task-memory-campaign.md): a mapped FlatI8 on annmetal reads its codes from the mapping.

// mappedFixture writes an n × dim FlatI8 to a file and returns two mapped loads of it: one to enable the GPU on,
// one CPU-only reference.
func mappedFixture(t *testing.T, n, dim int) (gpuIdx, cpuIdx *ann.FlatI8, queries [][]float32) {
	t.Helper()
	if registered == nil {
		t.Skip("no Metal device")
	}
	rng := rand.New(rand.NewSource(int64(n + dim)))
	blob, err := ann.NewFlatI8(randUnit(rng, n, dim)).MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "index.fi8")
	if err := os.WriteFile(p, blob, 0o644); err != nil {
		t.Fatal(err)
	}
	if gpuIdx, err = ann.LoadFlatI8Mmap(p); err != nil {
		t.Fatal(err)
	}
	if cpuIdx, err = ann.LoadFlatI8Mmap(p); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = gpuIdx.Close(); _ = cpuIdx.Close() })
	return gpuIdx, cpuIdx, randUnit(rng, 48, dim)
}

// recordingBackend is the real backend, recording what EnableGPU hands NewI8IndexMapped — the views into the
// mapping, which the test needs and FlatI8 does not export — and whether the mapped path succeeded.
type recordingBackend struct {
	*metalBackend
	mapping   []byte
	bq        []int8
	scales    []float32
	mappedErr error
	calls     int
}

func (r *recordingBackend) NewI8IndexMapped(m []byte, bq []int8, s []float32, n, dim int) (ann.I8Index, error) {
	r.mapping, r.bq, r.scales, r.calls = m, bq, s, r.calls+1
	idx, err := r.metalBackend.NewI8IndexMapped(m, bq, s, n, dim)
	r.mappedErr = err
	return idx, err
}

// enableRecorded runs g.EnableGPU through a recordingBackend and requires that it took the mapped path.
func enableRecorded(t *testing.T, g *ann.FlatI8) *recordingBackend {
	t.Helper()
	rec := &recordingBackend{metalBackend: registered}
	ann.RegisterBackend(rec)
	defer ann.RegisterBackend(registered)
	if err := g.EnableGPU(); err != nil {
		t.Fatal(err)
	}
	if rec.calls != 1 || rec.mappedErr != nil {
		t.Fatalf("EnableGPU on a LoadFlatI8Mmap index did not take the mapped path (calls %d, err %v)", rec.calls, rec.mappedErr)
	}
	return rec
}

// sameRanking is the existing parity bar (TestMetalGEMV_parityWithCPU): identical indices in identical order, scores
// within 1e-3 (the GPU's float rescale is not bit-identical to the CPU's).
func sameRanking(t *testing.T, what string, got, want []ann.Hit) float64 {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d hits, want %d", what, len(got), len(want))
	}
	worst := 0.0
	for i := range got {
		if got[i].Index != want[i].Index {
			t.Fatalf("%s rank %d: index %d, CPU %d", what, i, got[i].Index, want[i].Index)
		}
		worst = max(worst, math.Abs(got[i].Score-want[i].Score))
	}
	if worst > 1e-3 {
		t.Fatalf("%s: worst score Δ %.3e > 1e-3", what, worst)
	}
	return worst
}

// Gate (a): on a mapped index, Query (GEMV), QueryBatch at k = 10 (device top-k, N ≥ 100k) and k = 50 (ScoreBatch)
// rank exactly as the CPU does (a2), and the mapped device index's results equal the copied one's bit for bit (a1).
func TestMappedIndex_matchesCPUAndCopy(t *testing.T) {
	const n, dim = 100_000, 256
	g, cpu, qs := mappedFixture(t, n, dim)
	rec := enableRecorded(t, g)
	// The copied device index, over the same codes, through the plain NewI8Index.
	copied, err := registered.NewI8Index(rec.bq, rec.scales, n, dim)
	if err != nil {
		t.Fatal(err)
	}
	defer copied.Close()
	mapped, err := registered.NewI8IndexMapped(rec.mapping, rec.bq, rec.scales, n, dim)
	if err != nil {
		t.Fatalf("NewI8IndexMapped declined a LoadFlatI8Mmap index: %v", err)
	}
	defer mapped.Close()
	worst := 0.0
	for i, q := range qs {
		worst = max(worst, sameRanking(t, "Query", g.Query(q, 10), cpu.Query(q, 10)))
		a, b := make([]float32, n), make([]float32, n)
		if err := mapped.Score(q, a); err != nil {
			t.Fatal(err)
		}
		if err := copied.Score(q, b); err != nil {
			t.Fatal(err)
		}
		for j := range a {
			if math.Float32bits(a[j]) != math.Float32bits(b[j]) {
				t.Fatalf("query %d row %d: mapped %v, copied %v", i, j, a[j], b[j])
			}
		}
	}
	for _, k := range []int{10, 50} {
		got, want := g.QueryBatch(qs, k), cpu.QueryBatch(qs, k)
		for i := range got {
			worst = max(worst, sameRanking(t, "QueryBatch", got[i], want[i]))
		}
	}
	mb := make([]float32, len(qs)*n)
	cb := make([]float32, len(qs)*n)
	if err := mapped.(ann.I8BatchIndex).ScoreBatch(qs, mb); err != nil {
		t.Fatal(err)
	}
	if err := copied.(ann.I8BatchIndex).ScoreBatch(qs, cb); err != nil {
		t.Fatal(err)
	}
	for j := range mb {
		if math.Float32bits(mb[j]) != math.Float32bits(cb[j]) {
			t.Fatalf("ScoreBatch element %d: mapped %v, copied %v", j, mb[j], cb[j])
		}
	}
	t.Logf("mapped GPU ≡ CPU ranking (Query, QueryBatch k=10/50; worst score Δ %.2e); mapped ≡ copied bit for bit (Score, ScoreBatch)", worst)
}

// footprintBytes is this process's physical footprint as macOS's footprint(1) reports it: dirty and compressed
// memory, which a copy into a shared Metal buffer adds and a no-copy buffer over clean file pages does not.
func footprintBytes(t *testing.T) int64 {
	t.Helper()
	out, err := exec.Command("footprint", "-f", "bytes", "--noCategories", strconv.Itoa(os.Getpid())).Output()
	if err != nil {
		t.Skipf("footprint(1): %v", err)
	}
	m := regexp.MustCompile(`Footprint: ([0-9.]+) (B|KB|MB|GB)`).FindSubmatch(out)
	if m == nil {
		t.Fatalf("footprint(1) output has no Footprint line:\n%s", out)
	}
	v, err := strconv.ParseFloat(string(m[1]), 64)
	if err != nil {
		t.Fatalf("footprint(1) value %q: %v", m[1], err)
	}
	return int64(v * map[string]float64{"B": 1, "KB": 1 << 10, "MB": 1 << 20, "GB": 1 << 30}[string(m[2])])
}

// Gate (b): EnableGPU on a mapped index adds about the scratch only — the codes are not copied. Bar (registered):
// ≤ n × 8 B (scales + output) + 1 MiB after EnableGPU, measured as the process's physical footprint (and, beside
// it, after a first query, when the GPU has read every code; that delta also holds the query's own host
// allocations, Go heap, so it is reported, not barred). The registration named Device.CurrentAllocatedSize as the instrument, but Metal counts a no-copy buffer
// there at its full length although it allocates nothing (measured: 25.6 MiB mapped vs 25.2 copied), so it cannot
// tell the two apart; it is reported beside, for the record. The copied path is reported for comparison.
func TestMappedIndex_deviceBytes(t *testing.T) {
	const n, dim = 100_000, 256
	g, _, qs := mappedFixture(t, n, dim)
	dev := registered.dev
	runtime.GC()
	fp0, dv0 := footprintBytes(t), int64(dev.CurrentAllocatedSize())
	rec := enableRecorded(t, g)
	fp1, dv1 := footprintBytes(t), int64(dev.CurrentAllocatedSize())
	g.Query(qs[0], 10) // the GPU reads every code: the mapping's pages are now resident, as clean file pages
	fpQ := footprintBytes(t)
	fpC0, dvC0 := footprintBytes(t), int64(dev.CurrentAllocatedSize())
	copied, err := registered.NewI8Index(rec.bq, rec.scales, n, dim)
	if err != nil {
		t.Fatal(err)
	}
	fp2, dv2 := footprintBytes(t), int64(dev.CurrentAllocatedSize())
	_ = copied.Close()
	mib := func(b int64) float64 { return float64(b) / (1 << 20) }
	bar := int64(n*8 + 1<<20)
	t.Logf("n=%d dim=%d (codes %.2f MiB). Footprint added: mapped EnableGPU %.2f MiB (bar %.2f; after a first query %.2f), copied index %.2f MiB",
		n, dim, mib(int64(n*dim)), mib(fp1-fp0), mib(bar), mib(fpQ-fp0), mib(fp2-fpC0))
	t.Logf("Device.CurrentAllocatedSize added (counts a no-copy wrap at full length): mapped %.2f MiB, copied %.2f MiB", mib(dv1-dv0), mib(dv2-dvC0))
	if fp1-fp0 > bar {
		t.Fatalf("mapped EnableGPU added %.2f MiB of footprint, bar %.2f MiB", mib(fp1-fp0), mib(bar))
	}
	if fp2-fpC0 < int64(n*dim)*9/10 {
		t.Fatalf("the copied index added only %.2f MiB of footprint: the instrument cannot see the copy it should", mib(fp2-fpC0))
	}
}

// Gate (c): repeated ScoreBatch and TopKBatch calls at a fixed M allocate no new device memory after the first.
func TestBatchScratch_reusedAcrossCalls(t *testing.T) {
	const n, dim = 100_000, 64
	g, _, qs := mappedFixture(t, n, dim)
	rec := enableRecorded(t, g)
	idx, err := registered.NewI8IndexMapped(rec.mapping, rec.bq, rec.scales, n, dim)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()
	dst := make([]float32, len(qs)*n)
	b, tk := idx.(ann.I8BatchIndex), idx.(ann.I8TopKIndex)
	if err := b.ScoreBatch(qs, dst); err != nil {
		t.Fatal(err)
	}
	if _, err := tk.TopKBatch(qs, 10); err != nil {
		t.Fatal(err)
	}
	dev := registered.dev
	settled := dev.CurrentAllocatedSize()
	for range 20 {
		if err := b.ScoreBatch(qs, dst); err != nil {
			t.Fatal(err)
		}
		if _, err := tk.TopKBatch(qs, 10); err != nil {
			t.Fatal(err)
		}
		if _, err := tk.TopKBatch(qs[:7], 5); err != nil { // a smaller batch fits the same scratch
			t.Fatal(err)
		}
	}
	if after := dev.CurrentAllocatedSize(); after != settled {
		t.Fatalf("device bytes %d → %d across 60 batch calls at sizes the scratch already holds", settled, after)
	}
}
