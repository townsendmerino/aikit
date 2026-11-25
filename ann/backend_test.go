package ann

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// fakeBatchIndex is an I8BatchIndex that scores on the CPU, so the device
// batched path can be exercised without a device.
//
// It exists because that path is otherwise UNREACHABLE in the default build —
// f.gpu is nil unless a gpu build registers a backend — which is why §4.7's
// per-call M×N allocation survived this long: no test and no benchmark ever ran
// the line. calls counts dispatches so a test can prove the path was taken
// rather than assume it.
type fakeBatchIndex struct {
	f     *FlatI8
	calls int
}

func (b *fakeBatchIndex) Score(q []float32, dst []float32) error {
	refW8A8(q, b.f.bq, b.f.scales, dst, 1, b.f.dim, b.f.n)
	return nil
}
func (b *fakeBatchIndex) Close() error { return nil }
func (b *fakeBatchIndex) ScoreBatch(queries [][]float32, dst []float32) error {
	b.calls++
	for m, q := range queries {
		refW8A8(q, b.f.bq, b.f.scales, dst[m*b.f.n:(m+1)*b.f.n], 1, b.f.dim, b.f.n)
	}
	return nil
}

// TestQueryBatch_deviceBatchedPathMatchesPerQuery gates §4.7's change: pooling
// the M×N score matrix must not alter a single hit, and the batched path must
// actually be the one running.
func TestQueryBatch_deviceBatchedPathMatchesPerQuery(t *testing.T) {
	const n, d, k = 2000, 64, 10
	vecs := unitVecs(n, d, 47)
	f := NewFlatI8(vecs)
	queries := unitVecs(24, d, 470)

	want := make([][]Hit, len(queries))
	for i, q := range queries {
		want[i] = f.Query(q, k)
	}

	fb := &fakeBatchIndex{f: f}
	f.gpu = fb
	defer func() { f.gpu = nil }()

	// Twice, so the second call reuses a pooled buffer the first one filled.
	for round := range 2 {
		got := f.QueryBatch(queries, k)
		if fb.calls != round+1 {
			t.Fatalf("round %d: ScoreBatch called %d times; the batched path did not run", round, fb.calls)
		}
		for i := range want {
			if len(got[i]) != len(want[i]) {
				t.Fatalf("round %d query %d: %d hits vs %d", round, i, len(got[i]), len(want[i]))
			}
			for j := range want[i] {
				if got[i][j] != want[i][j] {
					t.Fatalf("round %d query %d hit %d: %+v vs per-query %+v", round, i, j, got[i][j], want[i][j])
				}
			}
		}
	}
}

// TestQueryBatch_pooledScratchIsInert poisons the pooled score matrix before a
// call. ScoreBatch assigns every element it is given, so stale contents must be
// invisible — and if that contract is ever broken, the symptom is a plausible
// wrong ranking rather than a crash.
func TestQueryBatch_pooledScratchIsInert(t *testing.T) {
	const n, d, k = 1500, 48, 10
	vecs := unitVecs(n, d, 48)
	f := NewFlatI8(vecs)
	queries := unitVecs(8, d, 480)
	fb := &fakeBatchIndex{f: f}
	f.gpu = fb
	defer func() { f.gpu = nil }()

	want := f.QueryBatch(queries, k)

	sc := batchScratchPool.Get().(*batchScratch)
	for i := range sc.dst {
		sc.dst[i] = float32(math.Inf(1))
	}
	batchScratchPool.Put(sc)

	got := f.QueryBatch(queries, k)
	for i := range want {
		for j := range want[i] {
			if got[i][j] != want[i][j] {
				t.Fatalf("query %d hit %d after a poisoned pool: %+v, want %+v", i, j, got[i][j], want[i][j])
			}
		}
	}
}

// BenchmarkQueryBatch_deviceBatched prices §4.7 through the fake backend.
//
// WHAT IT CAN AND CANNOT SHOW. The scoring here is the CPU kernel, not a device
// GEMM, so the TIME is not a device number and must not be read as one — a real
// backend's dispatch would dominate. What transfers is the allocation: the M×N
// host score matrix is the same size and lifetime whatever fills it, and that is
// the whole of the finding.
func BenchmarkQueryBatch_deviceBatched(b *testing.B) {
	const n, d, k = 50_000, 256, 10
	f := NewFlatI8(unitVecs(n, d, 407))
	fb := &fakeBatchIndex{f: f}
	f.gpu = fb
	defer func() { f.gpu = nil }()
	for _, m := range []int{8, 64, 256} {
		queries := unitVecs(m, d, int64(4070+m))
		b.Run(fmt.Sprintf("m%d", m), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				sinkBatchHits = f.QueryBatch(queries, k)
			}
		})
	}
}

var sinkBatchHits [][]Hit

// errFakeShard is the forced failure fakeShardIndex returns when fail is set —
// used to exercise queryBatchSharded's "either worker errors ⇒ fall through"
// contract.
var errFakeShard = errors.New("fake shard: forced failure")

// fakeShardIndex is an I8Index/I8BatchIndex/I8TopKIndex that scores on the CPU
// against its OWN bq/scales sub-slice — standing in for the smaller
// device-resident index EnableGPUShardSplit would build via
// backend.NewI8Index, so the shard-split path is testable without hardware.
// f is only used for its (field-independent) topHits method — see
// EnableGPUShardSplit's doc comment for why the shard is a sub-slice, not a
// separate FlatI8.
type fakeShardIndex struct {
	f          *FlatI8
	bq         []int8
	scales     []float32
	n, dim     int
	fail       bool // when true, every method errors
	topkCalls  int
	batchCalls int
}

func (s *fakeShardIndex) Score(q []float32, dst []float32) error {
	if s.fail {
		return errFakeShard
	}
	refW8A8(q, s.bq, s.scales, dst, 1, s.dim, s.n)
	return nil
}
func (s *fakeShardIndex) Close() error { return nil }
func (s *fakeShardIndex) ScoreBatch(queries [][]float32, dst []float32) error {
	if s.fail {
		return errFakeShard
	}
	s.batchCalls++
	for m, q := range queries {
		refW8A8(q, s.bq, s.scales, dst[m*s.n:(m+1)*s.n], 1, s.dim, s.n)
	}
	return nil
}
func (s *fakeShardIndex) TopKBatch(queries [][]float32, k int) ([][]Hit, error) {
	if s.fail {
		return nil, errFakeShard
	}
	s.topkCalls++
	out := make([][]Hit, len(queries))
	dst := make([]float32, s.n)
	for m, q := range queries {
		refW8A8(q, s.bq, s.scales, dst, 1, s.dim, s.n)
		out[m] = s.f.topHits(dst, k, nil)
	}
	return out, nil
}

// wireShard sets f's shard-split fields directly (mirroring how every test
// above sets f.gpu directly rather than going through EnableGPU) to a fake
// index over corpus rows [0, gpuShardRows) — the same split boundary
// EnableGPUShardSplit uses. topk selects whether the fake shard implements
// I8TopKIndex-preferred behavior (true) or is exercised via its ScoreBatch
// fallback only (false, by making TopKBatch always fail).
func wireShard(f *FlatI8, gpuShardRows int, topk bool) *fakeShardIndex {
	s := &fakeShardIndex{
		f: f, bq: f.bq[:gpuShardRows*f.dim], scales: f.scales[:gpuShardRows],
		n: gpuShardRows, dim: f.dim,
	}
	f.gpuShard = shardWrapper{s, topk}
	f.gpuShardRows = gpuShardRows
	return s
}

// shardWrapper hides fakeShardIndex.TopKBatch (via a distinct type, so the
// I8TopKIndex type assertion in scoreShardGPU fails) when topk is false,
// forcing the ScoreBatch fallback path — without needing a second fake type.
type shardWrapper struct {
	*fakeShardIndex
	topk bool
}

func (w shardWrapper) TopKBatch(queries [][]float32, k int) ([][]Hit, error) {
	if !w.topk {
		return nil, errFakeShard
	}
	return w.fakeShardIndex.TopKBatch(queries, k)
}

// TestQueryBatch_shardSplitMatchesSerial is the exactness gate: the CPU∥GPU
// shard-split result must be EXACTLY what serial per-query Query returns —
// same indices, same scores, same order — including under ties, the same
// adversarial shape ann/flat_shard_test.go's TestFlatQuery_shardedMatchesSerial
// uses for Flat's CPU-only sharding, since a wrong offset direction (GPU
// shard's indices are already global; the CPU shard's need +gpuShardRows)
// would silently corrupt rankings rather than crash.
func TestQueryBatch_shardSplitMatchesSerial(t *testing.T) {
	for _, tc := range []struct {
		name string
		n, d int
		tied bool
	}{
		{"random_4000x64", 4000, 64, false},
		{"random_9001x96", 9001, 96, false}, // n not evenly split by any share below
		{"all_identical_3000", 3000, 64, true},
	} {
		var vecs [][]float32
		if tc.tied {
			unit := make([]float32, tc.d)
			unit[0] = 1
			vecs = make([][]float32, tc.n)
			for i := range vecs {
				v := make([]float32, tc.d)
				copy(v, unit)
				vecs[i] = v
			}
		} else {
			vecs = unitVecs(tc.n, tc.d, int64(tc.n*97+tc.d))
		}
		f := NewFlatI8(vecs)
		queries := unitVecs(12, tc.d, int64(tc.n*13+tc.d))

		for _, share := range []float64{0.2, 0.5, 0.8} {
			for _, useTopK := range []bool{true, false} {
				gpuShardRows := int(share * float64(tc.n))
				if gpuShardRows < 1 || gpuShardRows > tc.n-1 {
					continue
				}
				wireShard(f, gpuShardRows, useTopK)

				for _, k := range []int{1, 5, 10} {
					want := make([][]Hit, len(queries))
					for i, q := range queries {
						want[i] = f.Query(q, k)
					}
					got := f.QueryBatch(queries, k)
					if !reflect.DeepEqual(got, want) {
						t.Fatalf("%s share=%.1f topk=%v k=%d: sharded QueryBatch differs from serial\n got  %v\nwant %v",
							tc.name, share, useTopK, k, got, want)
					}
				}
			}
		}
		f.gpuShard, f.gpuShardRows = nil, 0
	}
}

// TestQueryBatch_shardSplitFallsThrough confirms that when the device shard
// errors on every call, QueryBatch still returns the correct answer via its
// other tiers — the same "try, else fall through" contract as every other
// tier in this file, now covering the new one too.
func TestQueryBatch_shardSplitFallsThrough(t *testing.T) {
	const n, d, k = 3000, 64, 10
	f := NewFlatI8(unitVecs(n, d, 909))
	queries := unitVecs(6, d, 9090)

	want := make([][]Hit, len(queries))
	for i, q := range queries {
		want[i] = f.Query(q, k)
	}

	s := wireShard(f, n/2, true)
	s.fail = true
	defer func() { f.gpuShard, f.gpuShardRows = nil, 0 }()

	got := f.QueryBatch(queries, k)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("QueryBatch with a failing shard differs from serial\n got  %v\nwant %v", got, want)
	}
}

// TestQueryBatch_shardSplitScratchIsInert poisons the CPU-shard scratch pool
// before a call — the same hazard TestQueryBatch_pooledScratchIsInert and
// TestFlatI8_pooledScratchIsInert guard for their own pools, now for
// shardCPUScratchPool: scoreShardCPU's dst must be fully overwritten by
// MatmulBTW8A8Into, not merely reused.
func TestQueryBatch_shardSplitScratchIsInert(t *testing.T) {
	const n, d, k = 2500, 64, 10
	f := NewFlatI8(unitVecs(n, d, 55))
	queries := unitVecs(5, d, 5050)

	wireShard(f, n/2, true)
	defer func() { f.gpuShard, f.gpuShardRows = nil, 0 }()

	want := f.QueryBatch(queries, k)

	held := make([]*shardCPUScratch, 0, 8)
	for range 8 {
		sc := shardCPUScratchPool.Get().(*shardCPUScratch)
		need := len(queries) * (n - n/2)
		if cap(sc.dst) < need {
			sc.dst = make([]float32, need)
		}
		for i := range sc.dst[:need] {
			sc.dst[i] = float32(math.Inf(1))
		}
		held = append(held, sc)
	}
	for _, sc := range held {
		shardCPUScratchPool.Put(sc)
	}

	got := f.QueryBatch(queries, k)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("QueryBatch after a poisoned shard pool differs\n got  %v\nwant %v", got, want)
	}
}

// TestEnableGPUShardSplit_validation covers EnableGPUShardSplit's input
// guards. It does not cover the paged-index guard (errPagedGPU) — no existing
// test covers EnableGPU's identical guard either, and building the mmap-paged
// fixture (LoadFlatI8MmapPaged) just for this would be disproportionate to a
// guard this method shares verbatim with EnableGPU.
func TestEnableGPUShardSplit_validation(t *testing.T) {
	f := NewFlatI8(unitVecs(1000, 32, 3))

	if err := f.EnableGPUShardSplit(0.5); err != errNoBackend {
		t.Fatalf("no backend registered: err = %v, want errNoBackend", err)
	}

	backend = &fakeBackendForShardTest{}
	defer func() { backend = nil }()

	for _, share := range []float64{-0.1, 0, 1, 1.1} {
		if err := f.EnableGPUShardSplit(share); err != errShardRange {
			t.Errorf("gpuShare=%v: err = %v, want errShardRange", share, err)
		}
	}

	tiny := NewFlatI8(unitVecs(1, 32, 4))
	if err := tiny.EnableGPUShardSplit(0.5); err != errShardRange {
		t.Errorf("n=1: err = %v, want errShardRange", err)
	}

	if err := f.EnableGPUShardSplit(0.3); err != nil {
		t.Fatalf("valid split: unexpected error %v", err)
	}
	if f.gpuShardRows != 300 {
		t.Errorf("gpuShardRows = %d, want 300 (0.3 of 1000)", f.gpuShardRows)
	}
	if f.gpuShard == nil {
		t.Fatal("gpuShard not set after a successful EnableGPUShardSplit")
	}
	f.gpuShard, f.gpuShardRows = nil, 0
}

// fakeBackendForShardTest lets TestEnableGPUShardSplit_validation exercise the
// success path without a real device: it just wraps whatever bq/scales/n/dim
// it's given in a fakeShardIndex.
type fakeBackendForShardTest struct{}

func (fakeBackendForShardTest) NewI8Index(bq []int8, scales []float32, n, dim int) (I8Index, error) {
	return shardWrapper{&fakeShardIndex{bq: bq, scales: scales, n: n, dim: dim}, false}, nil
}
func (fakeBackendForShardTest) Name() string { return "fake-shard-test" }

// refW8A8 is the CPU W8A8 reference: dst[M, N] = q[M, K] · codesᵀ through a WeightMat over the codes, the kernel
// FlatI8's CPU scan runs.
func refW8A8(q []float32, codes []int8, scales, dst []float32, M, K, N int) {
	w := linalg.WrapInt8(codes, scales, N, K, true)
	w.MatmulBT(q, dst, M)
}
