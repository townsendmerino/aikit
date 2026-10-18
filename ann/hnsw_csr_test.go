package ann

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"unsafe"
)

// Item 7 (docs/internal/task-memory-campaign.md): the HNSW graph as CSR, blob v5.

// writeBlob writes b to a fresh file and returns its path.
func writeBlob(t *testing.T, b []byte) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "index.hnsw")
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// Gate (a): every campaign build, saved as v5 and loaded back by Load and by LoadHNSWMmap, answers every campaign
// query bit-identically to the pre-change tree (the item-4 query hash).
func TestHNSW_v5LoadsMatchBefore(t *testing.T) {
	if testing.Short() {
		t.Skip("builds 8 indexes of 5,000 vectors")
	}
	heap, mapped := sha256.New(), sha256.New()
	for _, blob := range campaignBuilds().blobs {
		l, err := Load(blob)
		if err != nil {
			t.Fatal(err)
		}
		campaignQueryHash(heap, l)
		m, err := LoadHNSWMmap(writeBlob(t, blob))
		if err != nil {
			t.Fatal(err)
		}
		campaignQueryHash(mapped, m)
		if err := m.Close(); err != nil {
			t.Fatal(err)
		}
	}
	for name, h := range map[string][]byte{"Load": heap.Sum(nil), "LoadHNSWMmap": mapped.Sum(nil)} {
		if got := hex.EncodeToString(h); got != campaignHNSWQueryHashBefore(t) {
			t.Errorf("%s of v5 blobs: query hash %s, before %s", name, got, campaignHNSWQueryHashBefore(t))
		}
	}
}

// Gate (a), old blobs: the v4 fixtures (written by the v4 encoder before this change) load under both loaders and
// answer exactly as the same build in memory does, and saving one again writes v5 that loads to the same answers.
func TestHNSW_v4BlobStillLoads(t *testing.T) {
	vecs := campaignUnitVecs(11, 400, 24)
	queries := campaignUnitVecs(12, 50, 24)
	for name, cfg := range map[string]Config{"f32": {Seed: 3}, "int8": {Seed: 3, Int8: true}} {
		path := "testdata/hnsw_v4_" + name + ".bin"
		blob, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if v := blob[4]; v != 4 {
			t.Fatalf("%s is format %d, want the v4 fixture", path, v)
		}
		want := BuildHNSW(vecs, cfg)
		heap, err := Load(blob)
		if err != nil {
			t.Fatalf("%s: Load: %v", name, err)
		}
		mapped, err := LoadHNSWMmap(path)
		if err != nil {
			t.Fatalf("%s: LoadHNSWMmap: %v", name, err)
		}
		v5, err := heap.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		again, err := Load(v5)
		if err != nil {
			t.Fatal(err)
		}
		for i, q := range queries {
			w := want.Query(q, 10)
			for which, h := range map[string]*HNSW{"Load": heap, "LoadHNSWMmap": mapped, "re-saved v5": again} {
				if got := h.Query(q, 10); !hitsEqual(got, w) {
					t.Fatalf("%s %s, query %d: %v, want %v", name, which, i, got, w)
				}
			}
		}
		_ = mapped.Close()
	}
}

func hitsEqual(a, b []Hit) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Gate (b): a mapped v5 load keeps the graph and vectors in the mapping: for n = 50,000, dim 64, int8, the heap it
// adds is ≤ 1 MiB. The same index saved as v4 (graph copied into the heap) is reported beside it.
func TestHNSWMmap_v5HeapAfterLoad(t *testing.T) {
	if testing.Short() || raceEnabled {
		t.Skip("builds a 50,000-vector index: minutes under -race, and a heap measurement is not what -race is for")
	}
	if !mmapIsReal {
		t.Skip("mmap.MapReadOnly heap-reads on this platform, so a mapped load has no heap to save")
	}
	h := BuildHNSW(campaignUnitVecs(21, 50000, 64), Config{Int8: true, EfConstruction: 40, Seed: 2})
	v5, err := h.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	// Signed: a load that adds nothing can read slightly negative, once the GC has reclaimed earlier garbage.
	added := func(path string) (int64, *HNSW) {
		runtime.GC()
		runtime.GC()
		var a, b runtime.MemStats
		runtime.ReadMemStats(&a)
		m, err := LoadHNSWMmap(path)
		if err != nil {
			t.Fatal(err)
		}
		runtime.GC()
		runtime.ReadMemStats(&b)
		return int64(b.HeapAlloc) - int64(a.HeapAlloc), m
	}
	got, m := added(writeBlob(t, v5))
	old, mOld := added(writeBlob(t, encodeV4ForTest(h)))
	t.Logf("heap added by LoadHNSWMmap, n=50,000 dim 64 int8: v5 %.3f MiB, v4 %.3f MiB (blob %.1f MiB)",
		float64(got)/(1<<20), float64(old)/(1<<20), float64(len(v5))/(1<<20))
	lo, hi := uintptr(unsafe.Pointer(&m.mmap[0])), uintptr(unsafe.Pointer(&m.mmap[0]))+uintptr(len(m.mmap))
	for name, p := range map[string]uintptr{"ids": uintptr(unsafe.Pointer(&m.csr.ids[0])), "scales": uintptr(unsafe.Pointer(&m.scales[0]))} {
		if p < lo || p >= hi {
			t.Errorf("v5 mapped load: %s is not in the mapping", name)
		}
	}
	if got > 1<<20 {
		t.Errorf("v5 mapped load added %d B of heap, bound 1 MiB", got)
	}
	_ = m.Close()
	_ = mOld.Close()
}

// addAfterLoadHash fingerprints Add after a load: an index of 1,000 vectors is saved and loaded with load, 200 more are
// added, and 60 queries (k = 10, and QueryEf at ef = 200) are hashed, for f32 and int8. It uses only API that predates
// v5, so the same function ran on the pre-change tree for the capture.
func addAfterLoadHash(tb testing.TB, load func([]byte) *HNSW) string {
	vecs := campaignUnitVecs(31, 1200, 32)
	hh := sha256.New()
	for _, cfg := range []Config{{Seed: 6}, {Seed: 6, Int8: true}} {
		blob, err := BuildHNSW(vecs[:1000], cfg).MarshalBinary()
		if err != nil {
			tb.Fatal(err)
		}
		h := load(blob)
		for _, v := range vecs[1000:] {
			h.Add(v)
		}
		for _, q := range campaignUnitVecs(32, 60, 32) {
			hashHits(hh, h.Query(q, 10))
			hashHits(hh, h.QueryEf(q, 10, 200))
		}
	}
	return hex.EncodeToString(hh.Sum(nil))
}

// addAfterLoadHashesBefore are addAfterLoadHash with Load, on the tree before v5 (06c5619: v4 blob, per-node lists),
// keyed by GOARCH (arm64 on the M1 Pro, amd64 on nobara; see campaignHNSWQueryHashBefore).
var addAfterLoadHashesBefore = map[string]string{"arm64": "679d2cf4668cac1cdb72c7354e4e6de23ade32ffc1631d01b9fe7a0837f321ee", "amd64": "16be32100bf8c95814c4caa025188a62115cbfd2cc9a9c17ef1cc5dc2ccfc320"}

// Gate (c): Add after a load (heap or mapped) — which thaws the CSR graph back into lists — builds exactly what Add
// after a load built before v5. (Registered as "matches Add on the original", which is not the contract: a loaded
// index re-seeds its level RNG from the stored seed, where the original's has advanced, so the two diverge by design;
// the first version of this test passed on arm64 only by coincidence of the top 10.) On a mapped index the thaw must
// copy, never write the read-only mapping (which would fault).
func TestHNSW_addAfterLoadMatchesBefore(t *testing.T) {
	want, ok := addAfterLoadHashesBefore[runtime.GOARCH]
	if !ok {
		t.Skipf("no before-hash recorded for GOARCH=%s", runtime.GOARCH)
	}
	heap := addAfterLoadHash(t, func(b []byte) *HNSW {
		h, err := Load(b)
		if err != nil {
			t.Fatal(err)
		}
		return h
	})
	mapped := addAfterLoadHash(t, func(b []byte) *HNSW {
		h, err := LoadHNSWMmap(writeBlob(t, b))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = h.Close() })
		return h
	})
	t.Logf("Add after Load %s, after LoadHNSWMmap %s, before v5 %s", heap, mapped, want)
	if heap != want || mapped != want {
		t.Fatalf("Add after a load differs from before v5 (Load %s, LoadHNSWMmap %s, want %s)", heap, mapped, want)
	}
}

// Gate (d): a truncated or corrupted v5 blob is an ErrFormat error, never a panic, and an accepted corruption still
// answers queries without panicking. Every truncation point and a flip at every byte of a small blob, both modes.
func TestLoad_v5RejectsDamage(t *testing.T) {
	queries := campaignUnitVecs(42, 5, 12)
	for _, cfg := range []Config{{M: 4, Seed: 8}, {M: 4, Seed: 8, Int8: true}} {
		blob, err := BuildHNSW(campaignUnitVecs(41, 120, 12), cfg).MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		for n := range len(blob) {
			if _, err := Load(blob[:n]); !errors.Is(err, ErrFormat) {
				t.Fatalf("int8=%v truncated to %d of %d: err %v, want ErrFormat", cfg.Int8, n, len(blob), err)
			}
		}
		for off := range len(blob) {
			blob[off] ^= 0x5a
			h, err := Load(blob)
			blob[off] ^= 0x5a
			if err != nil {
				if !errors.Is(err, ErrFormat) {
					t.Fatalf("flip at %d: err %v, want ErrFormat", off, err)
				}
				continue
			}
			for _, q := range queries { // must not panic, whatever the flip changed
				h.Query(q, 5)
				h.QueryEf(q, 5, 40)
			}
		}
	}
}
