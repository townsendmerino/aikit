package ann

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"hash"
	"math"
	"math/rand/v2"
	"runtime"
	"sync"
	"testing"
)

// The memory campaign's HNSW gates (docs/internal/task-memory-campaign.md items 4 and 7) check one claim: a
// change to how the graph is built or stored changes no search result. campaignHNSWHash is the fingerprint.

// campaignUnitVecs is n seeded unit vectors. It uses only PCG's Uint64 stream (a fixed algorithm), not
// NormFloat64, so the corpus cannot drift with a Go release.
func campaignUnitVecs(seed uint64, n, dim int) [][]float32 {
	rng := rand.New(rand.NewPCG(seed, seed^0x5bd1e995))
	out := make([][]float32, n)
	for i := range out {
		v := make([]float32, dim)
		var ss float64
		for j := range v {
			x := float64(rng.Uint64()>>11)/float64(1<<53)*2 - 1
			v[j] = float32(x)
			ss += x * x
		}
		inv := float32(1 / math.Sqrt(ss))
		for j := range v {
			v[j] *= inv
		}
		out[i] = v
	}
	return out
}

// campaignHNSWConfigs are the builds the gate covers: both storage modes, both neighbor selections, two seeds.
func campaignHNSWConfigs() []Config {
	var cs []Config
	for _, int8 := range []bool{false, true} {
		for _, simple := range []bool{false, true} {
			for _, seed := range []uint64{1, 42} {
				cs = append(cs, Config{Int8: int8, SimpleNeighbors: simple, Seed: seed})
			}
		}
	}
	return cs
}

const campaignHNSWN, campaignHNSWDim = 5000, 64

func hashHits(h hash.Hash, hits []Hit) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], uint64(len(hits)))
	h.Write(b[:])
	for _, x := range hits {
		binary.LittleEndian.PutUint64(b[:], uint64(x.Index))
		h.Write(b[:])
		binary.LittleEndian.PutUint64(b[:], math.Float64bits(x.Score))
		h.Write(b[:])
	}
}

// campaignQueryHash fingerprints what a query can observe of h: Query at k = 1, 10, 50 and QueryEf at ef = 16 and
// 200, for 200 seeded queries.
func campaignQueryHash(hh hash.Hash, h *HNSW) {
	for _, q := range campaignUnitVecs(9001, 200, campaignHNSWDim) {
		for _, k := range []int{1, 10, 50} {
			hashHits(hh, h.Query(q, k))
		}
		hashHits(hh, h.QueryEf(q, 10, 16))
		hashHits(hh, h.QueryEf(q, 10, 200))
	}
}

// campaignBuilds builds each campaign config over the same seeded corpus once per test process — the builds are the
// expensive part (minutes under -race), and three gates read them — and records: the query hash (every config's query
// results), the blob hash (each config's query results followed by its v4 encoding, so a build change that happened
// to leave the results alone still shows), and each config's saved blob.
var campaignBuilds = sync.OnceValue(func() (out struct {
	queryHash, blobHash string
	blobs               [][]byte
}) {
	vecs := campaignUnitVecs(7, campaignHNSWN, campaignHNSWDim)
	hq, hb := sha256.New(), sha256.New()
	for _, cfg := range campaignHNSWConfigs() {
		h := BuildHNSW(vecs, cfg)
		campaignQueryHash(teeHash{hq, hb}, h)
		hb.Write(encodeV4ForTest(h))
		blob, err := h.MarshalBinary()
		if err != nil {
			panic(err)
		}
		out.blobs = append(out.blobs, blob)
	}
	out.queryHash, out.blobHash = hex.EncodeToString(hq.Sum(nil)), hex.EncodeToString(hb.Sum(nil))
	return out
})

// The blob part of the hash is encodeV4ForTest's bytes since v5 (item 7): a test-only v4 writer, so the v4-era capture
// below still proves the built graph unchanged byte for byte.
// Item 4: pre-sizing the build changes neither a search result nor a byte of the serialized graph.
func TestBuildHNSW_campaignHashMatchesBefore(t *testing.T) {
	if testing.Short() {
		t.Skip("builds 8 indexes of 5,000 vectors")
	}
	q, b := campaignBuilds().queryHash, campaignBuilds().blobHash
	t.Logf("query hash %s, blob hash %s", q, b)
	if q != campaignHNSWQueryHashBefore(t) {
		t.Errorf("query hash %s, before %s: a search result changed", q, campaignHNSWQueryHashBefore(t))
	}
	if b != campaignHNSWBlobHashBefore(t) {
		t.Errorf("blob hash %s, before %s: the built graph changed", b, campaignHNSWBlobHashBefore(t))
	}
}

// Item 4, reported: allocations per Add on a 5,000-vector build, f32 and int8.
func TestHNSW_allocsPerAdd(t *testing.T) {
	if raceEnabled {
		t.Skip("a report, not a gate; under -race the build takes minutes and the allocation count is the detector's")
	}
	vecs := campaignUnitVecs(7, campaignHNSWN, campaignHNSWDim)
	for _, int8 := range []bool{false, true} {
		h := NewHNSW(Config{Int8: int8, Seed: 1})
		for _, v := range vecs[:4000] {
			h.Add(v)
		}
		i := 4000
		allocs := testing.AllocsPerRun(999, func() {
			h.Add(vecs[i])
			i++
		})
		t.Logf("int8=%v: %.2f allocations per Add (vectors 4,001-5,000)", int8, allocs)
	}
}

// encodeV4ForTest writes h in the v4 layout MarshalBinary used before v5 (item 7): the same header and vectors, no
// int8 scale pad, and the graph as per-node layer/count/ids runs. Only for comparing against v4-era hashes.
func encodeV4ForTest(h *HNSW) []byte {
	b, err := h.MarshalBinary()
	if err != nil {
		panic(err)
	}
	binary.LittleEndian.PutUint32(b[4:], hnswV4)
	vecEnd := hnswHeaderBytes + h.count()*h.dim*4
	if h.int8 {
		vecEnd = hnswHeaderBytes + len(h.bq) + len(h.scales)*4
		codesEnd := hnswHeaderBytes + len(h.bq)
		b = append(b[:codesEnd:codesEnd], b[codesEnd+pad4(len(h.bq)):]...) // drop the v5 pad
	}
	out := append([]byte{}, b[:vecEnd]...)
	var w [4]byte
	put := func(v int) {
		binary.LittleEndian.PutUint32(w[:], uint32(int32(v)))
		out = append(out, w[:]...)
	}
	for d := range h.count() {
		put(h.nodeLayer(d))
		for l := 0; l <= h.nodeLayer(d); l++ {
			nb := h.nbrs(d, l)
			put(len(nb))
			for _, id := range nb {
				put(int(id))
			}
		}
	}
	return out
}

// campaignHNSWQueryHashBefores are campaignHNSWHash without the blob, on the tree before the build pre-sizing (item 4), keyed by GOARCH: the SIMD kernels and the compiler's fused multiply-adds differ by
// architecture, so each arch has its own capture, each taken on its own pre-change tree — arm64 on the M1 Pro, amd64
// on nobara (2026-10-17, the pre-change commit checked out with this test file added). An arch with no capture skips.
var campaignHNSWQueryHashBefores = map[string]string{"arm64": "447c723ea62f87e1ea34f77f687c128354dd7e8864039edbbea3e3201a9cba53", "amd64": "529fc17d0a97dab78d1069bcc971b9ee29f6b42c584f19c5a7e94314345478c7"}

func campaignHNSWQueryHashBefore(tb testing.TB) string {
	tb.Helper()
	h, ok := campaignHNSWQueryHashBefores[runtime.GOARCH]
	if !ok {
		tb.Skipf("no before-hash recorded for GOARCH=%s", runtime.GOARCH)
	}
	return h
}

// campaignHNSWBlobHashBefores are campaignHNSWHash with the (v4) blob, on the same tree, keyed by GOARCH: the SIMD kernels and the compiler's fused multiply-adds differ by
// architecture, so each arch has its own capture, each taken on its own pre-change tree — arm64 on the M1 Pro, amd64
// on nobara (2026-10-17, the pre-change commit checked out with this test file added). An arch with no capture skips.
var campaignHNSWBlobHashBefores = map[string]string{"arm64": "3178f09875d1f63869be6d53bb551045ca783b378886bd71d677206d9faf2d73", "amd64": "10dfc0501a713ae523c0be54cb21340e6f6497d0252b7e9275e91d9b0cf30dd0"}

func campaignHNSWBlobHashBefore(tb testing.TB) string {
	tb.Helper()
	h, ok := campaignHNSWBlobHashBefores[runtime.GOARCH]
	if !ok {
		tb.Skipf("no before-hash recorded for GOARCH=%s", runtime.GOARCH)
	}
	return h
}

// teeHash writes to two hashes at once (Sum and the rest are the first's): one pass feeds the query and blob hashes.
type teeHash struct {
	hash.Hash
	also hash.Hash
}

func (t teeHash) Write(p []byte) (int, error) {
	t.also.Write(p)
	return t.Hash.Write(p)
}
