package ann

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"math/rand/v2"
	"reflect"
	"runtime"
	"testing"
)

// TestHNSW_WriteToMatchesMarshal is the format-identity gate. Both surfaces run
// the same encode(), differing only in whether the writer's buffer is the whole
// output or a 64 KiB window, so they cannot encode different bytes by
// construction — but that construction is the thing under test, and a future edit
// that gives MarshalBinary its own encoder again would be caught here rather than
// at the next person's Load.
//
// Both storage modes: the int8 path writes its code block straight out through an
// unsafe alias while f32 goes byte-by-byte, so they exercise different writer code.
func TestHNSW_WriteToMatchesMarshal(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	for _, tc := range []struct {
		name string
		cfg  Config
	}{
		{"f32", Config{M: 8, EfConstruction: 60, EfSearch: 40, Seed: 21}},
		{"int8", Config{M: 8, EfConstruction: 60, EfSearch: 40, Seed: 21, Int8: true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := BuildHNSW(randUnitSet(rng, 400, 48), tc.cfg)
			want, err := h.MarshalBinary()
			if err != nil {
				t.Fatal(err)
			}
			var buf bytes.Buffer
			n, err := h.WriteTo(&buf)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(buf.Bytes(), want) {
				t.Fatalf("WriteTo bytes differ from MarshalBinary (%d vs %d bytes)", buf.Len(), len(want))
			}
			if n != int64(len(want)) {
				t.Errorf("WriteTo returned n=%d, wrote %d bytes", n, len(want))
			}
			// blobSize is MarshalBinary's allocation size. If it drifts low the
			// encoder grows mid-write and the exact-size allocation is lost — which
			// is precisely the bug the previous version had, 131.0 MB for a 58.3 MB
			// blob; if it drifts high it over-allocates. Neither shows up as a
			// failure anywhere else.
			if got := h.blobSize(); got != len(want) {
				t.Errorf("blobSize() = %d, actual blob is %d bytes", got, len(want))
			}
			if _, err := Load(buf.Bytes()); err != nil {
				t.Fatalf("Load of streamed bytes: %v", err)
			}
		})
	}
}

// TestHNSW_formatGolden freezes the serialized bytes for a fixed fixture.
//
// The round-trip tests cannot catch a format change — they marshal and load with
// the same code, so a self-consistent change passes every one of them while every
// blob already written to disk or //go:embed-ed becomes unreadable. These hashes
// were captured from the tree BEFORE the WriteTo work and are unchanged by it,
// which is what makes that refactor a pure footprint change.
//
// A deliberate format change bumps hnswVersion AND these constants, in one commit,
// so the diff shows both.
func TestHNSW_formatGolden(t *testing.T) {
	for _, tc := range []struct {
		name  string
		i8    bool
		size  int
		sha   string // v5, MarshalBinary
		v4sha string // the same build in the v4 layout (encodeV4ForTest): pins that v5 changed the encoding only
	}{
		{"f32", false, 379248, "0f56602b7531450297933bf805f3d5f72f8861c4b583633926d504fc9c894687", "2f295d9ac4c985e17d5b7a167bff770fdca2c9c915f8a32754070b8a88666388"},
		{"int8", true, 205248, "679e493364926e9c3d6135fb7dbd4cb8488207c117fd89d944625a572e83b10f", "33bbc99f6dd210899a733ca3a4b44b61ec88b7ffbb054bd3f47acdd365673754"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(4242, 99))
			h := BuildHNSW(randUnitSet(rng, 1500, 40),
				Config{M: 10, EfConstruction: 80, EfSearch: 40, Seed: 31, Int8: tc.i8})
			b, err := h.MarshalBinary()
			if err != nil {
				t.Fatal(err)
			}
			if len(b) != tc.size {
				t.Errorf("blob is %d bytes, want %d", len(b), tc.size)
			}
			if got := hex.EncodeToString(sha256Sum(b)); got != tc.sha {
				t.Errorf("serialized bytes changed:\n got %s\nwant %s\n"+
					"If this is intentional, bump hnswVersion in the same commit.", got, tc.sha)
			}
			if got := hex.EncodeToString(sha256Sum(encodeV4ForTest(h))); got != tc.v4sha {
				t.Errorf("the graph changed (its v4 encoding, which v5 replaced but must not alter):\n got %s\nwant %s", got, tc.v4sha)
			}
		})
	}
}

func sha256Sum(b []byte) []byte {
	s := sha256.Sum256(b)
	return s[:]
}

// errAfterWriter fails the Nth Write, to check WriteTo reports a write error
// rather than reporting success for bytes that never landed.
type errAfterWriter struct {
	ok  int
	n   int
	err error
}

func (w *errAfterWriter) Write(p []byte) (int, error) {
	if w.n >= w.ok {
		return 0, w.err
	}
	w.n++
	return len(p), nil
}

// TestHNSW_WriteToPropagatesWriteError checks the sticky-error path at the FIRST
// write and at a LATER one. The fixture is deliberately larger than the 64 KiB
// buffer — 2000×64 f32 is ~512 KB, so the encode flushes ~8 times — because a
// small index fits in one Write and "fail at write 2" would then never fire,
// making the later-flush case a test that silently proves nothing.
func TestHNSW_WriteToPropagatesWriteError(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, tc := range []struct {
		name string
		cfg  Config
	}{
		{"f32", Config{M: 8, Seed: 3}},
		{"int8", Config{M: 8, Seed: 3, Int8: true}}, // exercises raw()'s direct Write too
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := BuildHNSW(randUnitSet(rng, 2000, 64), tc.cfg)
			var buf bytes.Buffer
			if _, err := h.WriteTo(&buf); err != nil {
				t.Fatal(err)
			}
			if buf.Len() <= 64<<10 {
				t.Fatalf("fixture is %d bytes, must exceed the 64 KiB buffer to test a later flush", buf.Len())
			}
			boom := io.ErrShortWrite
			for _, ok := range []int{0, 1, 3} {
				w := &errAfterWriter{ok: ok, err: boom}
				n, err := h.WriteTo(w)
				if err != boom {
					t.Errorf("failing at write %d: got err %v, want %v", ok, err, boom)
				}
				if n != int64(ok)*(64<<10) && ok == 0 && n != 0 {
					t.Errorf("failing at write 0 reported %d bytes written", n)
				}
			}
		})
	}
}

// TestLoad_graphArenasCutAllocations gates the load-side arenas. Before them Load
// did two allocations per node — one [][]int32 of layer headers and one []int32
// per layer — which is >2.1M allocations at 1M docs for a structure that is one
// contiguous run of int32s in the blob.
//
// The assertion is a per-doc ALLOCATION RATE, not a total: it has to hold at any
// fixture size, and the pre-arena code was ≥2/doc by construction. Anything under
// 1/doc means the arenas are being used; the observed value is ~0.02.
func TestLoad_graphArenasCutAllocations(t *testing.T) {
	const n = 2000
	h := BuildHNSW(randUnitSet(rand.New(rand.NewPCG(9, 10)), n, 32), Config{M: 8, EfConstruction: 60, Seed: 4})
	blob, err := h.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	loaded, err := Load(blob)
	runtime.ReadMemStats(&after)
	if err != nil {
		t.Fatal(err)
	}
	perDoc := float64(after.Mallocs-before.Mallocs) / float64(n)
	t.Logf("Load allocations: %d total, %.3f per doc (%d docs)", after.Mallocs-before.Mallocs, perDoc, n)
	if perDoc >= 1.0 {
		t.Errorf("Load made %.2f allocations per doc — the graph arenas are not being used "+
			"(the per-node version was ≥2/doc)", perDoc)
	}
	runtime.KeepAlive(loaded)
}

// TestLoad_arenaSubSlicesDoNotAlias: every neighbor list a loaded index hands out (a CSR sub-slice of one id array)
// is capped at its own length, so appending to one node's list cannot overwrite the next node's — and the same holds
// after a thaw. Without the three-index cap this test corrupts the graph, and nothing else in the suite would notice.
func TestLoad_arenaSubSlicesDoNotAlias(t *testing.T) {
	h := BuildHNSW(randUnitSet(rand.New(rand.NewPCG(13, 14)), 500, 32), Config{M: 8, Seed: 5})
	blob, err := h.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(blob)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.csr == nil {
		t.Fatal("a loaded index is not in CSR form; this test checks the CSR lists")
	}
	for _, stage := range []string{"csr", "thawed"} {
		if stage == "thawed" {
			loaded.thaw()
		}
		want := graphOf(loaded)
		// Append to every neighbor list; with correct caps nothing else moves.
		for d := range want {
			for l := range want[d] {
				_ = append(loaded.nbrs(d, l), -12345) //nolint:gocritic // the discarded append IS the test
			}
		}
		if got := graphOf(loaded); !reflect.DeepEqual(got, want) {
			t.Fatalf("%s: the graph changed after appending to its neighbor lists (sub-slices are aliasing)", stage)
		}
	}
	// The f32 vector arena has the same property.
	for d := range loaded.vecs {
		_ = append(loaded.vecs[d], 999) //nolint:gocritic // ditto
	}
	if !reflect.DeepEqual(loaded.vecs, h.vecs) {
		t.Error("vector rows changed after appending — the vector arena is aliasing")
	}
}

// TestScanGraph_truncatedIsNotFatal: scanGraph reporting ok=false must leave Load
// working, because its only job is to size the arenas. take() then falls back to
// make() and the read produces the same error it always did.
func TestScanGraph_truncatedIsNotFatal(t *testing.T) {
	h := BuildHNSW(randUnitSet(rand.New(rand.NewPCG(15, 16)), 200, 16), Config{M: 8, Seed: 6})
	blob, err := h.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	for _, cut := range []int{1, 4, 17, len(blob) / 2, len(blob) - 4} {
		if cut <= 0 || cut >= len(blob) {
			continue
		}
		if _, err := Load(blob[:cut]); err == nil {
			t.Errorf("Load of a blob truncated to %d bytes succeeded", cut)
		}
	}
}

// TestQuery_scratchIsPooled gates the search scratch. searchLayer's two heaps
// grow by append from nil to ef — about 7 doublings each — so an unpooled query
// allocates ~19 times and ~13.4 KB where a pooled one allocates twice. Nothing
// else in the suite can see that: the results are identical either way, which is
// the whole point of a pool.
//
// It measures BOTH arms in this process and compares them, rather than asserting
// an absolute count. An absolute bound is not portable across build modes: under
// -race the same pooled query allocates 10 times, not 2, and a bound tuned to the
// normal build fails a race run for no reason (it did). A ratio is immune to that
// and to GC settings, Go version, and this fixture's size.
func TestQuery_scratchIsPooled(t *testing.T) {
	h := BuildHNSW(randUnitSet(rand.New(rand.NewPCG(21, 22)), 3000, 32),
		Config{M: 16, EfConstruction: 100, EfSearch: 64, Seed: 8})
	q := randUnit(rand.New(rand.NewPCG(23, 24)), 32)
	h.Query(q, 10) // warm the pool: the first query legitimately grows the heaps

	// Cold arm: hand the pool a scratch with no heap capacity, reproducing the
	// pre-change behaviour where every searchLayer re-grew both heaps from nil.
	cold := testing.AllocsPerRun(200, func() {
		sc := h.getScratch()
		sc.cands.items, sc.results.items = nil, nil
		h.putScratch(sc)
		h.Query(q, 10)
	})
	// Measure the pooled (warm) arm as the MIN over several samples. Under -race the
	// race allocator adds transient, variable allocations to the pooled path — a single
	// AllocsPerRun spikes (7–13 seen where the steady floor is ~7) — so the reused-heap
	// cost is the floor, and comparing that floor to the (stable) cold arm denoises the
	// ratio. §1.34: the saving compresses from ~9.5× normally to ~2.5× under -race, and
	// the raw warm count is too noisy there to compare at a 2× bar without this — this
	// test's own predecessor already learned that -race moves the counts, and the CI
	// arm64 `go test -race ./...` job is where the residual noise tipped it red.
	warm := testing.AllocsPerRun(200, func() { h.Query(q, 10) })
	for range 4 {
		warm = min(warm, testing.AllocsPerRun(200, func() { h.Query(q, 10) }))
	}
	t.Logf("Query allocations: %.1f cold heaps, %.1f pooled min-of-5 (%.1f×)", cold, warm, cold/warm)
	if warm*2 > cold {
		t.Errorf("pooled query allocates %.1f per call (min of 5) against %.1f with cold heaps — "+
			"less than a 2× saving means the search heaps are not being reused", warm, cold)
	}
}
