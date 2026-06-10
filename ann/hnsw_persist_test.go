package ann

import (
	"bytes"
	"encoding/binary"
	"math/rand/v2"
	"reflect"
	"testing"
)

func TestHNSW_roundTrip(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	vecs := randUnitSet(rng, 800, 64)
	orig := BuildHNSW(vecs, Config{M: 12, EfConstruction: 100, EfSearch: 50, Seed: 99})

	blob, err := orig.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	got, err := Load(blob)
	if err != nil {
		t.Fatal(err)
	}

	// Config + structure preserved exactly.
	if got.dim != orig.dim || got.m != orig.m || got.m0 != orig.m0 ||
		got.efConstruction != orig.efConstruction || got.efSearch != orig.efSearch ||
		got.entry != orig.entry || got.maxLayer != orig.maxLayer ||
		got.mL != orig.mL || got.seed != orig.seed || got.heuristic != orig.heuristic || got.Len() != orig.Len() {
		t.Fatalf("config mismatch:\norig %+v\nload %+v", orig, got)
	}
	if !reflect.DeepEqual(got.vecs, orig.vecs) {
		t.Error("vectors differ after round-trip")
	}
	if !reflect.DeepEqual(got.nodes, orig.nodes) {
		t.Error("graph nodes differ after round-trip")
	}

	// The whole point: identical query results.
	for qi := 0; qi < 25; qi++ {
		q := randUnit(rng, 64)
		want := orig.Query(q, 10)
		gotHits := got.Query(q, 10)
		if !reflect.DeepEqual(want, gotHits) {
			t.Fatalf("query %d: loaded results differ\norig=%v\nload=%v", qi, want, gotHits)
		}
	}
}

func TestHNSW_marshalDeterministicAndStable(t *testing.T) {
	vecs := randUnitSet(rand.New(rand.NewPCG(5, 6)), 300, 32)
	orig := BuildHNSW(vecs, Config{M: 8, Seed: 7})

	b1, _ := orig.MarshalBinary()
	b2, _ := orig.MarshalBinary()
	if !bytes.Equal(b1, b2) {
		t.Fatal("MarshalBinary is not deterministic")
	}
	loaded, err := Load(b1)
	if err != nil {
		t.Fatal(err)
	}
	b3, _ := loaded.MarshalBinary()
	if !bytes.Equal(b1, b3) {
		t.Fatal("re-marshal of a loaded index differs from the original blob")
	}
}

func TestLoad_rejectsBadInput(t *testing.T) {
	good, _ := BuildHNSW(randUnitSet(rand.New(rand.NewPCG(1, 1)), 100, 16), Config{Seed: 1}).MarshalBinary()

	cases := map[string][]byte{
		"nil":           nil,
		"too short":     []byte("HN"),
		"bad magic":     []byte("XXXXyyyy"),
		"truncated":     good[:len(good)/2],
		"trailing junk": append(append([]byte{}, good...), 0, 0, 0),
	}
	for name, data := range cases {
		if _, err := Load(data); err == nil {
			t.Errorf("%s: Load succeeded, want error", name)
		}
	}

	// Corrupt the version field of an otherwise-valid blob.
	badVer := append([]byte{}, good...)
	binary.LittleEndian.PutUint32(badVer[4:], 0xDEAD)
	if _, err := Load(badVer); err == nil {
		t.Error("bad version: Load succeeded, want error")
	}

	// An empty index round-trips cleanly.
	empty, err := Load(mustMarshal(t, NewHNSW(Config{})))
	if err != nil {
		t.Fatalf("empty index round-trip: %v", err)
	}
	if empty.Len() != 0 {
		t.Fatalf("empty index Len = %d, want 0", empty.Len())
	}
}

func mustMarshal(t *testing.T, h *HNSW) []byte {
	t.Helper()
	b, err := h.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// FuzzLoadHNSW asserts Load never panics on arbitrary bytes — and that a blob it
// accepts is safe to Query (the integrity pass must catch out-of-range ids and
// layer-inconsistent edges that would otherwise panic mid-query).
func FuzzLoadHNSW(f *testing.F) {
	seed, _ := BuildHNSW(randUnitSet(rand.New(rand.NewPCG(2, 3)), 60, 16), Config{Seed: 1}).MarshalBinary()
	f.Add(seed)
	f.Add([]byte("HNSW\x01\x00\x00\x00"))
	f.Add([]byte{})

	f.Fuzz(func(t *testing.T, data []byte) {
		h, err := Load(data)
		if err != nil {
			return
		}
		// A loaded index must be queryable without panic.
		if h.Len() > 0 && h.dim > 0 && h.dim < 1<<16 {
			_ = h.Query(make([]float32, h.dim), 5)
		}
	})
}
