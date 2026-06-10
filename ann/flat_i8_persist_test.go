package ann

import (
	"bytes"
	"math/rand/v2"
	"reflect"
	"testing"
)

func TestFlatI8_roundTrip(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	vecs := randUnitSet(rng, 500, 64)
	f := NewFlatI8(vecs)

	blob, err := f.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	g, err := LoadFlatI8(blob)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(f, g) {
		t.Fatal("round-trip: loaded index differs from original")
	}
	// Behavioral equivalence: identical query results.
	for i := 0; i < 25; i++ {
		q := randUnit(rng, 64)
		if !reflect.DeepEqual(f.Query(q, 10), g.Query(q, 10)) {
			t.Fatalf("query %d: loaded index returns different hits", i)
		}
	}
	// Re-marshal is byte-identical (canonical format).
	if blob2, _ := g.MarshalBinary(); !bytes.Equal(blob, blob2) {
		t.Error("re-marshal not byte-identical")
	}
}

func TestFlatI8_emptyRoundTrip(t *testing.T) {
	f := NewFlatI8(nil)
	blob, err := f.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	g, err := LoadFlatI8(blob)
	if err != nil {
		t.Fatalf("load empty: %v", err)
	}
	if g.Len() != 0 {
		t.Errorf("empty index Len = %d, want 0", g.Len())
	}
	if g.Query(make([]float32, 64), 5) != nil {
		t.Error("query on empty index should be nil")
	}
}

func TestLoadFlatI8_rejectsBadBlobs(t *testing.T) {
	good, _ := NewFlatI8(randUnitSet(rand.New(rand.NewPCG(3, 4)), 100, 32)).MarshalBinary()

	cases := map[string][]byte{
		"too short":      {1, 2, 3},
		"empty":          {},
		"bad magic":      flip(good, 0),
		"truncated body": good[:len(good)-4],
		"trailing byte":  append(append([]byte(nil), good...), 0),
	}
	for name, blob := range cases {
		if _, err := LoadFlatI8(blob); err == nil {
			t.Errorf("%s: expected an error, got nil", name)
		}
	}
	// Sanity: the unmodified blob still loads.
	if _, err := LoadFlatI8(good); err != nil {
		t.Errorf("unmodified blob failed to load: %v", err)
	}
}

func flip(b []byte, i int) []byte {
	out := append([]byte(nil), b...)
	out[i]++
	return out
}

// FuzzLoadFlatI8 asserts LoadFlatI8 never panics on arbitrary bytes, and that any
// blob it accepts is query-ready and round-trips byte-identically.
func FuzzLoadFlatI8(f *testing.F) {
	seed, _ := NewFlatI8(randUnitSet(rand.New(rand.NewPCG(5, 6)), 50, 16)).MarshalBinary()
	f.Add(seed)
	f.Add([]byte{})
	f.Add([]byte{0, 0, 0, 0})
	empty, _ := NewFlatI8(nil).MarshalBinary()
	f.Add(empty)

	f.Fuzz(func(t *testing.T, data []byte) {
		idx, err := LoadFlatI8(data)
		if err != nil {
			return
		}
		// Accepted → must be queryable without panic.
		_ = idx.Query(make([]float32, idx.dim), 5)
		// …and re-serialize to the same bytes (canonical round-trip).
		re, err := idx.MarshalBinary()
		if err != nil {
			t.Fatalf("re-marshal of a loaded index failed: %v", err)
		}
		if !bytes.Equal(re, data) {
			t.Fatalf("loaded blob did not round-trip byte-identically")
		}
	})
}
