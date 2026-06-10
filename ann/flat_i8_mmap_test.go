package ann

import (
	"math/rand/v2"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func writeFlatI8Blob(t *testing.T, f *FlatI8) string {
	t.Helper()
	blob, err := f.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "index.fi8")
	if err := os.WriteFile(p, blob, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// TestLoadFlatI8Mmap_matchesCopy: the zero-copy mmap index must be byte- and
// behavior-identical to the in-memory one built from the same corpus.
func TestLoadFlatI8Mmap_matchesCopy(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	vecs := randUnitSet(rng, 800, 64)
	orig := NewFlatI8(vecs)

	m, err := LoadFlatI8Mmap(writeFlatI8Blob(t, orig))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	if m.Len() != orig.Len() || m.dim != orig.dim {
		t.Fatalf("shape: got (%d,%d) want (%d,%d)", m.Len(), m.dim, orig.Len(), orig.dim)
	}
	if !reflect.DeepEqual(m.scales, orig.scales) {
		t.Error("scales differ")
	}
	if !reflect.DeepEqual([]int8(m.bq), orig.bq) {
		t.Error("aliased codes differ from original")
	}
	for i := 0; i < 25; i++ {
		q := randUnit(rng, 64)
		if !reflect.DeepEqual(m.Query(q, 10), orig.Query(q, 10)) {
			t.Fatalf("query %d: mmap result differs from in-memory", i)
		}
	}
}

func TestFlatI8Mmap_closePanicsOnQuery(t *testing.T) {
	path := writeFlatI8Blob(t, NewFlatI8(randUnitSet(rand.New(rand.NewPCG(3, 4)), 100, 32)))
	m, err := LoadFlatI8Mmap(path)
	if err != nil {
		t.Fatal(err)
	}
	q := randUnit(rand.New(rand.NewPCG(5, 6)), 32)
	_ = m.Query(q, 5) // fine before Close

	if err := m.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if err := m.Close(); err != nil {
		t.Fatalf("second close (must be idempotent): %v", err)
	}
	defer func() {
		if recover() == nil {
			t.Error("Query after Close should panic")
		}
	}()
	m.Query(q, 5)
}

// In-memory indexes have nothing to release: Close is a no-op and leaves them
// queryable (only a mmap-backed index becomes unusable after Close).
func TestFlatI8_inMemoryCloseIsNoop(t *testing.T) {
	f := NewFlatI8(randUnitSet(rand.New(rand.NewPCG(7, 8)), 50, 16))
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if got := f.Query(randUnit(rand.New(rand.NewPCG(9, 10)), 16), 5); got == nil {
		t.Error("in-memory index should still query after Close")
	}
}

func TestLoadFlatI8Mmap_emptyAndBad(t *testing.T) {
	m, err := LoadFlatI8Mmap(writeFlatI8Blob(t, NewFlatI8(nil)))
	if err != nil {
		t.Fatalf("empty index via mmap: %v", err)
	}
	if m.Len() != 0 {
		t.Errorf("empty len %d", m.Len())
	}
	_ = m.Close()

	bad := filepath.Join(t.TempDir(), "bad.fi8")
	if err := os.WriteFile(bad, []byte("not a valid FlatI8 blob, but long enough"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadFlatI8Mmap(bad); err == nil {
		t.Error("corrupt blob should error (and not leak the mapping)")
	}
	if _, err := LoadFlatI8Mmap(filepath.Join(t.TempDir(), "nope.fi8")); err == nil {
		t.Error("missing file should error")
	}
}
