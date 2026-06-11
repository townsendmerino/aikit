package ann

import (
	"errors"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"
)

// TestErrFormat: every load path wraps ErrFormat for a corrupt/truncated blob, and a
// valid blob does not.
func TestErrFormat(t *testing.T) {
	bad := []byte("definitely not a serialized index")
	if _, err := Load(bad); !errors.Is(err, ErrFormat) {
		t.Errorf("Load(corrupt): want ErrFormat, got %v", err)
	}
	if _, err := LoadFlatI8(bad); !errors.Is(err, ErrFormat) {
		t.Errorf("LoadFlatI8(corrupt): want ErrFormat, got %v", err)
	}
	p := filepath.Join(t.TempDir(), "bad.bin")
	if err := os.WriteFile(p, bad, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadFlatI8Mmap(p); !errors.Is(err, ErrFormat) {
		t.Errorf("LoadFlatI8Mmap(corrupt): want ErrFormat, got %v", err)
	}

	// A valid blob round-trips without ErrFormat; truncating it re-triggers ErrFormat.
	rng := rand.New(rand.NewPCG(1, 2))
	blob, err := NewFlatI8(randUnitSet(rng, 20, 8)).MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := LoadFlatI8(blob); err != nil {
		t.Errorf("LoadFlatI8(valid): unexpected error %v", err)
	}
	if _, err := LoadFlatI8(blob[:len(blob)-4]); !errors.Is(err, ErrFormat) {
		t.Errorf("LoadFlatI8(truncated): want ErrFormat, got %v", err)
	}
}
