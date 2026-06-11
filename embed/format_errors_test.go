package embed

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// TestErrFormat: the loaders wrap ErrFormat for bad-magic / unsupported-version /
// truncated blobs (so callers can errors.Is instead of string-matching).
func TestErrFormat(t *testing.T) {
	dir := t.TempDir()
	write := func(name string, b []byte) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, b, 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	if _, err := OpenSafetensors(write("short.st", []byte("xx"))); !errors.Is(err, ErrFormat) {
		t.Errorf("safetensors truncated: want ErrFormat, got %v", err)
	}
	if _, err := OpenGGUF(write("badmagic.gguf", []byte("NOPE\x00\x00\x00\x00\x00\x00\x00\x00"))); !errors.Is(err, ErrFormat) {
		t.Errorf("gguf bad magic: want ErrFormat, got %v", err)
	}
	// correct magic ("GGUF"), unsupported version 99
	if _, err := OpenGGUF(write("badver.gguf", []byte("GGUF\x63\x00\x00\x00\x00\x00\x00\x00"))); !errors.Is(err, ErrFormat) {
		t.Errorf("gguf bad version: want ErrFormat, got %v", err)
	}
}
