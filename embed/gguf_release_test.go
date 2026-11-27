package embed

import (
	"path/filepath"
	"slices"
	"testing"
)

// releaseFixture is a GGUF of two F32 tensors, the first a few pages long, with its values.
func releaseFixture(t *testing.T) (string, []float32) {
	t.Helper()
	big := make([]float32, 1<<16) // 256 KiB
	for i := range big {
		big[i] = float32(i%977) * 0.5
	}
	path := filepath.Join(t.TempDir(), "release.gguf")
	writeTestGGUF(t, path, []ggufKV{{"general.architecture", "llama"}}, []ggufF32{
		{"big", []uint64{256, 256}, big},
		{"small", []uint64{4}, []float32{1, 2, 3, 4}},
	})
	return path, big
}

// TestGGUFReleaseTensors_readsBackUnchanged: a released tensor reads back bit for bit on every kind of file, a heap-backed one
// keeps its bytes (advice would discard them), and an unknown name is an error.
func TestGGUFReleaseTensors_readsBackUnchanged(t *testing.T) {
	path, big := releaseFixture(t)
	mm, err := OpenGGUFMmap(path)
	if err != nil {
		t.Fatal(err)
	}
	defer mm.Close()
	heap, err := OpenGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, g := range []*GGUFFile{mm, heap} {
		if _, _, err := g.Tensor("big"); err != nil {
			t.Fatal(err)
		}
		if err := g.ReleaseTensors("big"); err != nil {
			t.Fatalf("ReleaseTensors: %v", err)
		}
		_, got, err := g.Tensor("big")
		if err != nil || !slices.Equal(got, big) {
			t.Fatalf("after ReleaseTensors the tensor reads back changed (err %v)", err)
		}
		if _, small, _ := g.Tensor("small"); !slices.Equal(small, []float32{1, 2, 3, 4}) {
			t.Fatalf("the neighbour tensor changed: %v", small)
		}
		if err := g.ReleaseTensors("missing"); err == nil {
			t.Fatal("ReleaseTensors of an unknown tensor returned no error")
		}
	}
}
