package embed

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestQ4KRaw_matchesDequant: on a real Q4_K_M file, every Q4_K tensor's raw super-blocks, wrapped as a
// linalg Q4_K WeightMat, dequantize row for row bit-identically to RowDequantizer; every other type
// reports ok=false. AIKIT_Q4K_GGUF overrides the default path; with no file the test skips (a skip is
// not a pass).
func TestQ4KRaw_matchesDequant(t *testing.T) {
	path := os.Getenv("AIKIT_Q4K_GGUF")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "llama-3.2-1b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no Q4_K_M GGUF at %s", path)
	}
	g, err := OpenGGUFMmap(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	checked, declined := 0, 0
	for _, name := range g.Names() {
		dims, raw, ok, err := g.Q4KRaw(name)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if !ok {
			if g.tensors[name].typ == ggmlTypeQ4_K {
				t.Fatalf("%s is Q4_K but Q4KRaw declined it", name)
			}
			declined++
			continue
		}
		if len(dims) != 2 {
			t.Fatalf("%s: dims %v", name, dims)
		}
		in, out := dims[0], dims[1]
		wm, err := linalg.WrapQ4K(raw, out, in)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		_, into, err := g.RowDequantizer(name)
		if err != nil {
			t.Fatal(err)
		}
		got, want := make([]float32, in), make([]float32, in)
		for _, r := range []int{0, out / 2, out - 1} {
			wm.Row(r, got)
			if err := into(r*in, want); err != nil {
				t.Fatal(err)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s row %d [%d]: q4k %v, dequant %v", name, r, i, got[i], want[i])
				}
			}
		}
		checked++
	}
	if checked == 0 {
		t.Fatalf("no Q4_K tensor in %s", path)
	}
	t.Logf("%d Q4_K tensors bit-identical to RowDequantizer, %d other tensors declined", checked, declined)
}
