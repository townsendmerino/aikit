//go:build gpu

package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/townsendmerino/aikit/internal/linalg"
)

// TestWebGPUBackend_matchesCPU validates the M9 WebGPU matmul against the CPU
// reference on real hardware (run with `go test -tags gpu ./decoder/`). It
// skips cleanly when no adapter is present (headless CI). The shapes are the
// decoder's actual M=1 projections plus a slice of the LM head.
func TestWebGPUBackend_matchesCPU(t *testing.T) {
	be, err := newWebGPUBackend()
	if err != nil {
		t.Skipf("no GPU backend: %v", err)
	}
	defer be.Close()
	if be.Name() == "cpu" {
		t.Skip("webgpu fell back to cpu (no adapter)")
	}
	t.Logf("GPU backend: %s", be.Name())

	rng := rand.New(rand.NewSource(1))
	randVec := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	for _, s := range []struct {
		name    string
		M, K, N int
	}{
		{"qproj", 1, 640, 1024},
		{"oproj", 1, 1024, 640},
		{"gate", 1, 640, 2048},
		{"down", 1, 2048, 640},
		{"lmhead_slice", 1, 640, 8192},
	} {
		a := randVec(s.M * s.K)
		b := randVec(s.N * s.K)
		want := make([]float32, s.M*s.N)
		linalg.MatmulBT(a, b, want, s.M, s.K, s.N)
		got := make([]float32, s.M*s.N)
		be.MatmulBT(a, b, got, s.M, s.K, s.N)

		var maxRel float64
		for i := range want {
			d := math.Abs(float64(got[i] - want[i]))
			rel := d / (1 + math.Abs(float64(want[i])))
			if rel > maxRel {
				maxRel = rel
			}
		}
		if maxRel > 1e-3 {
			t.Errorf("%s (M=%d K=%d N=%d): max rel diff %.2e vs CPU, want ≤ 1e-3", s.name, s.M, s.K, s.N, maxRel)
		} else {
			t.Logf("%s: max rel diff %.2e", s.name, maxRel)
		}
		// Call again with a fresh activation: exercises the resident-weight
		// reuse path (the weight must NOT be re-uploaded, result still correct).
		a2 := randVec(s.M * s.K)
		linalg.MatmulBT(a2, b, want, s.M, s.K, s.N)
		be.MatmulBT(a2, b, got, s.M, s.K, s.N)
		for i := range want {
			if math.Abs(float64(got[i]-want[i]))/(1+math.Abs(float64(want[i]))) > 1e-3 {
				t.Errorf("%s reuse: result wrong after resident reuse", s.name)
				break
			}
		}
	}
}

// TestWebGPU_forwardParity runs a full forward on the GPU backend and checks
// the argmax matches the f32 oracle — end-to-end proof that resident weights
// (every projection + the tied LM head) produce correct logits on real
// hardware. Run with `go test -tags gpu ./decoder/`.
func TestWebGPU_forwardParity(t *testing.T) {
	raw, err := os.ReadFile(gemmaForwardGoldenPath)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no forward golden at %s", gemmaForwardGoldenPath)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if _, err := os.Stat(gemmaModelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s", gemmaModelDir)
	}

	m, err := Load(gemmaModelDir, Options{Backend: "webgpu"})
	if err != nil {
		t.Fatalf("Load webgpu: %v", err)
	}
	if m.be.Name() == "cpu" {
		t.Skip("webgpu fell back to cpu (no adapter)")
	}
	t.Logf("backend: %s", m.be.Name())

	cache := m.NewCache(len(g.IDs))
	for _, id := range g.IDs[:len(g.IDs)-1] {
		if _, err := m.runLayers(id, cache); err != nil {
			t.Fatalf("runLayers: %v", err)
		}
	}
	logits, err := m.forward(g.IDs[len(g.IDs)-1], cache)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if got := argmax(logits); got != g.Argmax {
		t.Errorf("GPU argmax = %d, want %d", got, g.Argmax)
	}
}
