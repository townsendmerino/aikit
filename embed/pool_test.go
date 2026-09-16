package embed

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestL2Normalize(t *testing.T) {
	// Empty slice
	if res := L2Normalize(nil); res != nil {
		t.Fatalf("expected nil, got %v", res)
	}

	// Zero-norm vector
	zeros := []float32{0, 0, 0, 0, 0}
	L2Normalize(zeros)
	for i, v := range zeros {
		if v != 0 {
			t.Fatalf("zeros[%d] = %v, want 0", i, v)
		}
	}

	// Known 3-4-0-0 vector
	v := []float32{3, 4, 0, 0}
	L2Normalize(v)
	if math.Abs(float64(v[0])-0.6) > 1e-6 || math.Abs(float64(v[1])-0.8) > 1e-6 {
		t.Fatalf("v = %v, want [0.6 0.8 0 0]", v)
	}

	// Random vectors of various dimensions comparing against reference math
	rng := rand.New(rand.NewPCG(42, 99))
	for _, dim := range []int{1, 2, 3, 4, 7, 8, 15, 16, 64, 128, 256, 384, 768, 1024} {
		vec := make([]float32, dim)
		for i := range vec {
			vec[i] = float32(rng.NormFloat64())
		}
		ref := append([]float32(nil), vec...)
		var sq float64
		for _, x := range ref {
			sq += float64(x) * float64(x)
		}
		norm := math.Sqrt(sq)
		for i, x := range ref {
			ref[i] = float32(float64(x) / norm)
		}

		L2Normalize(vec)
		for i := range vec {
			if math.Float32bits(vec[i]) != math.Float32bits(ref[i]) {
				t.Fatalf("dim=%d i=%d: got %v (%08x), want %v (%08x)",
					dim, i, vec[i], math.Float32bits(vec[i]), ref[i], math.Float32bits(ref[i]))
			}
		}
	}
}
