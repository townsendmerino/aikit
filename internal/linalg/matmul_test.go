package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// naiveMatmulBT is the obvious triple-loop reference MatmulBT must match.
func naiveMatmulBT(a, b []float32, M, K, N int) []float32 {
	dst := make([]float32, M*N)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var s float32
			for k := 0; k < K; k++ {
				s += a[i*K+k] * b[j*K+k]
			}
			dst[i*N+j] = s
		}
	}
	return dst
}

func TestMatmulBT_matchesNaive(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	randVec := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	// Cover M=1 (decode), M>1 (prefill-style), and shapes straddling the
	// serial/parallel threshold and the dot kernel's 4-wide tail.
	shapes := []struct{ M, K, N int }{
		{1, 4, 1}, {1, 64, 1}, {1, 640, 1}, // tiny / serial
		{1, 640, 1024}, {1, 640, 2048}, {1, 2048, 640}, // decode projections
		{1, 640, 262144},                  // LM head
		{1, 6, 5}, {1, 7, 9}, {1, 13, 17}, // non-multiple-of-4 K (scalar tail)
		{4, 640, 1024}, {8, 256, 300}, // M>1
	}
	for _, s := range shapes {
		a := randVec(s.M * s.K)
		b := randVec(s.N * s.K)
		got := make([]float32, s.M*s.N)
		MatmulBT(a, b, got, s.M, s.K, s.N)
		want := naiveMatmulBT(a, b, s.M, s.K, s.N)
		for i := range want {
			// f32 reduction-order differences vs the kernel: relative tol.
			d := math.Abs(float64(got[i] - want[i]))
			if d > 1e-3+1e-3*math.Abs(float64(want[i])) {
				t.Fatalf("shape %+v out[%d] = %g, want %g (Δ%g)", s, i, got[i], want[i], d)
				break
			}
		}
	}
}
