package linalg

import (
	"math"
	"math/rand/v2"
	"testing"
)

func randF32(rng *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	return v
}

// seqF64 is the downstream reference: dst[i,j] = float32(Σ_k f64(a)·f64(b)) summed
// in sequential index order.
func seqF64(a, b []float32, M, K, N int) []float32 {
	dst := make([]float32, M*N)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var s float64
			for k := 0; k < K; k++ {
				s += float64(a[i*K+k]) * float64(b[j*K+k])
			}
			dst[i*N+j] = float32(s)
		}
	}
	return dst
}

// TestMatmulBTAcc64_bitIdentical is the precision contract: MatmulBTAcc64 must
// equal the sequential f64 reference bit-for-bit (parallelism is across outputs,
// not within a dot, so no reassociation). Covers attention shapes (K=128, M=1).
func TestMatmulBTAcc64_bitIdentical(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, sh := range []struct{ M, K, N int }{
		{1, 128, 2048}, {512, 128, 257}, {4, 64, 1000}, {1, 128, 1}, {7, 33, 129}, {1, 4096, 3},
	} {
		a := randF32(rng, sh.M*sh.K)
		b := randF32(rng, sh.N*sh.K)
		dst := make([]float32, sh.M*sh.N)
		MatmulBTAcc64(a, b, dst, sh.M, sh.K, sh.N)
		ref := seqF64(a, b, sh.M, sh.K, sh.N)
		for idx := range dst {
			if dst[idx] != ref[idx] {
				t.Fatalf("shape %+v idx %d: %v != f64 ref %v (not bit-identical)", sh, idx, dst[idx], ref[idx])
			}
		}
	}
}

// TestMatmulBTAcc64_beatsF32 demonstrates the motivation: plain f32 MatmulBT
// deviates from the f64 reference (the error that flips an MoE router), while
// MatmulBTAcc64 is exact.
func TestMatmulBTAcc64_beatsF32(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	const M, K, N = 8, 1024, 512 // long K amplifies f32 reassociation
	a := randF32(rng, M*K)
	b := randF32(rng, N*K)
	ref := seqF64(a, b, M, K, N)

	f32 := make([]float32, M*N)
	MatmulBT(a, b, f32, M, K, N)
	acc := make([]float32, M*N)
	MatmulBTAcc64(a, b, acc, M, K, N)

	var f32Max, accMax float64
	for i := range ref {
		f32Max = math.Max(f32Max, math.Abs(float64(f32[i]-ref[i])))
		accMax = math.Max(accMax, math.Abs(float64(acc[i]-ref[i])))
	}
	t.Logf("max |Δ| vs f64 ref — MatmulBT(f32) %.3e, MatmulBTAcc64 %.3e", f32Max, accMax)
	if accMax != 0 {
		t.Errorf("MatmulBTAcc64 should be exact vs the f64 ref, got max |Δ| %.3e", accMax)
	}
	if f32Max == 0 {
		t.Skip("f32 happened to match exactly here; the point stands at scale")
	}
}

func benchShapeMatmul(b *testing.B, name string, fn func(a, bb, dst []float32, M, K, N int)) {
	const M, K, N = 512, 128, 2048
	rng := rand.New(rand.NewPCG(5, 6))
	a := randF32(rng, M*K)
	bb := randF32(rng, N*K)
	dst := make([]float32, M*N)
	b.Run(name, func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			fn(a, bb, dst, M, K, N)
		}
	})
}

func BenchmarkMatmulBTVariants(b *testing.B) {
	benchShapeMatmul(b, "MatmulBT_f32", MatmulBT)
	benchShapeMatmul(b, "MatmulBTAcc64", MatmulBTAcc64)
	benchShapeMatmul(b, "scalar_f64_seq", func(a, bb, dst []float32, M, K, N int) {
		copy(dst, seqF64(a, bb, M, K, N))
	})
}
