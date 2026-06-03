package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// relL2 is the per-row relative L2 error ‖got-want‖ / ‖want‖.
func relL2(got, want []float32) float64 {
	var num, den float64
	for i := range want {
		d := float64(got[i] - want[i])
		num += d * d
		den += float64(want[i]) * float64(want[i])
	}
	if den == 0 {
		return num
	}
	return math.Sqrt(num / den)
}

func TestQuantizeRowsInt8_roundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const rows, cols = 40, 256
	for _, kind := range []string{"normal", "uniform"} {
		w := make([]float32, rows*cols)
		for i := range w {
			if kind == "normal" {
				w[i] = float32(rng.NormFloat64())
			} else {
				w[i] = float32(rng.Float64()*2 - 1)
			}
		}
		q, scales := QuantizeRowsInt8(w, rows, cols)
		recon := make([]float32, rows*cols)
		for i := range rows {
			DequantizeRowInt8(q[i*cols:(i+1)*cols], scales[i], recon[i*cols:(i+1)*cols])
		}
		if e := relL2(recon, w); e > 1e-2 {
			t.Errorf("%s round-trip relL2 = %.4f, want ≤ 1e-2", kind, e)
		}
	}
}

func TestMatmulBTQ8_closeToF32(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	randVec := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	for _, s := range []struct{ M, K, N int }{
		{1, 640, 1024}, {1, 2048, 640}, {1, 640, 4096}, {4, 256, 300},
	} {
		a := randVec(s.M * s.K)
		b := randVec(s.N * s.K)
		// Reference: exact f32 matmul.
		want := make([]float32, s.M*s.N)
		MatmulBT(a, b, want, s.M, s.K, s.N)
		// int8-weight matmul.
		q, scales := QuantizeRowsInt8(b, s.N, s.K)
		got := make([]float32, s.M*s.N)
		MatmulBTQ8(a, q, scales, got, s.M, s.K, s.N)
		if e := relL2(got, want); e > 2e-2 {
			t.Errorf("shape %+v: int8 matmul relL2 = %.4f, want ≤ 2e-2", s, e)
		}
	}
}
