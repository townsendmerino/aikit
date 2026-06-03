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

func TestQuantizeGroupsInt4_roundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	// 300 = ragged final group at group 32 (300 = 9*32 + 12) — exercises the
	// remainder path in both quantize and dequant.
	for _, cols := range []int{256, 300, 2048} {
		const rows, group = 16, 32
		w := make([]float32, rows*cols)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		packed, scales := QuantizeGroupsInt4(w, rows, cols, group)
		nGroups, bpr := groupsFor(cols, group)
		recon := make([]float32, rows*cols)
		for i := range rows {
			DequantizeRowInt4(packed[i*bpr:(i+1)*bpr], scales[i*nGroups:(i+1)*nGroups], group, cols, recon[i*cols:(i+1)*cols])
		}
		// 4-bit is coarse but per-group scaling keeps the relative L2 modest.
		if e := relL2(recon, w); e > 1.5e-1 {
			t.Errorf("cols=%d int4 round-trip relL2 = %.4f, want ≤ 0.15", cols, e)
		}
	}
}

// TestMatmulBTQ4_matchesDequant: the int4 matmul must equal an exact f32 matmul
// against the dequantized weights — the kernel and DequantizeRowInt4 share one
// reconstruction formula, so this is bit-exact (any drift is a packing/indexing
// bug), independent of the (separately bounded) quantization error.
func TestMatmulBTQ4_matchesDequant(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	randVec := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	const group = 32
	for _, s := range []struct{ M, K, N int }{
		{1, 640, 1024}, {1, 2048, 640}, {1, 300, 128}, {4, 256, 300},
	} {
		a := randVec(s.M * s.K)
		b := randVec(s.N * s.K)
		packed, scales := QuantizeGroupsInt4(b, s.N, s.K, group)

		// Reference: dequantize b row-by-row, then exact f32 matmul.
		nGroups, bpr := groupsFor(s.K, group)
		bDeq := make([]float32, s.N*s.K)
		for j := range s.N {
			DequantizeRowInt4(packed[j*bpr:(j+1)*bpr], scales[j*nGroups:(j+1)*nGroups], group, s.K, bDeq[j*s.K:(j+1)*s.K])
		}
		want := make([]float32, s.M*s.N)
		MatmulBT(a, bDeq, want, s.M, s.K, s.N)

		got := make([]float32, s.M*s.N)
		MatmulBTQ4(a, packed, scales, got, s.M, s.K, s.N, group)
		for i := range want {
			if d := math.Abs(float64(got[i] - want[i])); d > 1e-4 {
				t.Fatalf("shape %+v: MatmulBTQ4[%d]=%v != dequant matmul %v (Δ%.2e)", s, i, got[i], want[i], d)
			}
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
