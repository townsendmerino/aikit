package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// TestW4A8_dotMatchesScalar is the asm-correctness gate: the fused SDOT kernel
// (exercised through dotW4A8 on a DotProd box) must match the portable scalar
// reference. The integer accumulation is exact; only the per-group f32 fold can
// differ by float rounding, so the bar is tight.
func TestW4A8_dotMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	const group = 32
	for _, K := range []int{32, 64, 1536, 2048, 96, 300 /* ragged tail */} {
		nGroups := (K + group - 1) / group
		act := make([]int8, K)
		for i := range act {
			act[i] = int8(rng.Intn(255) - 128)
		}
		packed := make([]byte, (K+1)/2)
		for i := range packed {
			packed[i] = byte(rng.Intn(256))
		}
		scales := make([]float32, nGroups)
		for i := range scales {
			scales[i] = float32(rng.NormFloat64())
		}
		got := dotW4A8(act, packed, scales, group, K)
		want := dotW4A8Scalar(act, packed, scales, group, K)
		rel := math.Abs(float64(got-want)) / (math.Abs(float64(want)) + 1e-9)
		if rel > 1e-5 {
			t.Errorf("K=%d: dotW4A8=%v scalar=%v relErr=%.2e, want ≤ 1e-5", K, got, want, rel)
		}
	}
}

// TestMatmulBTW4A8_closeToF32 mirrors the W8A8 parity gate: against the f32
// matmul of the SAME dequantized int4 weights, W4A8's only added error is the
// int8 activation quantization, so it stays within the W8A8 tolerance (5e-2).
func TestMatmulBTW4A8_closeToF32(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	randVec := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	for _, s := range []struct {
		M, K, N, group int
	}{
		{1, 640, 1024, 32},
		{1, 2048, 640, 32},
		{64, 256, 320, 32},
		{1, 300, 128, 32},  // ragged final group (K % 32 ≠ 0)
		{1, 512, 256, 64},  // non-32 group → scalar path
		{4, 384, 192, 128}, // larger group
	} {
		a := randVec(s.M * s.K)
		b := randVec(s.N * s.K)
		packed, scales := QuantizeGroupsInt4(b, s.N, s.K, s.group)

		// Reference: dequantize the int4 weights to f32, then f32 matmul — so the
		// only difference left is W4A8's int8 activation quantization.
		nGroups, bpr := groupsFor(s.K, s.group)
		bDeq := make([]float32, s.N*s.K)
		for j := range s.N {
			DequantizeRowInt4(packed[j*bpr:(j+1)*bpr], scales[j*nGroups:(j+1)*nGroups], s.group, s.K, bDeq[j*s.K:(j+1)*s.K])
		}
		want := make([]float32, s.M*s.N)
		MatmulBT(a, bDeq, want, s.M, s.K, s.N)

		got := make([]float32, s.M*s.N)
		MatmulBTW4A8(a, packed, scales, got, s.M, s.K, s.N, s.group)

		e := relL2(got, want)
		t.Logf("shape %+v: W4A8 relL2 = %.4f", s, e)
		if e > 5e-2 {
			t.Errorf("shape %+v: W4A8 vs dequant-f32 relL2 = %.4f, want ≤ 5e-2", s, e)
		}
	}
}
