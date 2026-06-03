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

// TestMatmulBTQ4_matchesDequant: the int4 matmul must equal an f32 matmul against
// the dequantized weights — the kernel and DequantizeRowInt4 share one
// reconstruction formula, so any drift is a packing/indexing bug, independent of
// the (separately bounded) quantization error. Both sides use the SIMD dotF32
// kernel, so the bound is a tight relative L2 (float reassociation, not bits).
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

		// Reference: dequantize b row-by-row, then f32 matmul.
		nGroups, bpr := groupsFor(s.K, group)
		bDeq := make([]float32, s.N*s.K)
		for j := range s.N {
			DequantizeRowInt4(packed[j*bpr:(j+1)*bpr], scales[j*nGroups:(j+1)*nGroups], group, s.K, bDeq[j*s.K:(j+1)*s.K])
		}
		want := make([]float32, s.M*s.N)
		MatmulBT(a, bDeq, want, s.M, s.K, s.N)

		got := make([]float32, s.M*s.N)
		MatmulBTQ4(a, packed, scales, got, s.M, s.K, s.N, group)
		if e := relL2(got, want); e > 1e-5 {
			t.Fatalf("shape %+v: MatmulBTQ4 vs dequant matmul relL2 = %.2e, want ≤ 1e-5", s, e)
		}
	}
}

// matmulBTQ4Scalar is the pre-SIMD reference: a fused scalar unpack+MAC inner
// loop. Kept in the test to benchmark the SIMD kernel's speedup against it.
func matmulBTQ4Scalar(a []float32, bPacked []byte, bScales []float32, dst []float32, M, K, N, group int) {
	nGroups, bpr := groupsFor(K, group)
	for i := range M {
		arow := a[i*K : i*K+K]
		drow := dst[i*N : i*N+N]
		for j := range N {
			prow := bPacked[j*bpr : j*bpr+bpr]
			srow := bScales[j*nGroups : j*nGroups+nGroups]
			var total float32
			for g := range nGroups {
				ks := g * group
				ke := min(ks+group, K)
				var gsum float32
				for k := ks; k < ke; k++ {
					b := prow[k>>1]
					nib := b & 0x0F
					if k&1 == 1 {
						nib = b >> 4
					}
					gsum += arow[k] * float32(int(nib)-8)
				}
				total += gsum * srow[g]
			}
			drow[j] = total
		}
	}
}

// BenchmarkMatmulBTQ4 / _Scalar: the SIMD kernel vs the fused-scalar reference on
// a decode-step shape (M=1). Run: go test ./internal/linalg -bench MatmulBTQ4 -benchmem
func benchInt4(b *testing.B, simd bool) {
	const M, K, N, group = 1, 2048, 2048, 32
	rng := rand.New(rand.NewSource(7))
	a := make([]float32, M*K)
	w := make([]float32, N*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	packed, scales := QuantizeGroupsInt4(w, N, K, group)
	dst := make([]float32, M*N)
	b.ResetTimer()
	for range b.N {
		if simd {
			MatmulBTQ4(a, packed, scales, dst, M, K, N, group)
		} else {
			matmulBTQ4Scalar(a, packed, scales, dst, M, K, N, group)
		}
	}
}

func BenchmarkMatmulBTQ4(b *testing.B)       { benchInt4(b, true) }
func BenchmarkMatmulBTQ4Scalar(b *testing.B) { benchInt4(b, false) }

// matmulBTQ8Scalar is the pre-SIMD int8 reference (scalar widen-in-loop MAC),
// kept to benchmark the SIMD kernel's speedup.
func matmulBTQ8Scalar(a []float32, bQ []int8, bScales []float32, dst []float32, M, K, N int) {
	for i := range M {
		arow := a[i*K : i*K+K]
		drow := dst[i*N : i*N+N]
		for j := range N {
			bq := bQ[j*K : j*K+K]
			var s float32
			for k := range K {
				s += arow[k] * float32(bq[k])
			}
			drow[j] = s * bScales[j]
		}
	}
}

func benchInt8(b *testing.B, simd bool) {
	const M, K, N = 1, 2048, 2048
	rng := rand.New(rand.NewSource(7))
	a := make([]float32, M*K)
	w := make([]float32, N*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	q, scales := QuantizeRowsInt8(w, N, K)
	dst := make([]float32, M*N)
	b.ResetTimer()
	for range b.N {
		if simd {
			MatmulBTQ8(a, q, scales, dst, M, K, N)
		} else {
			matmulBTQ8Scalar(a, q, scales, dst, M, K, N)
		}
	}
}

func BenchmarkMatmulBTQ8(b *testing.B)       { benchInt8(b, true) }
func BenchmarkMatmulBTQ8Scalar(b *testing.B) { benchInt8(b, false) }

// BenchmarkMatmulBTW8A8 measures the full int8×int8 (W8A8) matmul — dynamic
// activation quant + the AVX2 int8 dot — against MatmulBTQ8 (weight-only int8,
// f32-widen + dotFMA) on the same decode-step shape.
func BenchmarkMatmulBTW8A8(b *testing.B) {
	const M, K, N = 1, 2048, 2048
	rng := rand.New(rand.NewSource(7))
	a := make([]float32, M*K)
	w := make([]float32, N*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	q, scales := QuantizeRowsInt8(w, N, K)
	dst := make([]float32, M*N)
	b.ResetTimer()
	for range b.N {
		MatmulBTW8A8(a, q, scales, dst, M, K, N)
	}
}

// TestDotI8_matchesScalar: the SIMD int8 dot (AVX2 on amd64) must be bit-exact
// to the scalar reference — integer arithmetic, so equality is exact, not a
// tolerance. Covers 16-multiples and ragged tails, and the ±127 saturation range.
func TestDotI8_matchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	for _, n := range []int{0, 1, 7, 15, 16, 17, 31, 64, 127, 2048, 2049} {
		a := make([]int8, n)
		b := make([]int8, n)
		for i := range a {
			a[i] = int8(rng.Intn(255) - 127) // [-127,127]
			b[i] = int8(rng.Intn(255) - 127)
		}
		if got, want := dotI8(a, b), dotI8Scalar(a, b); got != want {
			t.Errorf("n=%d: dotI8 = %d, want %d", n, got, want)
		}
	}
	// Extremes: all -127 / +127 (largest |accumulator|).
	for _, n := range []int{16, 2048} {
		a := make([]int8, n)
		b := make([]int8, n)
		for i := range a {
			a[i], b[i] = -127, 127
		}
		if got, want := dotI8(a, b), dotI8Scalar(a, b); got != want {
			t.Errorf("extremes n=%d: dotI8 = %d, want %d", n, got, want)
		}
	}
}

// TestMatmulBTW8A8_closeToF32: full int8×int8 (activations also quantized) is
// lossier than weight-only int8, but on well-behaved (no-outlier) data it should
// still track f32 closely. This is the best-case signal; real activations have
// outlier channels that hurt more (measured separately on the decoder).
func TestMatmulBTW8A8_closeToF32(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	randVec := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	for _, s := range []struct{ M, K, N int }{
		{1, 640, 1024}, {1, 2048, 640}, {4, 256, 300},
	} {
		a := randVec(s.M * s.K)
		b := randVec(s.N * s.K)
		want := make([]float32, s.M*s.N)
		MatmulBT(a, b, want, s.M, s.K, s.N)
		q, scales := QuantizeRowsInt8(b, s.N, s.K)
		got := make([]float32, s.M*s.N)
		MatmulBTW8A8(a, q, scales, got, s.M, s.K, s.N)
		e := relL2(got, want)
		t.Logf("shape %+v: W8A8 relL2 = %.4f", s, e)
		if e > 5e-2 {
			t.Errorf("shape %+v: W8A8 relL2 = %.4f, want ≤ 5e-2", s, e)
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
