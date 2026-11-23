package linalg

import (
	"math"
	"math/rand/v2"
	"testing"
)

// f16Cache returns n halves drawn from N(0, 1) and their exact f32 widening: the f16 kernels must equal the f32 kernels run on the
// widened values, bit for bit, since widening is exact and the kernels then do the same f64 adds in the same order.
func f16Cache(rng *rand.Rand, n int) ([]uint16, []float32) {
	f := make([]float32, n)
	for i := range f {
		f[i] = float32(rng.NormFloat64())
	}
	f[0], f[n/2] = 3e-6, -6e-8 // a subnormal half and one near the bottom of its range
	h := make([]uint16, n)
	F32ToF16Slice(h, f)
	w := make([]float32, n)
	F16ToF32Slice(w, h)
	return h, w
}

// TestMatmulQKAcc64F16_matchesF32OnWidened pins MatmulQKAcc64F16 to MatmulQKAcc64 over the widened cache: head dims 64, 96 and 128,
// key counts with and without a 16-key remainder, a key stride wider than the head (a head slice of a kvDim row), and M > 1.
func TestMatmulQKAcc64F16_matchesF32OnWidened(t *testing.T) {
	rng := rand.New(rand.NewPCG(37, 41))
	for _, K := range []int{64, 96, 128} {
		for _, N := range []int{1, 15, 16, 61, 3900} {
			for _, M := range []int{1, 3} {
				nKV := 3
				rowStride := nKV * K
				bOff := 1 * K
				h, w := f16Cache(rng, (N-1)*rowStride+bOff+K)
				a := make([]float32, M*K)
				for i := range a {
					a[i] = float32(rng.NormFloat64())
				}
				got, want := make([]float32, M*N), make([]float32, M*N)
				MatmulQKAcc64F16(a, h, got, M, K, N, bOff, rowStride)
				MatmulQKAcc64(a, w, want, M, K, N, bOff, rowStride)
				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						t.Fatalf("K=%d N=%d M=%d: score %d = %08x, f32 kernel on the widened cache %08x", K, N, M, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
					}
				}
			}
		}
	}
}

// TestMatmulAVAcc64F16_matchesF32OnWidened pins MatmulAVAcc64F16 to MatmulAVAcc64 over the widened cache, at head dims that leave no
// remainder, a 16-dim block and a ragged tail.
func TestMatmulAVAcc64F16_matchesF32OnWidened(t *testing.T) {
	rng := rand.New(rand.NewPCG(43, 47))
	for _, hd := range []int{64, 80, 96, 100, 128} {
		for _, nKeys := range []int{1, 7, 3900} {
			for _, M := range []int{1, 2} {
				rowStride := 2*hd + 4
				headOff := hd + 4
				h, w := f16Cache(rng, (nKeys-1)*rowStride+headOff+hd)
				scores := make([]float32, M*nKeys)
				for i := range scores {
					scores[i] = rng.Float32() / float32(nKeys)
				}
				got, want := make([]float32, M*hd), make([]float32, M*hd)
				acc := make([]float64, hd)
				MatmulAVAcc64F16(scores, h, got, acc, M, nKeys, hd, headOff, rowStride)
				MatmulAVAcc64(scores, w, want, acc, M, nKeys, hd, headOff, rowStride)
				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						t.Fatalf("hd=%d nKeys=%d M=%d: out %d = %08x, f32 kernel on the widened cache %08x", hd, nKeys, M, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
					}
				}
			}
		}
	}
}
