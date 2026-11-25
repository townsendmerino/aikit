package attn

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// f16Cache returns n halves drawn from N(0, 1) and their exact f32 widening: the f16 kernels must equal the f32 kernels run on the
// widened values, bit for bit, since widening is exact and the kernels then do the same f64 adds in the same order. Every index in
// subnormal gets a half below f16's normal range, alternating 3e-6 and -6e-8 (near the bottom of it); a caller places them inside the
// columns its kernel reads.
func f16Cache(rng *rand.Rand, n int, subnormal ...int) ([]uint16, []float32) {
	f := make([]float32, n)
	for i := range f {
		f[i] = float32(rng.NormFloat64())
	}
	for i, at := range subnormal {
		f[at] = []float32{3e-6, -6e-8}[i%2]
	}
	h := make([]uint16, n)
	linalg.F32ToF16Slice(h, f)
	w := make([]float32, n)
	linalg.F16ToF32Slice(w, h)
	return h, w
}

// mustBeSubnormal fails t unless every index in at widened to a non-zero half below f16's normal range (2^-14).
func mustBeSubnormal(t *testing.T, w []float32, at ...int) {
	t.Helper()
	for _, i := range at {
		if v := math.Abs(float64(w[i])); v == 0 || v >= 0x1p-14 {
			t.Fatalf("the half planted at %d widened to %v, not a subnormal", i, w[i])
		}
	}
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
				// Subnormals in the columns the kernel reads: key 0's first dim, a middle key's second, the first key after the 16-key
				// groups' third (the 4-key kernel's block when N%16 >= 4), and the last key's last dim.
				plants := []int{bOff, bOff + (N/2)*rowStride + 1, bOff + min(N-1, 16*(N/16))*rowStride + 2, bOff + (N-1)*rowStride + K - 1}
				h, w := f16Cache(rng, (N-1)*rowStride+bOff+K, plants...)
				mustBeSubnormal(t, w, plants...)
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
				plants := []int{headOff, headOff + (nKeys/2)*rowStride + 1, headOff + (nKeys-1)*rowStride + hd - 1}
				h, w := f16Cache(rng, (nKeys-1)*rowStride+headOff+hd, plants...)
				mustBeSubnormal(t, w, plants...)
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
