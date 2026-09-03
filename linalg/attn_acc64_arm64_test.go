//go:build arm64

package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// The NEON kernels against the Go kernels they replace, called directly so no
// dispatch decision can hide a defect, over adversarial data: values spanning
// ~60 binades with random signs, so every accumulator step is a rounding that a
// different order or a different fusion would change. Raw bits, not ==, so a
// NaN could not pass by accident either.

func adversarialF32(rng *rand.Rand, n int) []float32 {
	r := make([]float32, n)
	for i := range r {
		mag := math.Ldexp(1+rng.Float64(), rng.Intn(60)-30)
		if rng.Intn(2) == 0 {
			mag = -mag
		}
		r[i] = float32(mag)
	}
	return r
}

// avGoBlock is the Go definition of one 32-dim block, written as the
// reference loop-nest (dims-outer, keys-inner, one f64 chain per dim).
func avGoBlock(scores, vals []float32, nKeys, rowStride, off int, out []float32) {
	for d := range 32 {
		var acc float64
		for s := range nKeys {
			acc += float64(scores[s]) * float64(vals[off+s*rowStride+d])
		}
		out[d] = float32(acc)
	}
}

func TestAVAcc64NEON32_matchesGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5109))
	for _, nKeys := range []int{1, 2, 3, 7, 16, 17, 130, 131, 1000, 8192} {
		for _, rowStride := range []int{32, 64, 256, 300} {
			for _, off := range []int{0, 32, rowStride - 32} {
				if off+32 > rowStride || off < 0 {
					continue
				}
				scores := adversarialF32(rng, nKeys)
				vals := adversarialF32(rng, nKeys*rowStride)
				want := make([]float32, 32)
				avGoBlock(scores, vals, nKeys, rowStride, off, want)
				got := make([]float32, 32)
				for i := range got {
					got[i] = float32(math.NaN())
				}
				avAcc64NEON32(&scores[0], &vals[off], nKeys, rowStride*4, &got[0])
				for d := range 32 {
					if math.Float32bits(got[d]) != math.Float32bits(want[d]) {
						t.Fatalf("nKeys=%d rowStride=%d off=%d dim %d: NEON %v (%08x) Go %v (%08x)",
							nKeys, rowStride, off, d, got[d], math.Float32bits(got[d]), want[d], math.Float32bits(want[d]))
					}
				}
			}
		}
	}
}

// qkGoBlock is the Go definition of nBlocks×16 keys: one f64 chain per key in
// ascending d.
func qkGoBlock(q, rows []float32, rowStride, K, nBlocks int, out []float32) {
	for j := range nBlocks * 16 {
		var acc float64
		for d := range K {
			acc += float64(q[d]) * float64(rows[j*rowStride+d])
		}
		out[j] = float32(acc)
	}
}

func TestQKAcc64NEON16_matchesGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x510A))
	for _, K := range []int{4, 8, 12, 64, 80, 96, 128, 256} {
		for _, nBlocks := range []int{1, 2, 3, 9, 512} {
			for _, rowStride := range []int{K, K + 4, 2 * K, 3*K + 8} {
				q := adversarialF32(rng, K)
				rows := adversarialF32(rng, nBlocks*16*rowStride)
				want := make([]float32, nBlocks*16)
				qkGoBlock(q, rows, rowStride, K, nBlocks, want)
				got := make([]float32, nBlocks*16)
				for i := range got {
					got[i] = float32(math.NaN())
				}
				qkAcc64NEON16(&q[0], &rows[0], rowStride*4, K/4, nBlocks, &got[0])
				for j := range want {
					if math.Float32bits(got[j]) != math.Float32bits(want[j]) {
						t.Fatalf("K=%d nBlocks=%d rowStride=%d key %d: NEON %v (%08x) Go %v (%08x)",
							K, nBlocks, rowStride, j, got[j], math.Float32bits(got[j]), want[j], math.Float32bits(want[j]))
					}
				}
			}
		}
	}
}

// The comparison must be able to fail: a reference with one term dropped from
// the LAST key (AV) / the LAST dim (QK) must differ from the kernel on
// adversarial data. Guards against a vacuous oracle.
func TestAttnAcc64NEON_mutationDetected(t *testing.T) {
	rng := rand.New(rand.NewSource(0x510B))
	const nKeys, rowStride, K = 37, 64, 128
	scores := adversarialF32(rng, nKeys)
	vals := adversarialF32(rng, nKeys*rowStride)
	got := make([]float32, 32)
	avAcc64NEON32(&scores[0], &vals[0], nKeys, rowStride*4, &got[0])
	mut := make([]float32, 32)
	avGoBlock(scores, vals, nKeys-1, rowStride, 0, mut)
	same := 0
	for d := range 32 {
		if math.Float32bits(got[d]) == math.Float32bits(mut[d]) {
			same++
		}
	}
	if same == 32 {
		t.Fatal("AV: dropping the last key changed no output bits — the oracle is vacuous")
	}

	q := adversarialF32(rng, K)
	rows := adversarialF32(rng, 16*rowStride+K)
	qgot := make([]float32, 16)
	qkAcc64NEON16(&q[0], &rows[0], rowStride*4, K/4, 1, &qgot[0])
	qmut := make([]float32, 16)
	qkGoBlock(q, rows, rowStride, K-1, 1, qmut)
	same = 0
	for j := range 16 {
		if math.Float32bits(qgot[j]) == math.Float32bits(qmut[j]) {
			same++
		}
	}
	if same == 16 {
		t.Fatal("QK: dropping the last dim changed no output bits — the oracle is vacuous")
	}
}

// End to end through the public kernels on shapes that mix NEON blocks with the Go
// remainders: hd%32 dims, N%16 keys, K%4 (QK takes the Go path entirely), M>1.
func TestAttnAcc64_publicKernelsMixedShapes(t *testing.T) {
	rng := rand.New(rand.NewSource(0x510C))
	for _, sh := range []struct{ M, nKeys, hd, nKV int }{
		{1, 1, 32, 1}, {1, 15, 32, 1}, {1, 16, 32, 1}, {1, 17, 48, 2}, {2, 33, 80, 2},
		{3, 100, 96, 1}, {1, 257, 128, 2}, {2, 40, 160, 1}, {1, 31, 256, 1}, {1, 50, 20, 2},
		{1, 70, 12, 3}, {4, 129, 128, 4},
	} {
		kvDim := sh.nKV * sh.hd
		for kvh := range sh.nKV {
			// AV
			scores := adversarialF32(rng, sh.M*sh.nKeys)
			vals := adversarialF32(rng, sh.nKeys*kvDim)
			want := make([]float32, sh.M*sh.hd)
			MatmulBTAcc64Strided(scores, vals, want, sh.M, sh.nKeys, sh.hd, kvh*sh.hd, 1, kvDim)
			got := make([]float32, sh.M*sh.hd)
			MatmulAVAcc64(scores, vals, got, make([]float64, sh.hd), sh.M, sh.nKeys, sh.hd, kvh*sh.hd, kvDim)
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("AV %+v kvh=%d idx %d: %08x vs %08x", sh, kvh, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
				}
			}
			// QK
			q := adversarialF32(rng, sh.M*sh.hd)
			keys := adversarialF32(rng, sh.nKeys*kvDim)
			wantQ := make([]float32, sh.M*sh.nKeys)
			MatmulBTAcc64Strided(q, keys, wantQ, sh.M, sh.hd, sh.nKeys, kvh*sh.hd, kvDim, 1)
			gotQ := make([]float32, sh.M*sh.nKeys)
			MatmulQKAcc64(q, keys, gotQ, sh.M, sh.hd, sh.nKeys, kvh*sh.hd, kvDim)
			for i := range wantQ {
				if math.Float32bits(gotQ[i]) != math.Float32bits(wantQ[i]) {
					t.Fatalf("QK %+v kvh=%d idx %d: %08x vs %08x", sh, kvh, i, math.Float32bits(gotQ[i]), math.Float32bits(wantQ[i]))
				}
			}
		}
	}
}
