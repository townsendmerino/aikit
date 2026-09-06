//go:build arm64

package linalg

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestSoftmaxContractNEON_bitIdenticalToScalar is the gate for the order-pinned
// softmax. Raw bits, both the distribution and the denominator, over the row
// lengths goinfer actually calls with — including lengths that are NOT multiples
// of 4, which is where the vector body and the scalar tail have to agree about
// which partial each element belongs to.
func TestSoftmaxContractNEON_bitIdenticalToScalar(t *testing.T) {
	rng := rand.New(rand.NewPCG(0x50f, 0x7a))
	lens := []int{1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 128, 130, 512, 1023, 1536, 2048, 3900, 8960}
	for _, n := range lens {
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(rng.NormFloat64() * 8)
		}
		a := make([]float32, n)
		b := make([]float32, n)
		softmaxRowContract(a, src)
		softmaxRowContractNEON(b, src)
		for i := range a {
			if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
				t.Fatalf("n=%d i=%d: scalar %v (%08x) != NEON %v (%08x)",
					n, i, a[i], math.Float32bits(a[i]), b[i], math.Float32bits(b[i]))
			}
		}
	}
}

// TestSoftmaxContractNEON_hostileRows covers the cases that break naive kernels:
// a −Inf entry (masked attention), an all-equal row, a row that underflows
// entirely, and one with a huge dynamic range. The −Inf case is the reason the
// contract clamps: without it the vector path saturates VFCVTZS while Go's
// float→int conversion does something else, and the two disagree on garbage.
func TestSoftmaxContractNEON_hostileRows(t *testing.T) {
	inf := float32(math.Inf(1))
	rows := [][]float32{
		{0, 0, 0, 0, 0, 0, 0, 0},
		{-inf, 1, 2, 3},
		{-inf, -inf, -inf, 5, 5, 5, 5, 5},
		{1e30, -1e30, 0, 1, -1, 2, -2, 3},
		{-200, -300, -400, -500},
		{88, -87, 0, 1},
	}
	for ri, src := range rows {
		a := make([]float32, len(src))
		b := make([]float32, len(src))
		softmaxRowContract(a, src)
		softmaxRowContractNEON(b, src)
		for i := range a {
			if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
				t.Fatalf("row %d (%v) i=%d: scalar %08x != NEON %08x",
					ri, src, i, math.Float32bits(a[i]), math.Float32bits(b[i]))
			}
		}
		var s float64
		for _, v := range a {
			s += float64(v)
		}
		if math.Abs(s-1) > 1e-4 && !math.IsNaN(s) {
			t.Errorf("row %d sums to %v, not ~1", ri, s)
		}
	}
}
