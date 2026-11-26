package bitvec

import (
	"math/bits"
	"math/rand"
	"testing"
)

// TestPackSignBitsRow_bitOrder pins the packing convention. It is not
// arbitrary-but-consistent from the caller's side: ann.FlatBinary persists
// codes, so a change here silently invalidates every stored index.
func TestPackSignBitsRow_bitOrder(t *testing.T) {
	v := make([]float32, 130)
	for i := range v {
		v[i] = -1
	}
	v[0], v[63], v[64], v[129] = 1, 1, 1, 1
	dst := make([]uint64, PackedWords(len(v)))
	// Pre-dirty every word: PackSignBitsRow promises to fully write dst, so a
	// reused buffer must not leak stale bits (notably in the ragged last word).
	for i := range dst {
		dst[i] = ^uint64(0)
	}
	PackSignBitsRow(dst, v)

	want := []uint64{1 | 1<<63, 1, 1 << 1}
	for i := range want {
		if dst[i] != want[i] {
			t.Errorf("word %d = %#016x, want %#016x", i, dst[i], want[i])
		}
	}
	// Zero is a positive bit, and the last word's 126 unused bits stay clear.
	v[2] = 0
	PackSignBitsRow(dst, v)
	if dst[0]&(1<<2) == 0 {
		t.Error("zero packed as a negative bit; it must pack as 1")
	}
	if dst[2]>>2 != 0 {
		t.Errorf("trailing bits of the ragged last word not cleared: %#016x", dst[2])
	}
}

// TestHammingRows_matchesScalar gates the amd64 POPCNTQ kernel against a plain
// popcount loop written independently of hammingRowsGeneric — so a bug in the
// shared reference cannot hide a bug in the kernel.
//
// The word counts span the kernel's unrolled-by-4 structure deliberately: 1, 2
// and 3 exercise the tail with an empty main loop, 4 and 8 the main loop with
// an empty tail, and 12 (dim 768, the real encoder shape) and 13 both at once.
func TestHammingRows_matchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(38))
	for _, words := range []int{1, 2, 3, 4, 5, 8, 12, 13, 16} {
		for _, n := range []int{1, 7, 64} {
			q := make([]uint64, words)
			codes := make([]uint64, n*words)
			for i := range q {
				q[i] = rng.Uint64()
			}
			for i := range codes {
				codes[i] = rng.Uint64()
			}
			got := make([]uint16, n)
			HammingRows(q, codes, words, n, got)
			for i := range n {
				want := 0
				for j := range words {
					x := codes[i*words+j] ^ q[j]
					for ; x != 0; x &= x - 1 {
						want++
					}
				}
				if int(got[i]) != want {
					t.Fatalf("words=%d n=%d row %d: got %d, want %d", words, n, i, got[i], want)
				}
			}
		}
	}
}

// TestHammingRows_writesEveryDestination catches a kernel that skips rows —
// the tail-handling bug an all-equal-distance input would hide. Every row here
// has a distinct distance, and dst starts poisoned.
func TestHammingRows_writesEveryDestination(t *testing.T) {
	const words, n = 12, 64
	q := make([]uint64, words)
	codes := make([]uint64, n*words)
	for i := range n {
		// Row i differs from q (all zero) in exactly i bits.
		for b := range i {
			codes[i*words+b/64] |= 1 << uint(b%64)
		}
	}
	dst := make([]uint16, n)
	for i := range dst {
		dst[i] = 0xBEEF
	}
	HammingRows(q, codes, words, n, dst)
	for i := range n {
		if int(dst[i]) != i {
			t.Fatalf("row %d = %d, want %d", i, dst[i], i)
		}
	}
}

// TestHammingRows_angleProxy is the property the prefilter actually depends on:
// Hamming distance tracks the ANGLE between the underlying vectors
// (E[H]/dim = θ/π). Without this, ordering by ascending distance would be
// ordering by nothing in particular.
//
// It checks monotonicity in the mean over random pairs at each of several
// planted angles, not per-pair — per-pair the estimator has a standard
// deviation of ~√(dim)/dim, which at dim 768 is ~3.6% of the range and would
// make a strict pairwise test flaky by construction.
func TestHammingRows_angleProxy(t *testing.T) {
	const dim, pairs = 768, 200
	rng := rand.New(rand.NewSource(1038))
	words := PackedWords(dim)

	// mean measured distance at a given angle
	meanAt := func(theta float64) float64 {
		total := 0
		for range pairs {
			a := make([]float32, dim)
			for i := range a {
				a[i] = float32(rng.NormFloat64())
			}
			// b = a rotated by theta toward an independent direction.
			b := make([]float32, dim)
			cos, sin := float32(cosApprox(theta)), float32(sinApprox(theta))
			for i := range b {
				b[i] = cos*a[i] + sin*float32(rng.NormFloat64())
			}
			qa := make([]uint64, words)
			qb := make([]uint64, words)
			PackSignBitsRow(qa, a)
			PackSignBitsRow(qb, b)
			for j := range words {
				total += bits.OnesCount64(qa[j] ^ qb[j])
			}
		}
		return float64(total) / float64(pairs)
	}

	prev := -1.0
	for _, theta := range []float64{0.1, 0.3, 0.6, 1.0, 1.4} {
		m := meanAt(theta)
		t.Logf("theta=%.1f rad: mean Hamming %.1f / %d (θ/π·dim = %.1f)", theta, m, dim, theta/3.141592653589793*dim)
		if m <= prev {
			t.Errorf("mean Hamming distance did not increase with angle: %.1f at theta=%.1f, %.1f before", m, theta, prev)
		}
		prev = m
	}
}

// cosApprox/sinApprox keep this test free of a math import whose float64
// transcendentals are irrelevant to what is being checked.
func cosApprox(x float64) float64 {
	s := 1.0
	t := 1.0
	for n := 1; n <= 12; n++ {
		t *= -x * x / float64((2*n-1)*(2*n))
		s += t
	}
	return s
}

func sinApprox(x float64) float64 {
	s := x
	t := x
	for n := 1; n <= 12; n++ {
		t *= -x * x / float64((2*n)*(2*n+1))
		s += t
	}
	return s
}
