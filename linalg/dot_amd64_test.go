//go:build amd64

package linalg

import (
	"math/rand/v2"
	"testing"
)

// TestAVX2_dotFMA_matchesGeneric directly compares the AVX2 asm kernel
// against the scalar generic kernel on the same data, independent of
// runtime dispatch. Skips when the host lacks AVX2 (the asm would
// SIGILL). On an AVX2 box this is the real proof the asm is correct:
// it hits the 32-wide body, the 8-wide tail, and the scalar tail.
func TestAVX2_dotFMA_matchesGeneric(t *testing.T) {
	if !hasAVX2 {
		t.Skip("CPU lacks AVX2/FMA; dotFMA asm path not exercised")
	}
	// Cover multiples of 4 (the only n callers pass) that exercise all
	// three code regions: <8 (scalar only), 8..31 (8-wide + scalar),
	// ≥32 (32-wide + remainders). dotF32 strips the %4 tail, so n is
	// always a multiple of 4 in production — but test odd tails too to
	// prove the scalar-tail arithmetic.
	cases := []int{0, 1, 3, 4, 7, 8, 12, 31, 32, 36, 64, 100, 768, 769, 3072, 3075}
	rng := rand.New(rand.NewPCG(17, 19))
	for _, n := range cases {
		a := make([]float32, n)
		b := make([]float32, n)
		for i := 0; i < n; i++ {
			a[i] = float32(rng.NormFloat64() * 0.1)
			b[i] = float32(rng.NormFloat64() * 0.1)
		}
		var asm float32
		if n > 0 {
			asm = dotFMA(&a[0], &b[0], n)
		}
		// dotGeneric works in n4*4 strides; feed it the same element
		// count by passing n/4 and adding the %4 scalar tail by hand so
		// the reference covers the exact same elements as dotFMA.
		var ref float32
		if n >= 4 {
			ref = dotGeneric(&a[0], &b[0], n/4)
		}
		for i := (n / 4) * 4; i < n; i++ {
			ref += a[i] * b[i]
		}
		tol := float32(1e-4)*absF32(ref) + 1e-6
		if absF32(asm-ref) > tol {
			t.Errorf("n=%d: asm=%v generic=%v (tol %v)", n, asm, ref, tol)
		}
	}
}

// TestAVX2_blockedKernels_oddN4 directly exercises the register-blocked AVX2
// kernels Dot8x4/Dot4x4 (the path the blocked GEMM uses) against a scalar
// reference, over ODD and EVEN n4. The dotFMA4/dotFMA8 loop consumes 8 floats
// (two 4-groups) per YMM iteration; a trailing single 4-group (n4 odd, i.e.
// n%8==4) must accumulate WITHOUT zeroing the YMM accumulators' upper lanes —
// the regression fixed in v1.7.3 (the XMM/VEX.128 tail FMA zeroed lanes 4..7,
// dropping the loop's partials). Pre-fix this fails for every odd n4; the
// single-row dotFMA (TestAVX2_dotFMA_matchesGeneric) was unaffected and hid it.
func TestAVX2_blockedKernels_oddN4(t *testing.T) {
	if !hasAVX2 {
		t.Skip("CPU lacks AVX2/FMA; blocked asm path not exercised")
	}
	rng := rand.New(rand.NewPCG(7, 11))
	randVec := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64() * 0.1)
		}
		return v
	}
	// Odd n4 (1,3,5,75) is the regression; even (4,8,160) was always correct.
	for _, n4 := range []int{1, 3, 4, 5, 8, 75, 160} {
		n := n4 * 4
		a := randVec(n)
		b := make([][]float32, 8)
		for r := range b {
			b[r] = randVec(n)
		}
		ref := func(bv []float32) float32 {
			var s float32
			for i := 0; i < n; i++ {
				s += a[i] * bv[i]
			}
			return s
		}
		tol := func(want float32) float32 { return 1e-3*absF32(want) + 1e-4 }

		var s8 [32]float32
		Dot8x4(&a[0], &b[0][0], &b[1][0], &b[2][0], &b[3][0], &b[4][0], &b[5][0], &b[6][0], &b[7][0], n4, &s8)
		for r := 0; r < 8; r++ {
			want := ref(b[r])
			if got := s8[r*4]; absF32(got-want) > tol(want) {
				t.Errorf("Dot8x4 n4=%d row=%d: got=%v want=%v (Δ%v)", n4, r, got, want, absF32(got-want))
			}
		}

		var s4 [16]float32
		Dot4x4(&a[0], &b[0][0], &b[1][0], &b[2][0], &b[3][0], n4, &s4)
		for r := 0; r < 4; r++ {
			want := ref(b[r])
			if got := s4[r*4]; absF32(got-want) > tol(want) {
				t.Errorf("Dot4x4 n4=%d row=%d: got=%v want=%v (Δ%v)", n4, r, got, want, absF32(got-want))
			}
		}
	}
}

// TestAVX2_detection sanity-checks that feature detection runs without
// faulting and reports a plausible result (CPUID leaf 0 must be ≥1 on
// any real amd64). It does not assert AVX2 presence — that's host-
// dependent — only that the probe machinery itself works.
func TestAVX2_detection(t *testing.T) {
	maxLeaf, _, _, _ := cpuid(0, 0)
	if maxLeaf < 1 {
		t.Fatalf("CPUID leaf 0 reported maxLeaf=%d; probe is broken", maxLeaf)
	}
	t.Logf("hasAVX2=%v (maxLeaf=%d)", hasAVX2, maxLeaf)
}
