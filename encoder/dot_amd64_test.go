//go:build amd64

package encoder

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
