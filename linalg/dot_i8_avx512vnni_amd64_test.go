//go:build amd64

package linalg

import (
	"math/rand/v2"
	"testing"

	"github.com/townsendmerino/aikit/internal/cpufeat"
)

// TestAVX512VNNI_detection sanity-checks that the VNNI probe runs without
// faulting and reports a plausible result — mirrors TestAVX2_detection. It
// does not assert VNNI presence (host-dependent), only that the CPUID/XGETBV
// machinery itself works.
func TestAVX512VNNI_detection(t *testing.T) {
	maxLeaf, _, _, _ := cpufeat.CPUID(0, 0)
	if maxLeaf < 1 {
		t.Fatalf("CPUID leaf 0 reported maxLeaf=%d; probe is broken", maxLeaf)
	}
	t.Logf("cpufeat.HasAVX512VNNI=%v cpufeat.HasAVX2=%v (maxLeaf=%d)", cpufeat.HasAVX512VNNI, cpufeat.HasAVX2, maxLeaf)
}

// TestAVX512VNNI_dotI8AVX512VNNI_matchesScalar directly exercises the raw
// VPDPBUSD kernel (bypassing dotI8's dispatch) against dotI8Scalar, at n
// values that are exact multiples of 64 — the kernel's contract. Skips when
// the host lacks AVX-512 VNNI (the asm would SIGILL).
//
// Weighted toward the sign-correction math specifically (dotI8AVX2 stays
// fully signed and has no such correction; this kernel's whole risk is the
// u8×s8-to-s8×s8 offset trick in dot_i8_avx512vnni_amd64.go) — full [-128,127]
// random data plus the four sign-combination extremes at the largest
// magnitude (-128/-128, -128/127, 127/-128, 127/127), which is where an
// off-by-one in the +128 correction or the ones-vector would first show up.
func TestAVX512VNNI_dotI8AVX512VNNI_matchesScalar(t *testing.T) {
	if !cpufeat.HasAVX512VNNI {
		t.Skip("CPU lacks AVX-512 VNNI; dotI8AVX512VNNI asm path not exercised")
	}
	rng := rand.New(rand.NewPCG(41, 43))
	for _, n := range []int{64, 128, 192, 256, 384, 2048, 2112, 4096, 4160} {
		a := make([]int8, n)
		b := make([]int8, n)
		for i := range a {
			a[i] = int8(rng.IntN(256) - 128) // full [-128,127]
			b[i] = int8(rng.IntN(256) - 128)
		}
		got := dotI8AVX512VNNI(&a[0], &b[0], n)
		want := dotI8Scalar(a, b)
		if got != want {
			t.Errorf("n=%d: dotI8AVX512VNNI = %d, want %d", n, got, want)
		}
	}

	for _, n := range []int{64, 192, 2048} {
		a := make([]int8, n)
		b := make([]int8, n)
		for _, combo := range [][2]int8{{-128, 127}, {127, -128}, {-128, -128}, {127, 127}} {
			for i := range a {
				a[i], b[i] = combo[0], combo[1]
			}
			got := dotI8AVX512VNNI(&a[0], &b[0], n)
			want := dotI8Scalar(a, b)
			if got != want {
				t.Errorf("n=%d combo=%v: dotI8AVX512VNNI = %d, want %d", n, combo, got, want)
			}
		}
	}
}

// TestAVX512VNNI_dotI8_dispatchesThroughEveryTier exercises the full dotI8
// dispatcher (not the raw kernel) at n straddling all three tier boundaries —
// a VNNI-then-AVX2-then-scalar combination (n%64 in (0,64), n%16!=0), a
// VNNI-then-scalar combination (n%64!=0, remainder<16), and pure-VNNI
// (n%64==0) — so a regrouping bug across tiers, not just within one kernel,
// would show up. Runs regardless of cpufeat.HasAVX512VNNI/cpufeat.HasAVX2: on lesser hardware
// it's exercising whichever tiers ARE present, which is the point.
func TestAVX512VNNI_dotI8_dispatchesThroughEveryTier(t *testing.T) {
	rng := rand.New(rand.NewPCG(53, 59))
	// 191 = 2×64 + 63 (VNNI takes 128, AVX2 takes 48, scalar takes 15);
	// 129 = 2×64 + 1  (VNNI takes 128, scalar takes 1, AVX2 tier empty);
	// 256 = 4×64      (pure VNNI, nothing left for AVX2 or scalar).
	for _, n := range []int{0, 1, 15, 63, 65, 129, 191, 192, 193, 256, 4097} {
		a := make([]int8, n)
		b := make([]int8, n)
		for i := range a {
			a[i] = int8(rng.IntN(256) - 128)
			b[i] = int8(rng.IntN(256) - 128)
		}
		if got, want := dotI8(a, b), dotI8Scalar(a, b); got != want {
			t.Errorf("n=%d: dotI8 = %d, want %d", n, got, want)
		}
	}
}
