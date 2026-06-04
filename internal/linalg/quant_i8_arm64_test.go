//go:build arm64

package linalg

import (
	"math/rand"
	"testing"
)

// TestDotI8SDOT_matchesScalar validates the ARMv8.2 DotProd kernel directly
// (bit-exact vs the scalar reference — integer arithmetic, so equality is exact).
// It only runs when this CPU advertises DotProd: the SDOT opcode would trap on a
// core without it, and dotI8 would never dispatch here anyway. Under
// qemu-aarch64 with a DotProd-capable -cpu (e.g. `max`), hasDotProd is true and
// this covers the WORD-encoded SDOT block + 64-wide main loop + 16-wide tail +
// reduction. TestDotI8_matchesScalar already covers whichever kernel dotI8
// actually selects on the host.
func TestDotI8SDOT_matchesScalar(t *testing.T) {
	if !hasDotProd {
		t.Skip("CPU has no DotProd (ASIMDDP); SDOT kernel not exercisable here")
	}
	rng := rand.New(rand.NewSource(11))
	// 16, 48, 80 are multiples of 16 that are not multiples of 64 — they force
	// the tail loop after 0, 0, and 1 main-loop blocks respectively. 2048/2064
	// stress the long path; 1024 is exactly 16 blocks (no tail).
	for _, n := range []int{16, 32, 48, 64, 80, 1024, 2048, 2064} {
		a := make([]int8, n)
		b := make([]int8, n)
		for i := range a {
			a[i] = int8(rng.Intn(255) - 127)
			b[i] = int8(rng.Intn(255) - 127)
		}
		if got, want := dotI8SDOT(&a[0], &b[0], n), dotI8Scalar(a, b); got != want {
			t.Errorf("n=%d: dotI8SDOT = %d, want %d", n, got, want)
		}
	}
	// Saturation extremes: all -127 / +127 (largest |accumulator| per element).
	for _, n := range []int{64, 2048} {
		a := make([]int8, n)
		b := make([]int8, n)
		for i := range a {
			a[i], b[i] = -127, 127
		}
		if got, want := dotI8SDOT(&a[0], &b[0], n), dotI8Scalar(a, b); got != want {
			t.Errorf("extremes n=%d: dotI8SDOT = %d, want %d", n, got, want)
		}
	}
}
