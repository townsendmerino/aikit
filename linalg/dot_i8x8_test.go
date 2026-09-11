package linalg

import (
	"math/rand/v2"
	"testing"
)

// TestDotI8x8_matchesScalar gates DotI8x8 against the SCALAR reference, not
// against another SIMD kernel — the same choice dot_i8x8_amd64.s's own comment
// makes, and for the same reason: integer arithmetic is associative and
// int8×int8→int32 cannot overflow for any length this library sees (|Σ| ≤
// n·127², n would have to exceed 133,000), so the blocked/shared-widening
// kernel must equal the plain scalar loop EXACTLY. Runs on every arch —
// amd64's AVX2 path and the portable fallback both owe this the same answer.
func TestDotI8x8_matchesScalar(t *testing.T) {
	rng := rand.New(rand.NewPCG(23, 29))
	for _, n := range []int{1, 8, 15, 16, 32, 48, 64, 128, 768, 3072} {
		a := make([]int8, n)
		for i := range a {
			a[i] = int8(rng.IntN(256) - 128) // full int8 range, -128 included
		}
		cols := make([][]int8, 8)
		for c := range cols {
			cols[c] = make([]int8, n)
			for i := range cols[c] {
				cols[c][i] = int8(rng.IntN(256) - 128)
			}
		}
		got := DotI8x8(a, cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7])
		for c := range cols {
			var want int32
			for i := range a {
				want += int32(a[i]) * int32(cols[c][i])
			}
			if got[c] != want {
				t.Errorf("n=%d col%d: DotI8x8 %d, scalar %d", n, c, got[c], want)
			}
		}
	}
}

// TestDotI8x8_columnsAreIndependent guards the failure a shared-query kernel
// invites: an accumulator wired to the wrong column, or one column's product
// landing in another's. Every column gets a distinct constant, so a crossed
// wire changes the value rather than producing a plausible-looking one.
func TestDotI8x8_columnsAreIndependent(t *testing.T) {
	const n = 64
	a := make([]int8, n)
	for i := range a {
		a[i] = 1
	}
	cols := make([][]int8, 8)
	for c := range cols {
		cols[c] = make([]int8, n)
		for i := range cols[c] {
			cols[c][i] = int8(c + 1)
		}
	}
	got := DotI8x8(a, cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7])
	for c := range cols {
		if want := int32(n * (c + 1)); got[c] != want {
			t.Errorf("col%d = %d, want %d — accumulators are crossed", c, got[c], want)
		}
	}
}

// TestDotI8x8_lengthMismatchPanics mirrors DotI8's own guard.
func TestDotI8x8_lengthMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected a panic on mismatched lengths")
		}
	}()
	a := make([]int8, 16)
	short := make([]int8, 15)
	full := make([]int8, 16)
	DotI8x8(a, full, full, full, full, full, full, full, short)
}
