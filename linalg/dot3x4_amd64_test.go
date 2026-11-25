//go:build amd64

package linalg

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/townsendmerino/aikit/internal/cpufeat"
)

// TestDot3x4_matchesDotFMA8 is the same bit-identity contract for the 12-accumulator
// kernel: three rows against four columns must equal three separate dotFMA8 calls,
// exactly. Same odd-n4 sweep, for the same reason — that is where the 4-element tail
// runs, and where the 1x8 kernel's historical VEX-zeroing bug lived.
func TestDot3x4_matchesDotFMA8(t *testing.T) {
	if !cpufeat.HasAVX2 {
		t.Skip("no AVX2+FMA on this CPU")
	}
	rng := rand.New(rand.NewPCG(13, 17))
	rv := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = rng.Float32()*2 - 1
		}
		return v
	}
	for _, n4 := range []int{1, 2, 3, 5, 7, 8, 15, 16, 63, 96, 191, 192} {
		n := n4 * 4
		a := [3][]float32{rv(n), rv(n), rv(n)}
		b := [4][]float32{rv(n), rv(n), rv(n), rv(n)}

		var got [12]float32
		dotFMA3x4(&a[0][0], &a[1][0], &a[2][0], &b[0][0], &b[1][0], &b[2][0], &b[3][0], n, &got)

		for r := range 3 {
			var ref [8]float32
			dotFMA8(&a[r][0], &b[0][0], &b[1][0], &b[2][0], &b[3][0], &b[0][0], &b[1][0], &b[2][0], &b[3][0], n, &ref)
			for j := range 4 {
				if got[r*4+j] != ref[j] {
					t.Errorf("n4=%d row%d col%d: 3x4 gave %v (%08x), dotFMA8 gave %v (%08x) — not bit-identical",
						n4, r, j, got[r*4+j], math.Float32bits(got[r*4+j]), ref[j], math.Float32bits(ref[j]))
				}
			}
		}
	}
}
