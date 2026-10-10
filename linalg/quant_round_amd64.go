//go:build amd64

package linalg

import "math"

// roundInt4 rounds x to the nearest integer, ties away from zero, and converts it: the value of
// int(math.Round(float64(x))) for every float32, NaN and the infinities included
// (TestRoundInt4_matchesMathRound). Adding a half of x's sign and truncating replaces math.Round, which is a function
// call on amd64; the sum is exact wherever it changes the result, because a float32 widened to float64 has 29 spare
// low bits.
func roundInt4(x float32) int {
	f := float64(x)
	return int(f + math.Copysign(0.5, f))
}
