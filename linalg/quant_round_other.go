//go:build !amd64

package linalg

import "math"

// roundInt4 rounds x to the nearest integer, ties away from zero, and converts it. math.Round is a single instruction
// on arm64, where the amd64 form of this function is no faster.
func roundInt4(x float32) int { return int(math.Round(float64(x))) }
