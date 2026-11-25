// Package shape holds the overflow-safe length checks every linalg kernel package runs at its public entry, before any
// parallel fan-out: a shape violation must panic on the caller's goroutine, where recover can catch it, not inside a
// worker. The products (M·K, N·K, ...) are overflow-checked, so a hostile shape whose product wraps cannot slip past a
// length compare. The panic messages keep the "linalg:" prefix whichever package calls them.
package shape

import "fmt"

// Mul returns a*b for non-negative a, b, or -1 if either is negative or the product overflows int. RequireLen and
// RequireExactLen report the -1 sentinel as an overflow.
func Mul(a, b int) int {
	if a < 0 || b < 0 {
		return -1
	}
	if a == 0 || b == 0 {
		return 0
	}
	p := a * b
	if p/a != b { // overflow
		return -1
	}
	return p
}

// RequireLen panics if have < need, or if need is the -1 overflow sentinel. For matmul operands, which only require
// "at least" the shape's worth.
func RequireLen(kernel, operand string, have, need int) {
	if need < 0 {
		panic(fmt.Sprintf("linalg: %s dimension product overflows int (%s)", kernel, operand))
	}
	if have < need {
		panic(fmt.Sprintf("linalg: %s: %s length %d < required %d", kernel, operand, have, need))
	}
}

// RequireExactLen is RequireLen for constructors that take exactly-sized pre-quantized slices: a wrong length either
// way is a caller bug.
func RequireExactLen(kernel, operand string, have, need int) {
	if need < 0 {
		panic(fmt.Sprintf("linalg: %s dimension product overflows int (%s)", kernel, operand))
	}
	if have != need {
		panic(fmt.Sprintf("linalg: %s: %s length %d != required %d", kernel, operand, have, need))
	}
}
