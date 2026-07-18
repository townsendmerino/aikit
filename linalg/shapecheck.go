package linalg

import "fmt"

// Public matmul + WeightMat shape validation (code review M2).
//
// The parallel matmuls fan work out across goroutines (parallelSpawnCols). A
// shape violation reached inside a worker goroutine panics THERE — and a panic
// on a spawned goroutine is unrecoverable even by a caller wrapping the call in
// recover(), so a crafted/mismatched checkpoint could hard-kill the process
// instead of surfacing a catchable error. These helpers do the O(1) shape check
// at the public entry, BEFORE any fan-out, so a violation panics on the caller's
// goroutine with a clear "linalg:" message. The cost is a handful of comparisons
// against the O(M·N·K) matmul — negligible, and always on (unlike the richer
// -tags aikit_checks contract asserts, which stay compiled out in production).
//
// The products (M·K, N·K, …) are overflow-checked: a hostile shape whose product
// wraps int negative-and-small must not slip past a length compare.

// mul returns a*b for non-negative a, b, or -1 if either is negative or the
// product overflows int. The -1 sentinel is reported as an overflow by the
// checkers below.
func mul(a, b int) int {
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

// requireLen panics if have < need, or if need is the -1 overflow sentinel.
// For matmul operands, which only require "at least" the shape's worth.
func requireLen(kernel, operand string, have, need int) {
	if need < 0 {
		panic(fmt.Sprintf("linalg: %s dimension product overflows int (%s)", kernel, operand))
	}
	if have < need {
		panic(fmt.Sprintf("linalg: %s: %s length %d < required %d", kernel, operand, have, need))
	}
}

// requireExactLen is requireLen for the Wrap*/Quantize* constructors, which
// take exactly-sized pre-quantized slices — a wrong length (either way) is a
// caller bug the constructors have always rejected, now overflow-safe.
func requireExactLen(kernel, operand string, have, need int) {
	if need < 0 {
		panic(fmt.Sprintf("linalg: %s dimension product overflows int (%s)", kernel, operand))
	}
	if have != need {
		panic(fmt.Sprintf("linalg: %s: %s length %d != required %d", kernel, operand, have, need))
	}
}

// checkMatmulBT validates the f32 dst[M,N] = a[M,K]·b[N,K]ᵀ contract before the
// parallel fan-out.
func checkMatmulBT(kernel string, aLen, bLen, dstLen, M, K, N int) {
	if M < 0 || K < 0 || N < 0 {
		panic(fmt.Sprintf("linalg: %s negative dim (M=%d K=%d N=%d)", kernel, M, K, N))
	}
	requireLen(kernel, "a", aLen, mul(M, K))
	requireLen(kernel, "b", bLen, mul(N, K))
	requireLen(kernel, "dst", dstLen, mul(M, N))
}

// checkMatmulQ8 validates the int8-weight dst[M,N] contract (bQ is N×K int8, one
// f32 scale per output row) before the fan-out — shared by MatmulBTQ8, W8A8, and
// the W8A8 Into/Batch variants.
func checkMatmulQ8(kernel string, aLen, bQLen, bScalesLen, dstLen, M, K, N int) {
	if M < 0 || K < 0 || N < 0 {
		panic(fmt.Sprintf("linalg: %s negative dim (M=%d K=%d N=%d)", kernel, M, K, N))
	}
	requireLen(kernel, "a", aLen, mul(M, K))
	requireLen(kernel, "bQ", bQLen, mul(N, K))
	requireLen(kernel, "bScales", bScalesLen, N)
	requireLen(kernel, "dst", dstLen, mul(M, N))
}

// checkMatmulW4A8 validates the group-int4-weight dst[M,N] contract (w4 is
// N×⌈K/2⌉ packed nibbles, wScales is N×⌈K/group⌉) before the fan-out.
func checkMatmulW4A8(kernel string, aLen, w4Len, wScalesLen, dstLen, M, K, N, group int) {
	if M < 0 || K < 0 || N < 0 {
		panic(fmt.Sprintf("linalg: %s negative dim (M=%d K=%d N=%d)", kernel, M, K, N))
	}
	if group <= 0 {
		panic(fmt.Sprintf("linalg: %s group must be > 0, got %d", kernel, group))
	}
	nGroups, bpr := groupsFor(K, group)
	requireLen(kernel, "a", aLen, mul(M, K))
	requireLen(kernel, "w4", w4Len, mul(N, bpr))
	requireLen(kernel, "wScales", wScalesLen, mul(N, nGroups))
	requireLen(kernel, "dst", dstLen, mul(M, N))
}
