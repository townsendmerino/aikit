package linalg

import (
	"fmt"

	"github.com/townsendmerino/aikit/linalg/internal/shape"
)

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
// The products (M·K, N·K, …) are overflow-checked (internal/shape): a hostile shape
// whose product wraps int negative-and-small must not slip past a length compare.

// checkMatmulBT validates the f32 dst[M,N] = a[M,K]·b[N,K]ᵀ contract before the
// parallel fan-out.
func checkMatmulBT(kernel string, aLen, bLen, dstLen, M, K, N int) {
	if M < 0 || K < 0 || N < 0 {
		panic(fmt.Sprintf("linalg: %s negative dim (M=%d K=%d N=%d)", kernel, M, K, N))
	}
	shape.RequireLen(kernel, "a", aLen, shape.Mul(M, K))
	shape.RequireLen(kernel, "b", bLen, shape.Mul(N, K))
	shape.RequireLen(kernel, "dst", dstLen, shape.Mul(M, N))
}

// checkMatmulQ8 validates the int8-weight dst[M,N] contract (bQ is N×K int8, one
// f32 scale per output row) before the fan-out — shared by MatmulBTQ8, W8A8, and
// the W8A8 Into/Batch variants.
func checkMatmulQ8(kernel string, aLen, bQLen, bScalesLen, dstLen, M, K, N int) {
	if M < 0 || K < 0 || N < 0 {
		panic(fmt.Sprintf("linalg: %s negative dim (M=%d K=%d N=%d)", kernel, M, K, N))
	}
	shape.RequireLen(kernel, "a", aLen, shape.Mul(M, K))
	shape.RequireLen(kernel, "bQ", bQLen, shape.Mul(N, K))
	shape.RequireLen(kernel, "bScales", bScalesLen, N)
	shape.RequireLen(kernel, "dst", dstLen, shape.Mul(M, N))
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
	shape.RequireLen(kernel, "a", aLen, shape.Mul(M, K))
	shape.RequireLen(kernel, "w4", w4Len, shape.Mul(N, bpr))
	shape.RequireLen(kernel, "wScales", wScalesLen, shape.Mul(N, nGroups))
	shape.RequireLen(kernel, "dst", dstLen, shape.Mul(M, N))
}
