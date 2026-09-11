package linalg

import "fmt"

// DotI8 returns the int32 dot product of two equal-length int8 vectors via the
// platform SIMD kernel (AVX2 VPMADDWD on amd64, SDOT on ARMv8.2, scalar fallback
// elsewhere). It is the node-node integer similarity primitive for the int8 ann
// indexes (rescale by the two per-vector scales to recover cosine).
// Experimental, like the rest of linalg. (QuantizeRowInt8, the single-row
// f32→int8 quantizer for a query vector, is already exported in quant.go.)
//
// Panics if len(a) != len(b): the SIMD dispatch reads n=len(a) elements of both,
// so a short b would read past its allocation (silent garbage or a fault) on the
// vector path and bounds-panic on the scalar tail — an arch-dependent failure
// mode. The check makes it uniform and cheap (matches dotF32).
func DotI8(a, b []int8) int32 {
	if len(a) != len(b) {
		panic(fmt.Sprintf("linalg: DotI8 length mismatch (%d != %d)", len(a), len(b)))
	}
	return dotI8(a, b)
}

// DotI8x8 computes eight int8 dot products of a shared query row against eight
// candidate rows, sharing the query's widening across all eight instead of
// paying it once per candidate (25 ops per 128 MACs on the AVX2 path against
// eight separate DotI8 calls' 32) — the int8 twin of Dot8x4's gathered shape,
// for a scattered candidate set (int8-mode HNSW's neighbour list, audit M-21).
// Bit-identical to eight separate DotI8 calls: integer addition is associative
// and int8×int8→int32 cannot overflow for any length this library sees.
// Experimental, like the rest of linalg.
//
// Panics if any of b0..b7's lengths differ from len(a).
func DotI8x8(a, b0, b1, b2, b3, b4, b5, b6, b7 []int8) [8]int32 {
	n := len(a)
	for i, b := range [8][]int8{b0, b1, b2, b3, b4, b5, b6, b7} {
		if len(b) != n {
			panic(fmt.Sprintf("linalg: DotI8x8 length mismatch (a=%d, b%d=%d)", n, i, len(b)))
		}
	}
	var out [8]int32
	dotI8x8(a, b0, b1, b2, b3, b4, b5, b6, b7, &out)
	return out
}
