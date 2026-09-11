//go:build amd64

package linalg

// dotI8x8AVX2 computes eight int8 dot products of one a-row against eight b-rows,
// sharing the a-side widening across all eight. n must be a multiple of 16.
//
//go:noescape
func dotI8x8AVX2(a, b0, b1, b2, b3, b4, b5, b6, b7 *int8, n int, out *[8]int32)

// dotI8x8 is the AVX2-dispatched int8 twin of Dot8x4's gathered shape (audit
// M-21). This kernel — dotI8x8AVX2, gpu/../GEMM's dotI8Cols8 before it — was
// deleted once as dead code (c888546) when the GEMM use it was built for was
// reverted: a LINEAR B-matrix scan lost to the hardware prefetcher once B
// stopped being cache-resident, because eight streams K bytes apart is worse
// for sequential prefetch than one. HNSW's candidate set has no sequential
// prefetch to lose — h.code(ids[i]) is already an arbitrary GATHER regardless
// of kernel shape — so that retirement reason does not apply here; recovered
// (byte-identical assembly, from the commit before deletion) for this
// different caller instead of rewritten from scratch.
//
// The K%16 remainder is finished per column in Go, same split doti8cols_amd64
// used: integer arithmetic is associative, so splitting the sum this way is
// exactly equal to the unsplit one.
func dotI8x8(a, b0, b1, b2, b3, b4, b5, b6, b7 []int8, out *[8]int32) {
	n := len(a)
	if !hasAVX2 || n < 16 {
		dotI8x8Generic(a, b0, b1, b2, b3, b4, b5, b6, b7, out)
		return
	}
	n16 := n &^ 15
	dotI8x8AVX2(&a[0], &b0[0], &b1[0], &b2[0], &b3[0], &b4[0], &b5[0], &b6[0], &b7[0], n16, out)
	if n16 == n {
		return
	}
	bs := [8][]int8{b0, b1, b2, b3, b4, b5, b6, b7}
	for c := range bs {
		s := out[c]
		bc := bs[c]
		for k := n16; k < n; k++ {
			s += int32(a[k]) * int32(bc[k])
		}
		out[c] = s
	}
}
