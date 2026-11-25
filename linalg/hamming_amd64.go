//go:build amd64

package linalg

import "github.com/townsendmerino/aikit/internal/cpufeat"

// hammingPOPCNT is the POPCNTQ kernel in hamming_amd64.s.
//
//go:noescape
func hammingPOPCNT(q *uint64, codes *uint64, words int, n int, dst *uint16)

func hammingRows(q, codes []uint64, words, n int, dst []uint16) {
	if !cpufeat.HasPOPCNT || n == 0 || words == 0 {
		hammingRowsGeneric(q, codes, words, n, dst)
		return
	}
	hammingPOPCNT(&q[0], &codes[0], words, n, &dst[0])
}
