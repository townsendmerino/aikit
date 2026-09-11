//go:build !amd64 && !arm64

package linalg

// w4a8TileRows claims no activation rows on architectures with neither tile. The W4A8 activation-blocked
// tile is an AVX2 kernel on amd64 (S-01) and an SDOT kernel on arm64 (audit
// M-02); elsewhere there is neither.
//
// Returning 0 makes w4a8Span run precisely the loop it ran before any tile
// existed.
func w4a8TileRows(aq []int8, aScales []float32, w4 []byte, wScales, dst []float32, M, K, N, group, nGroups, bpr, j0, j1 int) int {
	return 0
}
