//go:build !amd64 && !arm64

package linalg

// w4a8GroupedFastRows: no repacked W4A8 kernel on this architecture; matmulW4A8Grouped uses the
// canonical kernel or the reference.
func w4a8GroupedFastRowsQ(ws *Workspace, q *ActQ, l int4Layout, dst []float32, M, N int) bool {
	return false
}
