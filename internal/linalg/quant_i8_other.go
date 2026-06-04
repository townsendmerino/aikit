//go:build !amd64 && !arm64

package linalg

// dotI8 is the int8×int8→int32 inner product for the W8A8 matmul. Off amd64/arm64
// it is the scalar reference.
func dotI8(a, b []int8) int32 { return dotI8Scalar(a, b) }
