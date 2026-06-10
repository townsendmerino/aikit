//go:build !aikit_checks

package linalg

// Production build: the quant-kernel contract checks compile to empty functions.
// They take concrete (non-interface) arguments, so the calls inline to nothing —
// no argument boxing, no branch, zero hot-path cost. Build or test with
// `-tags aikit_checks` to turn them into validating panics (see checks_on.go).
//
// These guard caller contracts the hot kernels otherwise trust silently: a wrong
// shape or group=0 would divide-by-zero or read out of bounds deep inside an asm
// kernel; the checked build fails loudly at the entry instead.

const checksEnabled = false

func checkDequantInt8(q []int8, dst []float32)                                         {}
func checkDequantInt4(packed []byte, scales []float32, group, cols int, dst []float32) {}
func checkGroupMatmul(name string, aLen int, packed []byte, scales []float32, dstLen, M, K, N, group int) {
}
