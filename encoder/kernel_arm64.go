//go:build arm64

package encoder

// has2x8Kernel gates the 2×8 dual-row GEMM micro-kernel (linalg.Dot2x8) on in the blocked
// matmul. It is a compile-time const so the dual-row block is dead-code-eliminated on
// arches without the NEON kernel — amd64 keeps its AVX2 Dot8x4 path untouched.
const has2x8Kernel = true
