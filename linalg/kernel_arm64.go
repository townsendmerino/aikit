//go:build arm64

package linalg

// has2x8Kernel gates the 2×8 dual-row micro-kernel (Dot2x8) in the blocked GEMM. It is
// a compile-time const so the dual-row block is dead-code-eliminated on arches without
// the NEON kernel — amd64 stays on the Dot8x4 path.
const has2x8Kernel = true
