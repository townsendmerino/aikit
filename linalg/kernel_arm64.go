//go:build arm64

package linalg

// has2x8Kernel gates the 2×8 dual-row micro-kernel (Dot2x8) in the blocked GEMM. It is
// a compile-time const so the dual-row block is dead-code-eliminated on arches without
// the NEON kernel — amd64 stays on the Dot8x4 path.
const has2x8Kernel = true

// HasFusedQ8Kernel reports whether MatmulBTQ8Fused's packed path uses the NEON
// Dot2x8/Dot8x4 kernels (arm64). Elsewhere the fused pack would run the scalar Dot2x8
// fallback — slower than amd64's AVX2 dequant-then-GEMM — so the encoder gates on this.
const hasFusedQ8Kernel = true
