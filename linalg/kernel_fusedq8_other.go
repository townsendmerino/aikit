//go:build !arm64 && !amd64

package linalg

// HasFusedQ8Kernel — see the arm64 definition. False on architectures with
// neither the NEON packed kernel nor an AVX2 blocked GEMM worth strip-fusing
// against: there the dequant-then-GEMM fallback is as good as anything here.
const HasFusedQ8Kernel = false
