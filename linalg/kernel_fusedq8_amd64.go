//go:build amd64

package linalg

// HasFusedQ8Kernel — see the arm64 definition. TRUE here since audit M-23.
//
// It used to be false off arm64, on the reasoning that "the fused packed path
// has no NEON kernel, so the encoder keeps the AVX2 dequant-then-f32-GEMM
// path". That was right about the PACKED form — packedFillQ8 runs Dot2x8, which
// is the scalar fallback here — but it left amd64 widening the entire [N,K]
// weight matrix to f32 on every matmul call, up to 9.4 MB written and read
// straight back per call, O(N*K) regardless of sequence length.
//
// fusedQ8Strip is the form that fits this architecture: widen one N-strip at a
// time and run the ORDINARY blockedFill over it, which on amd64 is the
// blockRows3x4/Dot8x4 path the f32 GEMM already uses. No new kernel, and
// bit-identical to the reference because it reuses blockedFill rather than
// reimplementing its reduction order.
const HasFusedQ8Kernel = true
