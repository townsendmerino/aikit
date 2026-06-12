//go:build arm64

package linalg

// fmaPeakARM64 runs a register-saturating f32 NEON FMA loop (no memory traffic) to
// measure the core's achievable FMA throughput — used by the GEMM peak-fraction gate
// to ground the "fraction of peak" denominator in a measured ceiling, not a spec sheet.
//
//go:noescape
func fmaPeakARM64(iters int64)
