// Package linalg holds aikit's shared SIMD compute kernels — the single home for
// the hand-written assembly, so the encoder and goinfer's LLM decoder share one
// copy. It provides float32 dot products and a row-parallel matmul, the
// register-blocked micro-kernels a cache-blocked matmul builds on, and the
// quantized matmuls (int8 weights, int4 groups, and the W8A8 / W4A8
// integer-activation paths) that LLM decode uses.
//
// # Kernel dispatch
//
// Every kernel has a portable scalar implementation; faster paths are selected
// by build tag (GOARCH) and, within an arch, by runtime CPU feature detection —
// so one binary runs the best kernel its CPU supports and never executes an
// instruction the CPU would trap on.
//
//	Arch    f32 dot / matmul             int8 dot (Q8 / W8A8 / W4A8)     selected by
//	------- ---------------------------- ------------------------------- ------------------------
//	arm64   NEON (dot_arm64.s)           NEON SMULL/SADALP → SDOT         build tag + HWCAP probe
//	amd64   AVX2+FMA, scalar fallback    AVX2, scalar fallback           build tag + CPUID/XGETBV
//	other   portable scalar              portable scalar                 build tag
//
// On arm64 the int8 path upgrades from the base SMULL/SADALP kernel to the
// ARMv8.2 SDOT kernel when detectDotProd reports HWCAP_ASIMDDP. On amd64 the f32
// and int8 kernels use AVX2+FMA when CPUID/XGETBV confirm OS+CPU support, else
// scalar. The fused int4×int8 decode kernel (MatmulBTW4A8) has both an arm64+SDOT
// and an amd64+AVX2 implementation; non-DotProd arm64 and non-AVX2 amd64 fall
// back to the scalar reference (a VNNI amd64 variant is a planned follow-up).
// Detection runs once at init; the hot path is a branch on a bool.
//
// # Choosing a kernel
//
//   - Dot, MatmulBT — f32, the general path. MatmulBT is row-parallel above a
//     MAC-count threshold (see SetParallelThreshold / SetParallelWidth).
//   - Dot4x4, Dot8x4 — register-blocked f32 micro-kernels for a cache-blocked
//     matmul that tiles K (see Dot8x4 on the large-K cliff).
//   - MatmulBTQ8, MatmulBTQ4 — quantized weights, f32 activations (prefill).
//   - MatmulBTW8A8, MatmulBTW4A8 — quantized weights AND int8 activations, the
//     fast M=1 (decode) paths. W8A8 also has zero-alloc (…Into) and batched forms.
//
// # Numerical invariant
//
// Parallelization and re-blocking partition output columns/rows — each output is
// computed by a single worker over the full K reduction — so the parallel and
// blocked paths are bit-identical to a SERIAL RUN OF THE SAME KERNEL: the
// differential tests assert exact equality for parallel==serial and for width-
// and M-invariance. (Against a naive scalar triple-loop the blocked kernel
// differs by float reassociation — that comparison is tolerance-checked, ~1e-3.)
package linalg
