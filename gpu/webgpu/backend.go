// Package webgpu holds the WGSL compute shader source library for aikit's
// future WebGPU dispatch path (tiled f32 GEMM, quantized int8 W8A8 GEMM,
// workgroup LayerNorm/RMSNorm, GELU/SiLU, and multi-head attention).
//
// It does NOT implement encoder.Backend and does not register a "webgpu"
// backend. Per encoder/backend.go's documented architecture, the real
// "webgpu" encoder.Backend lives in the opt-in github.com/townsendmerino/goinfer/gpu
// module, dispatching these (or equivalent) shaders against an actual device
// via github.com/townsendmerino/wgpu — so aikit itself never imports a cgo
// WebGPU implementation. A package here that silently registered "webgpu"
// while just calling the CPU matmul would shadow that real backend with one
// that does no GPU work and claims otherwise; this package is deliberately
// just the shader assets, for goinfer/gpu (or a future real dispatch layer)
// to consume.
package webgpu

import (
	_ "embed"
)

// Shaders embedded from the WGSL compute suite.
var (
	//go:embed wgsl/matmul_f32.wgsl
	MatmulF32WGSL string

	//go:embed wgsl/matmul_w8a8.wgsl
	MatmulW8A8WGSL string

	//go:embed wgsl/layernorm.wgsl
	LayerNormWGSL string

	//go:embed wgsl/rmsnorm.wgsl
	RMSNormWGSL string

	//go:embed wgsl/gelu.wgsl
	GELUWGSL string

	//go:embed wgsl/silu.wgsl
	SiLUWGSL string

	//go:embed wgsl/attention.wgsl
	AttentionWGSL string
)
