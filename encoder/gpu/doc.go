// Package gpu is an OPTIONAL WebGPU (Metal / Vulkan / DX12) compute
// backend for the encoder's matmul, compiled only under the `gpu`
// build tag.
//
// The default aikit build does not include this package's
// implementation: every file except this doc carries `//go:build gpu`,
// so the cgo dependency on github.com/cogentcore/webgpu — which bundles
// the wgpu-native Rust library — is built ONLY when you compile or test
// with `-tags gpu`. Without the tag this package is empty and the
// module stays pure-Go / no-cgo, preserving aikit's headline promise.
//
// Status: FOUNDATION cut. A single `dst = a·bᵀ` GEMM offloaded to the
// GPU (upload → dispatch → readback) for correctness and throughput
// measurement. A single offloaded matmul is expected to LOSE to the CPU
// path on small/medium shapes because of host↔device transfer + kernel-
// launch overhead; the win only appears once the whole forward stays
// resident on-GPU across layers for large batches. See the GPU section
// of docs/cpu-acceleration.md for the measured numbers and follow-ups.
package gpu
