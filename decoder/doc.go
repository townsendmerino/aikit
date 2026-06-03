// Package decoder runs autoregressive, decoder-only transformer language
// models (Gemma 3, the open-weights Gemini family) as a pure-Go forward
// pass. It is the generative sibling of encoder/ (bidirectional embeddings)
// and embed/ (Model2Vec), sharing embed/'s safetensors loader.
//
// Status: SCAFFOLD. The architecture, loader contract, backend seam, KV
// cache, RMSNorm and sampler are stubbed to compile and run with honest
// "not implemented" errors; the forward pass arrives across the milestones
// in docs/gemma-decoder-plan.md. Do not mistake a green build for a working
// model yet.
//
// # Carry-over invariants (read once)
//
//   - Gemma scales the embedding output by sqrt(hidden_dim) and ties the
//     input embedding table to the LM head. Both are correctness-critical.
//   - Gemma's RMSNorm scales by (1 + weight), not weight. A plain RMSNorm
//     silently shifts every activation.
//   - Gemma 3 uses TWO RoPE tables: local layers θ=10000, global layers
//     θ=1000000. Using one base corrupts long-context positions.
//   - Gemma 3 dropped Gemma 2's logit/attention soft-capping in favor of
//     query/key RMSNorm (QK-norm). A loader must reject a soft-capping
//     checkpoint rather than ignore the field.
//   - Attention is causal with grouped-query heads (270M: 4 query : 1 KV)
//     and a sliding window (512) on the local layers. The KV cache, not a
//     per-call growing buffer, is the memory model.
//
// All compute matmuls route through a Backend (see backend.go) so a WebGPU
// backend can replace the CPU one without touching the forward pass.
package decoder

import "errors"

// errNotImplemented marks a scaffold seam that a milestone in
// docs/gemma-decoder-plan.md will fill in. It is wrapped with the specific
// milestone/section so a caller hitting it gets a pointer, not a mystery.
var errNotImplemented = errors.New("decoder: not implemented (see docs/gemma-decoder-plan.md)")
