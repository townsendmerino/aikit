// Package elem holds aikit's elementwise f32 kernels: exp, tanh, erf, SiLU, GELU (erf and tanh forms) and softmax, each
// as a scalar function (ExpF32, ...) and a slice form (ExpF32Into, ...), and the contract forms (ExpContractInto, ...),
// in which every multiply-add is a fused multiply-add on every architecture, so they give the same bits on amd64 and
// arm64. The slice forms dispatch to AVX2 or NEON where the CPU has it (internal/cpufeat) and fall back to the scalar
// loops, which are the oracle each kernel is held to. Every writer's doc comment states its output contract.
package elem
