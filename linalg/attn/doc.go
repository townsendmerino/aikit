// Package attn holds aikit's attention kernels: the float64-accumulating QK and AV matmuls (MatmulQKAcc64, MatmulAVAcc64),
// their grouped forms that share one KV head's load across its query heads, their forms over a binary16 cache read at two
// bytes an element, and the fused attention tile (AttendTileFused). Each acc64 kernel folds every output lane in float64
// over ascending d, so it is bit-identical to a scalar float64 reference on every architecture; the fused tile is not
// bit-identical to a materialized QK, softmax and AV (its running-max rescale re-associates them). The kernels dispatch to
// AVX2/F16C or NEON where the CPU has them (internal/cpufeat). Every writer's doc comment states its output contract.
package attn
