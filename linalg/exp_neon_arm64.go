//go:build arm64

package linalg

// expF32ContractNEON computes e^src[i] into dst[i] for the first n elements, four
// lanes at a time, implementing the exp_contract.go numeric contract exactly.
//
// n must be a multiple of 4 and every input must be finite and within
// [expUnderflowF32, expOverflowF32] — the Go caller guards, exactly as ExpF32
// guards expF32Core. dst and src may alias.
//
// Bit-identical to expF32Contract, not merely close: the oracle's fma32 is a
// correctly-rounded f32 FMA and so is FMLA, so every intermediate matches.
// TestExpF32NEON_bitIdenticalToContract asserts Float32bits equality.
//
//go:noescape
func expF32ContractNEON(dst, src *float32, n int)

// siluF32ContractNEON computes dst[i] = src[i]/(1+exp(clamp(-src[i]))) for n
// elements (n a multiple of 4). See the assembly for why the clamp is required
// rather than defensive.
//
//go:noescape
func siluF32ContractNEON(dst, src *float32, n int)
