//go:build amd64

package linalg

// expF32ContractAVX2 computes e^src[i] into dst[i] for n elements (n a multiple
// of 8), implementing the exp_contract.go numeric contract exactly. Only safe
// when hasAVX2 (which already requires FMA3 — the contract needs a true fused
// multiply-add and AVX2 alone cannot provide one).
//
// Bit-identical to expF32Contract AND to the arm64 kernel, which is what makes a
// golden portable across the two.
//
//go:noescape
func expF32ContractAVX2(dst, src *float32, n int)

// tanhF32ContractAVX2 computes tanh(src[i]) into dst[i] for n elements (n a
// multiple of 8), implementing the contract exactly. Only safe when hasAVX2.
//
// Bit-identical to tanhF32Contract and to tanhF32ContractNEON, including the
// deliberate late saturation the contract specifies (exactly 1 only from
// |x| >= 9.02, not TanhF32's x > 9).
//
//go:noescape
func tanhF32ContractAVX2(dst, src *float32, n int)

// erfF32ContractAVX2 computes erf(src[i]) into dst[i] for n elements (n a
// multiple of 8), implementing the contract exactly. Only safe when hasAVX2.
//
// Bit-identical to erfF32Contract and to erfF32ContractNEON.
//
//go:noescape
func erfF32ContractAVX2(dst, src *float32, n int)
