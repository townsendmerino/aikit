//go:build !arm64

package linalg

// Non-arm64 impls: the scalar contract. Bit-identical to the arm64 kernels by
// construction, so nothing about a model's output depends on which build ran it
// — which is the entire point of the contract and the reason these are not left
// as a faster-but-different fallback.

func expContractImpl(dst, src []float32)      { expContractScalarInto(dst, src) }
func softmaxContractImpl(dst, src []float32)  { softmaxRowContract(dst, src) }
func siluContractImpl(dst, src []float32)     { siluContractScalarInto(dst, src) }
func geluTanhContractImpl(dst, src []float32) { geluTanhScalarInto(dst, src) }
