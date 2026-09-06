package linalg

import "math"

// Portable entry points for the S-06 numeric contract. Each selects the fastest
// conforming implementation for the build and they are the ONLY way the rest of
// the package should reach the contract, so a caller cannot accidentally get the
// scalar path on an arch that has a kernel, or vice versa.
//
// Every one of these is bit-identical across architectures BY CONSTRUCTION —
// that is what the contract in exp_contract.go buys. A NEON build and a pure-Go
// build produce the same bits, so a golden generated on one is valid on the
// other.
//
// This file also exists for a mundane reason worth recording: without it the
// scalar contract functions have no callers on amd64 (the arm64 kernels and
// their tests being excluded there), and golangci-lint's `unused` correctly
// reports them. That failure is invisible to a native `golangci-lint run` on an
// arm64 box and only appears in CI's linux/amd64 job — the cross-GOARCH lint
// gotcha this repo has hit before. Run `GOOS=linux GOARCH=amd64 golangci-lint
// run ./linalg/...` before pushing anything arch-split.

// ExpContractInto writes e^src[i] into dst[i] under the contract.
func ExpContractInto(dst, src []float32) {
	if len(dst) != len(src) {
		panic("linalg: ExpContractInto length mismatch")
	}
	expContractImpl(dst, src)
}

// SoftmaxRowContractInto normalises one row under the contract, with the
// denominator's summation order pinned (four f64 lane partials, fixed fold).
func SoftmaxRowContractInto(dst, src []float32) {
	if len(dst) != len(src) {
		panic("linalg: SoftmaxRowContractInto length mismatch")
	}
	softmaxContractImpl(dst, src)
}

// SiLUContractInto writes x/(1+e^-x) elementwise under the contract.
func SiLUContractInto(dst, src []float32) {
	if len(dst) != len(src) {
		panic("linalg: SiLUContractInto length mismatch")
	}
	siluContractImpl(dst, src)
}

// GELUTanhContractInto writes the tanh-approximation GELU elementwise under the
// contract.
func GELUTanhContractInto(dst, src []float32) {
	if len(dst) != len(src) {
		panic("linalg: GELUTanhContractInto length mismatch")
	}
	geluTanhContractImpl(dst, src)
}

// geluTanhScalarInto is the reference loop and the non-arm64 implementation.
func geluTanhScalarInto(dst, src []float32) {
	for i, v := range src {
		dst[i] = geluTanhF32Contract(v)
	}
}

// expContractScalarInto is the reference loop, shared by the non-arm64 impl and
// used directly by tests as the oracle.
func expContractScalarInto(dst, src []float32) {
	for i, v := range src {
		switch {
		case v != v:
			dst[i] = v
		case v > expOverflowF32:
			dst[i] = float32(math.Inf(1))
		case v < expUnderflowF32:
			dst[i] = 0
		default:
			dst[i] = expF32Contract(v)
		}
	}
}

func siluContractScalarInto(dst, src []float32) {
	for i, v := range src {
		dst[i] = siluF32Contract(v)
	}
}
