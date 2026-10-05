//go:build !arm64 && !amd64

package linalg

// archContractCases: the generic (!arm64 && !amd64) build has no arch-only writers beyond the portable ones outputContractCases already covers.
func archContractCases() []contractCase { return nil }
