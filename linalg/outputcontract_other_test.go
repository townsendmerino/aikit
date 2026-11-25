//go:build !arm64 && !amd64

package linalg

import "github.com/townsendmerino/aikit/linalg/internal/outputcontract"

// archContractCases: the generic (!arm64 && !amd64) build has no arch-only writers beyond the portable ones outputContractCases already covers.
func archContractCases() []outputcontract.Case { return nil }
