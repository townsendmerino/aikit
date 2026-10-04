//go:build !amd64 && !arm64

package linalg

import "runtime"

func archKernels() (arch string, detected, active []string) { return runtime.GOARCH, nil, nil }

func disableTopTier() string { return "" }
