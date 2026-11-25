//go:build !amd64 && !arm64

package cpufeat

import "runtime"

// Kernels reports the arch, the tiers the hardware has (detected) and the tiers the flags leave on (active), for
// linalg.ActiveKernels.
func Kernels() (arch string, detected, active []string) { return runtime.GOARCH, nil, nil }

// DisableTopTier has no tier to turn off on this arch.
func DisableTopTier() string { return "" }
