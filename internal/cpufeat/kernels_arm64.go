//go:build arm64

package cpufeat

// Kernels reports the arch, the tiers the hardware has (detected) and the tiers the flags leave on (active), for
// linalg.ActiveKernels.
func Kernels() (arch string, detected, active []string) {
	if DetectDotProd() {
		detected = append(detected, "dotprod")
	}
	if HasDotProd {
		active = append(active, "dotprod")
	}
	return "arm64", detected, active
}

// DisableTopTier turns DotProd off, the only tier above the base NEON kernels, and returns its name, or "" when it is already off.
func DisableTopTier() string {
	if HasDotProd {
		HasDotProd = false
		return "dotprod"
	}
	return ""
}
