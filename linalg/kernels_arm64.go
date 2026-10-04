//go:build arm64

package linalg

func archKernels() (arch string, detected, active []string) {
	if detectDotProd() {
		detected = append(detected, "dotprod")
	}
	if hasDotProd {
		active = append(active, "dotprod")
	}
	return "arm64", detected, active
}

// disableTopTier turns DotProd off, the only tier above the base NEON kernels, and returns its name, or "" when it is already off.
func disableTopTier() string {
	if hasDotProd {
		hasDotProd = false
		return "dotprod"
	}
	return ""
}
