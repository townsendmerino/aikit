//go:build arm64 && aikit_nodotprod

package linalg

// aikit_nodotprod: dotI8 and the other SDOT dispatchers take the base SMULL/SADALP NEON kernels (see force.go), the path a Cortex-A72 class core runs.
func init() {
	hasDotProd = false
	forcedFallbacks = append(forcedFallbacks, "nodotprod")
}
