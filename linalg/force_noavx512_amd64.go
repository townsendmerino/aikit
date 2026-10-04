//go:build amd64 && aikit_noavx512

package linalg

// aikit_noavx512: the AVX-512 VNNI kernels are never selected (see force.go). Package-level variables are initialised before any init function, so this
// overrides the detected values, and it sets the derived flag explicitly because hasAVX512VNNIVL was computed from hasAVX512VNNI at variable-initialisation time.
func init() {
	hasAVX512VNNI = false
	hasAVX512VNNIVL = false
	forcedFallbacks = append(forcedFallbacks, "noavx512")
}
