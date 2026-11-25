//go:build amd64 && aikit_noavx512

package cpufeat

// aikit_noavx512: the AVX-512 VNNI kernels are never selected (see force.go). Package-level variables are initialised before any init function, so this
// overrides the detected values, and it sets the derived flag explicitly because HasAVX512VNNIVL was computed from HasAVX512VNNI at variable-initialisation time.
func init() {
	HasAVX512VNNI = false
	HasAVX512VNNIVL = false
	forced = append(forced, "noavx512")
}
