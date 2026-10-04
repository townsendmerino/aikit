//go:build amd64 && aikit_noavx2

package linalg

// aikit_noavx2: every kernel behind a hasAVX2 branch takes its pure-Go path (see force.go). The flags derived from hasAVX2 at variable-initialisation time
// (hasQ4KAVX2) and the AVX-512 flags (a CPU with AVX-512 VNNI also has AVX2, so "no AVX2" implies "no AVX-512") are set explicitly. hasF16C and hasPOPCNT are
// separate CPU features and stay as detected; aikit_nopopcnt forces the latter.
func init() {
	hasAVX2 = false
	hasQ4KAVX2 = false
	hasAVX512VNNI = false
	hasAVX512VNNIVL = false
	forcedFallbacks = append(forcedFallbacks, "noavx2")
}
