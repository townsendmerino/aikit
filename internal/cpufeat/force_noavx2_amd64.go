//go:build amd64 && aikit_noavx2

package cpufeat

// aikit_noavx2: every kernel behind a HasAVX2 branch takes its pure-Go path (see force.go). The flags derived from HasAVX2 at variable-initialisation time
// (HasQ4KAVX2) and the AVX-512 flags (a CPU with AVX-512 VNNI also has AVX2, so "no AVX2" implies "no AVX-512") are set explicitly. HasF16C and HasPOPCNT are
// separate CPU features and stay as detected; aikit_nopopcnt forces the latter.
func init() {
	HasAVX2 = false
	HasQ4KAVX2 = false
	HasAVX512VNNI = false
	HasAVX512VNNIVL = false
	forced = append(forced, "noavx2")
}
