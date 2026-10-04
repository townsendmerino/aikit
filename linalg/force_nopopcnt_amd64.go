//go:build amd64 && aikit_nopopcnt

package linalg

// aikit_nopopcnt: the Hamming search takes its portable path (see force.go).
func init() {
	hasPOPCNT = false
	forcedFallbacks = append(forcedFallbacks, "nopopcnt")
}
