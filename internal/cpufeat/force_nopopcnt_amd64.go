//go:build amd64 && aikit_nopopcnt

package cpufeat

// aikit_nopopcnt: the Hamming search takes its portable path (see force.go).
func init() {
	HasPOPCNT = false
	forced = append(forced, "nopopcnt")
}
