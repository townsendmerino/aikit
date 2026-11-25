//go:build arm64 && darwin

package cpufeat

// DetectDotProd: every Apple Silicon core (M1 and later — the only arm64 Macs)
// implements DotProd, so the SDOT kernel is always available on darwin/arm64.
func DetectDotProd() bool { return true }
