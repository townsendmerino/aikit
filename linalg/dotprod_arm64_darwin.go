//go:build arm64 && darwin

package linalg

// detectDotProd: every Apple Silicon core (M1 and later — the only arm64 Macs)
// implements DotProd, so the SDOT kernel is always available on darwin/arm64.
func detectDotProd() bool { return true }
