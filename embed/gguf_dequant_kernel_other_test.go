//go:build !arm64 && !amd64

package embed

// dequantKernelsActive reports whether dequantQ4KBlock and dequantQ6KBlock run their vector kernels on this CPU.
func dequantKernelsActive() bool { return false }
