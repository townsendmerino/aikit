//go:build amd64

package linalg

func f16KernelActive() bool { return hasW4A8F16AVX2 && !hasAVX512VNNIVL }
