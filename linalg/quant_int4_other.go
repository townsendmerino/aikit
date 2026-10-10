//go:build !arm64 && !amd64

package linalg

// int4GroupHasKernel is false where there is no vector kernel: QuantizeGroupInt4Row is then scalar throughout.
func int4GroupHasKernel(ks, n int) bool { return false }

func quantizeInt4Block(row []float32, packed []byte, inv float32) {
	panic("linalg: quantizeInt4Block has no kernel on this architecture")
}
