//go:build !arm64

package linalg

// twRow4 is nils off arm64, where the row4 layout has no kernel and its repack functions
// do not exist.
func twRow4([]byte, []float32, twShape) ([]byte, []float32, bool) { return nil, nil, false }

// twRow4F16 is twRow4 with binary16 scales.
func twRow4F16([]byte, []uint16, twShape) ([]byte, []uint16, bool) { return nil, nil, false }
