//go:build !arm64 && !amd64

package embed

// dequantQ4KBlock dequantizes one 256-element Q4_K super-block (sb) into out[:256]. No vector kernel here: the scalar.
func dequantQ4KBlock(raw []byte, sb int, out []float32) { dequantQ4KBlockScalar(raw, sb, out) }
