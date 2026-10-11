//go:build !arm64 && !amd64

package embed

// dequantQ6KBlock dequantizes one 256-element Q6_K super-block (sb) into out[:256]. No vector kernel here: the scalar.
func dequantQ6KBlock(raw []byte, sb int, out []float32) { dequantQ6KBlockScalar(raw, sb, out) }
