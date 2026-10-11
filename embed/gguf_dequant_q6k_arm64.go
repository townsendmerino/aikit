//go:build arm64

package embed

// dequantQ6KBodyNEON writes the 256 weights of one Q6_K super-block from its 192 quant bytes at q (ql[128], then
// qh[64]) and the sixteen multipliers of q6kBlockScales: each weight is ds*float32(quant-32), one float32 product.
// Implemented in gguf_dequant_q6k_arm64.s.
//
//go:noescape
func dequantQ6KBodyNEON(q *byte, out *float32, ds *[16]float32)

// dequantQ6KBlock dequantizes one 256-element Q6_K super-block (sb) into out[:256], bit for bit as
// dequantQ6KBlockScalar does. The prologue is the scalar's; the weights go through the kernel.
func dequantQ6KBlock(raw []byte, sb int, out []float32) {
	base := sb * 210
	_ = raw[base+209]
	_ = out[255]
	ds := q6kBlockScales(raw, base)
	dequantQ6KBodyNEON(&raw[base], &out[0], &ds)
}
