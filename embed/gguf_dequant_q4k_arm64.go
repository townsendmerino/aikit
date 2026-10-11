//go:build arm64

package embed

// dequantQ4KBodyNEON writes the 256 weights of one Q4_K super-block: for sub-block j, out[32j+l] is
// fl(mul[j]*nibble) - dmin*mins[j], the subtract and its product fused, which is the scalar's order on arm64. qs is the
// block's 128 nibble bytes. Implemented in gguf_dequant_q4k_arm64.s.
//
//go:noescape
func dequantQ4KBodyNEON(qs *byte, out *float32, mul, mins *[8]float32, dmin float32)

// dequantQ4KBlock dequantizes one 256-element Q4_K super-block (sb) into out[:256], bit for bit as
// dequantQ4KBlockScalar does on this architecture. The prologue is the scalar's; the weights go through the kernel.
func dequantQ4KBlock(raw []byte, sb int, out []float32) {
	base := sb * 144
	_ = raw[base+143]
	_ = out[255]
	mul, mins, dmin := q4kBlockScales(raw, base)
	dequantQ4KBodyNEON(&raw[base+16], &out[0], &mul, &mins, dmin)
}
