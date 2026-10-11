//go:build amd64

package embed

import "golang.org/x/sys/cpu"

// hasAVX2 is true when the CPU has AVX2 and the OS saves the YMM state. Without it dequantQ4KBlock is the scalar body.
var hasAVX2 = cpu.X86.HasAVX2

// dequantQ4KBodyAVX2 writes the 256 weights of one Q4_K super-block: for sub-block j, out[32j+l] is
// mul[j]*nibble - off[j], a multiply and then a subtract, each rounded, as Go's amd64 code computes the scalar's
// `d1*float32(q) - off1`. qs is the block's 128 nibble bytes. Implemented in gguf_dequant_q4k_amd64.s.
//
//go:noescape
func dequantQ4KBodyAVX2(qs *byte, out *float32, mul, off *[8]float32)

// dequantQ4KBlock dequantizes one 256-element Q4_K super-block (sb) into out[:256], bit for bit as
// dequantQ4KBlockScalar does on this architecture. The prologue is the scalar's; the weights go through the kernel.
func dequantQ4KBlock(raw []byte, sb int, out []float32) {
	if !hasAVX2 {
		dequantQ4KBlockScalar(raw, sb, out)
		return
	}
	base := sb * 144
	_ = raw[base+143]
	_ = out[255]
	mul, off, dmin := q4kBlockScales(raw, base)
	for j := range off {
		off[j] = dmin * off[j] // the offset dmin·min, one float32 product, as the scalar computes it
	}
	dequantQ4KBodyAVX2(&raw[base+16], &out[0], &mul, &off)
}
