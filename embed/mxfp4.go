package embed

import "fmt"

// MXFP4 — OCP Microscaling FP4, ggml quant type 39, used by the gpt-oss family.
//
// A block is 32 elements: one e8m0 8-bit power-of-two scale and 16 bytes each packing two e2m1
// 4-bit values, so 4.25 bits per weight. The value table is the 16 representable e2m1 values
// DOUBLED, paired with a HALF-scale e8m0, so d_half · value_doubled recovers the true product
// while both stay integer — which is what lets a SIMD kernel skip a float dequant in its inner
// loop. Layout transcribed from the reference gguf Python library (gguf/quants.py MXFP4) and the
// OCP MX v1.0 spec.
//
// TWO LAYOUTS SHIP IN THE WILD, AND THEY ARE NOT INTERCHANGEABLE. They share the block size, the
// scale encoding and the value table, which makes them look like one function with different
// addressing:
//
//	GGUF/GGML   contiguous 17-byte blocks   byte j packs elements j and j+16
//	safetensors two separate tensors        byte j packs elements 2j and 2j+1
//
// This was measured, not assumed, and the assumption it replaced was wrong: dequantizing a real
// gpt-oss expert both ways and diffing against the same weight read through the validated GGUF
// path gave cosine 0.081 for GGML order against 1.000000 for sequential. Feeding safetensors data
// through the GGML core yields finite, plausibly-scaled, completely wrong weights — it does not
// error and it does not look broken. DequantMXFP4Blocks and DequantMXFP4Split therefore stay
// separate on purpose; embed/mxfp4_goinfer_ref_test.go has a gate that goes red if they are ever
// unified.

const (
	// MXFP4BlockElems is the number of weights in one MXFP4 block.
	MXFP4BlockElems = 32
	// MXFP4BlockBytes is one contiguous GGUF block: 1 scale byte + 16 packed-nibble bytes.
	MXFP4BlockBytes = 17
	// mxfp4PackedBytes is the packed-nibble half alone, which is what the safetensors *_blocks
	// tensor holds (its scales live in a separate *_scales tensor).
	mxfp4PackedBytes = 16
)

// MXFP4Scale converts an 8-bit e8m0 block scale to float32, matching ggml's
// ggml_e8m0_to_fp32_half (gguf/quants.py MXFP4.e8m0_to_fp32_half): x<2 is a subnormal bit pattern,
// otherwise the exponent field is x-1. The exact bit formula rather than 2^(x-128), because that
// is the only way the x in {0,1} subnormals come out bit-identical to the reference.
func MXFP4Scale(e8m0 byte) float32 { return e8m0ToF32Half(e8m0) }

// DequantMXFP4Blocks dequantizes nBlocks contiguous 17-byte MXFP4 blocks (the GGUF layout) into
// dst[:nBlocks*32], in natural element order. Returns an error rather than panicking on a short or
// misaligned buffer: these bytes come from a file, and hostile input must refuse, not crash.
//
// See the package comment: this is the j / j+16 nibble order. Safetensors checkpoints need
// DequantMXFP4Split instead, and picking the wrong one is silent.
func DequantMXFP4Blocks(raw []byte, nBlocks int, dst []float32) error {
	if nBlocks < 0 {
		return fmt.Errorf("mxfp4: negative block count %d", nBlocks)
	}
	if len(raw) < nBlocks*MXFP4BlockBytes {
		return fmt.Errorf("mxfp4: %d bytes for %d blocks (need %d)", len(raw), nBlocks, nBlocks*MXFP4BlockBytes)
	}
	if len(dst) < nBlocks*MXFP4BlockElems {
		return fmt.Errorf("mxfp4: dst has %d elements for %d blocks (need %d)", len(dst), nBlocks, nBlocks*MXFP4BlockElems)
	}
	for i := range nBlocks {
		dequantMXFP4Block(raw, i, dst[i*MXFP4BlockElems:])
	}
	return nil
}

// DequantMXFP4Split dequantizes the SAFETENSORS layout, where the packed nibbles and the scales
// are two different tensors rather than interleaved blocks:
//
//	*_blocks  U8 [..., nBlocks, 16]   the packed nibbles
//	*_scales  U8 [..., nBlocks]       one e8m0 exponent per block
//
// and byte j holds elements 2j and 2j+1 — sequential, NOT the GGML j / j+16 split. See the package
// comment for the measurement that established the difference.
//
// It writes into a caller-owned dst rather than allocating, because the caller that needs it is
// streaming: gpt-oss-20b's experts are ~76 GB dequantized to f32 across all layers, so they are
// consumed a row at a time into a quantized destination and never materialized. A per-row
// allocation there would be 24 layers x 32 experts x 8640 rows of garbage.
//
// The shapes are checked exactly rather than as lower bounds: blocks and scales are two
// independently-shaped tensors, and a mismatch between them is precisely the corruption worth
// refusing loudly.
func DequantMXFP4Split(blocks, scales []byte, nBlocks int, dst []float32) error {
	if nBlocks < 0 {
		return fmt.Errorf("mxfp4: negative block count %d", nBlocks)
	}
	if len(scales) != nBlocks {
		return fmt.Errorf("mxfp4: %d scale bytes for %d blocks", len(scales), nBlocks)
	}
	if len(blocks) != nBlocks*mxfp4PackedBytes {
		return fmt.Errorf("mxfp4: %d block bytes for %d blocks (want %d)", len(blocks), nBlocks, nBlocks*mxfp4PackedBytes)
	}
	if len(dst) != nBlocks*MXFP4BlockElems {
		return fmt.Errorf("mxfp4: dst has %d elements for %d blocks (want %d)", len(dst), nBlocks, nBlocks*MXFP4BlockElems)
	}
	for b := range nBlocks {
		d := e8m0ToF32Half(scales[b])
		qs := blocks[b*mxfp4PackedBytes : (b+1)*mxfp4PackedBytes]
		row := dst[b*MXFP4BlockElems : (b+1)*MXFP4BlockElems]
		for j := range mxfp4PackedBytes {
			v := qs[j]
			row[2*j] = d * float32(mxfp4KValues[v&0x0F])
			row[2*j+1] = d * float32(mxfp4KValues[v>>4])
		}
	}
	return nil
}
