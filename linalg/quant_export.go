package linalg

// DotI8 returns the int32 dot product of two equal-length int8 vectors via the
// platform SIMD kernel (AVX2 VPMADDWD on amd64, SDOT on ARMv8.2, scalar fallback
// elsewhere). The caller must pass len(a) == len(b); it is the node-node integer
// similarity primitive for the int8 ann indexes (rescale by the two per-vector
// scales to recover cosine). Experimental, like the rest of linalg.
// (QuantizeRowInt8, the single-row f32→int8 quantizer for a query vector, is
// already exported in quant.go.)
func DotI8(a, b []int8) int32 { return dotI8(a, b) }
