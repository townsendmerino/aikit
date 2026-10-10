//go:build arm64

package linalg

// int4QuantBlock is the NEON kernel's block: 16 weights in, 8 packed bytes out.
const int4QuantBlock = 16

// quantizeInt4F32NEON writes, for the first n weights of row (n a multiple of 16, n > 0), the packed nibbles
// packed[i/2] = nib(row[i]) | nib(row[i+1])<<4 with nib(x) = clamp(roundTiesAway(x*inv), -7, 7) + 8: one float32
// multiply, FCVTAS, an int32 clamp, the bias, then a narrow and a shift-accumulate that folds each pair of nibbles
// into a byte. FCVTAS gives 0 for NaN and saturates toward the sign of an infinity or an out-of-range value, which the
// clamp then covers: the values the scalar's FRINTA-and-convert gives on arm64. Implemented in quant_int4_arm64.s.
//
//go:noescape
func quantizeInt4F32NEON(row *float32, packed *byte, n int, inv float32)

// int4GroupHasKernel reports whether a group of n weights starting at weight ks goes through the kernel: it must start
// on a byte boundary and be a whole number of blocks.
func int4GroupHasKernel(ks, n int) bool {
	return ks&1 == 0 && n >= int4QuantBlock && n%int4QuantBlock == 0
}

// quantizeInt4Block is the kernel behind QuantizeGroupInt4Row's vector path. len(row) must be a positive multiple of
// int4QuantBlock and packed must hold len(row)/2 bytes; it overwrites all of them.
func quantizeInt4Block(row []float32, packed []byte, inv float32) {
	_ = packed[len(row)/2-1]
	quantizeInt4F32NEON(&row[0], &packed[0], len(row), inv)
}
