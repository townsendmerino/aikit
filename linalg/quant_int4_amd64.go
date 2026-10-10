//go:build amd64

package linalg

// int4QuantBlock is the AVX2 kernel's block: 8 weights in, 4 packed bytes out.
const int4QuantBlock = 8

// quantizeInt4F32AVX2 writes, for the first n weights of row (n a multiple of 8, n > 0), the packed nibbles
// packed[i/2] = nib(row[i]) | nib(row[i+1])<<4 with nib(x) = clamp(roundInt4(x*inv), -7, 7) + 8, for every float32
// product, the three cases where amd64's conversion is not the obvious one included (quant_int4_amd64.s).
// Implemented in quant_int4_amd64.s.
//
//go:noescape
func quantizeInt4F32AVX2(row *float32, packed *byte, n int, inv float32)

// int4GroupHasKernel reports whether a group of n weights starting at weight ks goes through the kernel: it must start
// on a byte boundary and be a whole number of blocks.
func int4GroupHasKernel(ks, n int) bool {
	return ks&1 == 0 && n >= int4QuantBlock && n%int4QuantBlock == 0
}

// quantizeInt4Block is the kernel behind QuantizeGroupInt4Row's vector path. len(row) must be a positive multiple of
// int4QuantBlock and packed must hold len(row)/2 bytes; it overwrites all of them.
func quantizeInt4Block(row []float32, packed []byte, inv float32) {
	_ = packed[len(row)/2-1]
	quantizeInt4F32AVX2(&row[0], &packed[0], len(row), inv)
}
