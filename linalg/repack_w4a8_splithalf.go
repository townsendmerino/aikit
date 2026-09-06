package linalg

// RepackW4A8SplitHalf converts a [rows, cols] int4 weight matrix from the canonical packed
// layout into the SPLIT-HALF layout the AVX2 kernel dotW4A8SplitHalfAVX2 consumes.
//
// Canonical: byte i of a group holds weights 2i (low nibble) and 2i+1 (high) — the two halves of
// a group are interleaved, so a kernel must undo the interleave with two shuffle-port ops per
// group (VPUNPCKLBW/VPUNPCKHBW on AVX2).
//
// Split-half: byte i holds weight i (low) and weight i+16 (high), so one 16-byte load yields two
// contiguous 16-weight halves and the shuffles disappear. This is llama.cpp's core Q4 trick, and
// on AVX2 it measured 1.12x hot and cold at K=5120 (dot_w4a8_amd64.s records the numbers).
//
// SCALES ARE UNCHANGED. Split-half permutes nibbles WITHIN a group and never reorders groups, so
// the per-group scale array is bit-identical between layouts — unlike the 4-row-interleaved
// layout, which needs RepackW4A8Row4Scales. Callers pass the original scales straight through.
//
// Portable Go on purpose: it is a load-time O(K) pass per tensor, never on the token path, and
// keeping it architecture-neutral means the layout can be tested and reasoned about on any host
// even though only the AVX2 kernel currently reads it.
func RepackW4A8SplitHalf(packed []byte, rows, cols, group int) []byte {
	if group != 32 || cols%32 != 0 {
		panic("RepackW4A8SplitHalf: requires group=32 and cols a multiple of 32")
	}
	bpr := cols / 2
	out := make([]byte, len(packed))
	nib := func(row []byte, k int) byte {
		b := row[k/2]
		if k%2 == 0 {
			return b & 0x0F
		}
		return b >> 4
	}
	for r := range rows {
		src := packed[r*bpr : (r+1)*bpr]
		dst := out[r*bpr : (r+1)*bpr]
		for g := 0; g < cols/32; g++ {
			gk, ob := g*32, g*16
			for i := range 16 {
				dst[ob+i] = nib(src, gk+i) | (nib(src, gk+i+16) << 4)
			}
		}
	}
	return out
}
