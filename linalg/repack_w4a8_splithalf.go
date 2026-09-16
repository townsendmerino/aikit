package linalg

import "fmt"

// RepackInt4SplitHalfRow writes ONE row's canonical int4 nibbles (src, bpr =
// cols/2 bytes) into dst (bpr bytes) in split-half order — byte i holds
// weight i (low nibble) and weight i+16 (high), within each 32-wide group;
// see RepackW4A8SplitHalf's own comment for why. Factored out (audit M-22)
// so RepackW4A8SplitHalf (below) and RepackInt4SplitHalfInPlace
// (weightmat.go) share one row transform.
//
// dst and src MUST NOT OVERLAP. RepackInt4SplitHalfInPlace gets this safely
// by snapshotting the live row into scratch before calling this with
// dst=the live array, src=the scratch — same hazard and same fix as
// RepackInt4Row4Quad's (the row4 arm64 twin), just at row granularity
// instead of quad, since split-half does not interleave rows.
// repackSplitHalfGroup repacks 16 bytes (32 canonical nibbles) into split-half order.
// Both dg and sg must have len >= 16.
func repackSplitHalfGroup(dg, sg []byte) {
	_ = sg[15]
	_ = dg[15]

	s0, s8 := sg[0], sg[8]
	dg[0] = (s0 & 0x0F) | ((s8 & 0x0F) << 4)
	dg[1] = (s0 >> 4) | (s8 & 0xF0)

	s1, s9 := sg[1], sg[9]
	dg[2] = (s1 & 0x0F) | ((s9 & 0x0F) << 4)
	dg[3] = (s1 >> 4) | (s9 & 0xF0)

	s2, s10 := sg[2], sg[10]
	dg[4] = (s2 & 0x0F) | ((s10 & 0x0F) << 4)
	dg[5] = (s2 >> 4) | (s10 & 0xF0)

	s3, s11 := sg[3], sg[11]
	dg[6] = (s3 & 0x0F) | ((s11 & 0x0F) << 4)
	dg[7] = (s3 >> 4) | (s11 & 0xF0)

	s4, s12 := sg[4], sg[12]
	dg[8] = (s4 & 0x0F) | ((s12 & 0x0F) << 4)
	dg[9] = (s4 >> 4) | (s12 & 0xF0)

	s5, s13 := sg[5], sg[13]
	dg[10] = (s5 & 0x0F) | ((s13 & 0x0F) << 4)
	dg[11] = (s5 >> 4) | (s13 & 0xF0)

	s6, s14 := sg[6], sg[14]
	dg[12] = (s6 & 0x0F) | ((s14 & 0x0F) << 4)
	dg[13] = (s6 >> 4) | (s14 & 0xF0)

	s7, s15 := sg[7], sg[15]
	dg[14] = (s7 & 0x0F) | ((s15 & 0x0F) << 4)
	dg[15] = (s7 >> 4) | (s15 & 0xF0)
}

func RepackInt4SplitHalfRow(dst, src []byte, K int) {
	if K%32 != 0 {
		panic(fmt.Sprintf("linalg: RepackInt4SplitHalfRow requires K a multiple of 32, got %d", K))
	}
	bpr := K / 2
	requireLen("RepackInt4SplitHalfRow", "dst", len(dst), bpr)
	requireLen("RepackInt4SplitHalfRow", "src", len(src), bpr)
	nGroups := K / 32
	for g := 0; g < nGroups; g++ {
		ob := g * 16
		repackSplitHalfGroup(dst[ob:ob+16], src[ob:ob+16])
	}
}

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
	for r := range rows {
		RepackInt4SplitHalfRow(out[r*bpr:(r+1)*bpr], packed[r*bpr:(r+1)*bpr], cols)
	}
	return out
}
