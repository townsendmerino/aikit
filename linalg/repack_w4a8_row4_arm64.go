//go:build arm64

package linalg

import "fmt"

// docs/task-w4a8-neon-bandwidth.md's item-3+4 harness (GO, 2026-08-23/24) —
// promoted from harness-only test code to production once the grid recorded
// GO. Repacks canonical int4-packed weights (QuantizeGroupInt4Row's
// interleaved even/odd-nibble layout) into split-half + 4-row-interleaved:
// no change to quant.go's canonical packer, no .giw/WeightMat change here —
// RepackW4A8Row4/RepackW4A8Row4Scales (matmul_w4a8_row4_arm64.go) are the
// public entry points a caller's load-time repack should use; these are
// their per-row/per-quad internals.
//
// Why this layout: canonical packing interleaves k and k+1 per byte, so
// unpacking needs VZIP1/VZIP2 to restore sequential order before the
// nibbles can feed SDOT. Split-half packs k and k+16 per byte instead (low
// nibble = block 0, k_local 0..15; high nibble = block 1, k_local 16..31) —
// each half is ALREADY sequential once masked/shifted out, so the unpack
// prologue drops from {AND, SHR, ZIP1, ZIP2} to {AND, SHR}. The 4-row
// interleave then stacks 4 rows' split-half groups contiguously so one
// activation load is shared across all 4 rows' SDOT.

// canonicalNibble returns the unsigned nibble value (1..15, zero=8) at global
// position k in the canonical packed layout — the same extraction
// dotW4A8Scalar does inline, factored out so the repack and the scalar
// oracles share one definition of "what canonical packing means".
func canonicalNibble(packed []byte, k int) byte {
	b := packed[k/2]
	if k&1 == 0 {
		return b & 0x0F
	}
	return (b >> 4) & 0x0F
}

// repackSplitHalfRow converts one canonical-packed row (K a multiple of 32,
// matching dotW4A8's own fast-path contract) into the split-half layout.
// Same byte count as the input; only the bit arrangement changes.
func repackSplitHalfRow(packed []byte, K int) []byte {
	if K%32 != 0 {
		panic("repackSplitHalfRow: K must be a multiple of 32")
	}
	nGroups := K / 32
	out := make([]byte, len(packed))
	for g := range nGroups {
		gk := g * 32
		obase := g * 16
		for i := range 16 {
			lo := canonicalNibble(packed, gk+i)
			hi := canonicalNibble(packed, gk+i+16)
			out[obase+i] = lo | (hi << 4)
		}
	}
	return out
}

// splitHalfNibble is canonicalNibble's counterpart for the split-half layout.
func splitHalfNibble(packed []byte, k int) byte {
	g := k / 32
	kl := k % 32
	base := g * 16
	if kl < 16 {
		return packed[base+kl] & 0x0F
	}
	return (packed[base+(kl-16)] >> 4) & 0x0F
}

// repackSplitHalf4RowBlock interleaves 4 rows' split-half-packed data,
// group-contiguous: for each 32-wide K-group, the 4 rows' 16 split-half
// bytes are stored back-to-back (row0's 16, row1's 16, row2's 16, row3's
// 16), repeated per group — a "Q4_0_4x4-style" repack: unlike a per-byte
// interleave, this keeps each row's own unpack sequence intact (no new
// shuffle cost) while making the 4 rows' data for one group contiguous for
// streaming, and lets a kernel load one activation chunk per group and
// reuse it across all 4 rows' SDOT instead of reloading it per row.
func repackSplitHalf4RowBlock(row0, row1, row2, row3 []byte, K int) []byte {
	if K%32 != 0 {
		panic("repackSplitHalf4RowBlock: K must be a multiple of 32")
	}
	nGroups := K / 32
	bpr := K / 2
	sh0 := repackSplitHalfRow(row0, K)
	sh1 := repackSplitHalfRow(row1, K)
	sh2 := repackSplitHalfRow(row2, K)
	sh3 := repackSplitHalfRow(row3, K)
	out := make([]byte, 4*bpr)
	for g := range nGroups {
		base := g * 16
		obase := g * 64
		copy(out[obase:obase+16], sh0[base:base+16])
		copy(out[obase+16:obase+32], sh1[base:base+16])
		copy(out[obase+32:obase+48], sh2[base:base+16])
		copy(out[obase+48:obase+64], sh3[base:base+16])
	}
	return out
}

// repackSplitHalf4RowDeshared is repackSplitHalf4RowBlock's non-interleaved
// counterpart (docs/task-w4a8-neon-bandwidth.md's cold-fix harness pass,
// chain/line de-sharing remedy): same per-row split-half packing
// (repackSplitHalfRow), but the 4 rows are returned as 4 SEPARATE slices
// instead of interleaved into one contiguous block — de-sharing the cache
// line dotW4A8SplitHalf4RowDeshared's 4 concurrent accumulator chains would
// otherwise read from on a cold miss. Scales need no repacking at all here
// (the caller's existing per-row scale slices are already separate); this
// only exists for the weight bytes.
func repackSplitHalf4RowDeshared(row0, row1, row2, row3 []byte, K int) (b0, b1, b2, b3 []byte) {
	return repackSplitHalfRow(row0, K), repackSplitHalfRow(row1, K), repackSplitHalfRow(row2, K), repackSplitHalfRow(row3, K)
}

// interleaveScales4Row is repackSplitHalf4RowBlock's counterpart for the
// per-group f32 scales: 4 scales per group (row0..row3), contiguous, same
// locality reasoning.
func interleaveScales4Row(s0, s1, s2, s3 []float32, nGroups int) []float32 {
	out := make([]float32, 4*nGroups)
	for g := range nGroups {
		out[4*g] = s0[g]
		out[4*g+1] = s1[g]
		out[4*g+2] = s2[g]
		out[4*g+3] = s3[g]
	}
	return out
}

// RepackInt4Row4Quad writes ONE quad's worth of canonical int4 nibbles (src,
// 4*bpr bytes: row0's bpr canonical bytes, then row1's, row2's, row3's,
// contiguous — QuantizeGroupsInt4's own row-major layout) into dst (4*bpr
// bytes) in row4 order: group g's 64-byte block at dst[g*64:g*64+64], row
// r's 16 split-half bytes within it at dst[g*64+r*16:g*64+r*16+16] — the
// same layout repackSplitHalf4RowBlock produces, factored out here as a
// caller-supplied-buffer primitive (audit M-22) so a loader can drive it
// without repackSplitHalf4RowBlock's 5 per-quad allocations (4 per-row
// split-half temporaries + 1 output). RepackW4A8Row4 (the existing
// out-of-place entry point) and RepackInt4Row4InPlace both call this now;
// repackSplitHalf4RowBlock itself is untouched — several existing tests and
// repackSplitHalf4RowDeshared still exercise it directly.
//
// bpr = K/2 (K a multiple of 32, group=32 — Int4Row4Usable's own gate, the
// row4 kernel's only supported shape).
//
// dst and src MUST NOT OVERLAP. The row4 and canonical layouts are
// different permutations of the same bytes, so a same-buffer call would
// read a source byte AFTER an earlier group's write had already overwritten
// it — see RepackInt4Row4InPlace, which snapshots the live quad into
// scratch BEFORE calling this with dst=the live array, src=the scratch, so
// that hazard never arises.
func RepackInt4Row4Quad(dst, src []byte, K int) {
	if K%32 != 0 {
		panic(fmt.Sprintf("linalg: RepackInt4Row4Quad requires K a multiple of 32, got %d", K))
	}
	bpr := K / 2
	requireLen("RepackInt4Row4Quad", "dst", len(dst), 4*bpr)
	requireLen("RepackInt4Row4Quad", "src", len(src), 4*bpr)
	nGroups := K / 32
	for g := range nGroups {
		gk := g * 32
		obase := g * 64
		for r := range 4 {
			row := src[r*bpr : (r+1)*bpr]
			rbase := obase + r*16
			for i := range 16 {
				lo := canonicalNibble(row, gk+i)
				hi := canonicalNibble(row, gk+i+16)
				dst[rbase+i] = lo | (hi << 4)
			}
		}
	}
}

// RepackInt4Row4ScalesQuad is RepackInt4Row4Quad's counterpart for the
// per-group f32 scales: src is 4*nGroups floats (row0's nGroups canonical
// scales, then row1's, row2's, row3's), dst is 4*nGroups floats interleaved
// — group g's 4 scales (row0..row3) at dst[4*g:4*g+4] — matching
// RepackW4A8Row4Scales/interleaveScales4Row's existing layout. dst and src
// must not overlap, same reason as RepackInt4Row4Quad.
func RepackInt4Row4ScalesQuad(dst, src []float32, nGroups int) {
	requireLen("RepackInt4Row4ScalesQuad", "dst", len(dst), 4*nGroups)
	requireLen("RepackInt4Row4ScalesQuad", "src", len(src), 4*nGroups)
	for g := range nGroups {
		for r := range 4 {
			dst[4*g+r] = src[r*nGroups+g]
		}
	}
}
