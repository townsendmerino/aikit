package linalg

// This file has no build tag: q4Row4/q4Row4Scales/q4SplitHalf are ordinary
// WeightMat fields on every architecture (nil where the platform has no
// repacked layout), and the gather below is pure Go arithmetic over bytes —
// no SIMD, no build-tag-gated kernel. Row() calls these directly regardless
// of GOARCH; they simply never run on a platform that never populates the
// fields they read.

// dequantizeRowFromRow4 dequantizes row i's cols elements from the row4
// (split-half + 4-row-interleaved) layout — audit M-22. Same reconstruction
// as DequantizeRowInt4 (nibble-8)*scale, over the SAME nibbles in a
// different byte arrangement: repackSplitHalf4RowBlock (repack_w4a8_row4_
// arm64.go) packs row r of quad q's group g into 16 bytes at offset
// q*4*bpr + g*64 + r*16 within q4Row4, low nibble = k_local 0..15, high
// nibble = k_local 16..31 (the same split-half rule repackSplitHalfRow uses
// per row, just quad-interleaved) — a strided GATHER of the same bytes
// canonical Row() reads, not a different set of bytes. group is fixed at 32:
// the row4 kernel's own contract (RepackInt4Row4 declines any other size),
// asserted rather than assumed by the caller passing it through.
//
// Raw-bit-identical to canonical Row() for the same logical weights —
// TestWeightMatRow_row4MatchesCanonical.
func dequantizeRowFromRow4(q4Row4 []byte, q4Row4Scales []float32, cols, i int, dst []float32) {
	const groupSize = 32
	nGroups := (cols + groupSize - 1) / groupSize
	bpr := (cols + 1) / 2
	q, r := i/4, i%4
	quadBase := q * 4 * bpr
	scaleQuadBase := q * 4 * nGroups
	for g := 0; g < nGroups; g++ {
		s := q4Row4Scales[scaleQuadBase+4*g+r]
		gk := g * groupSize
		end := min(gk+groupSize, cols)
		chunk := q4Row4[quadBase+g*64+r*16 : quadBase+g*64+r*16+16]
		for k := gk; k < end; k++ {
			kl := k - gk
			var nib byte
			if kl < 16 {
				nib = chunk[kl] & 0x0F
			} else {
				nib = chunk[kl-16] >> 4
			}
			dst[k] = float32(int(nib)-8) * s
		}
	}
}

// dequantizeRowFromSplitHalf dequantizes row i's cols elements from the
// amd64 split-half layout (RepackW4A8SplitHalf — no row interleave, so
// unlike row4 this is one row's OWN bpr-byte slice, byte-arrangement-only
// changed within it) — audit M-22. scales is q4s, shared with canonical
// (split-half permutes nibbles within a group and never reorders groups, so
// the per-row/per-group scale array means the same thing under either
// layout — the same sharing WeightMat's own field comment documents for the
// "both" case). group fixed at 32, RepackInt4SplitHalf's own contract.
//
// Raw-bit-identical to canonical Row() for the same logical weights —
// TestWeightMatRow_splitHalfMatchesCanonical.
func dequantizeRowFromSplitHalf(q4SplitHalf []byte, scales []float32, cols, i int, dst []float32) {
	const groupSize = 32
	nGroups := (cols + groupSize - 1) / groupSize
	bpr := (cols + 1) / 2
	row := q4SplitHalf[i*bpr : (i+1)*bpr]
	srow := scales[i*nGroups : (i+1)*nGroups]
	for g := 0; g < nGroups; g++ {
		s := srow[g]
		gk := g * groupSize
		end := min(gk+groupSize, cols)
		ob := g * 16
		for k := gk; k < end; k++ {
			kl := k - gk
			var nib byte
			if kl < 16 {
				nib = row[ob+kl] & 0x0F
			} else {
				nib = row[ob+kl-16] >> 4
			}
			dst[k] = float32(int(nib)-8) * s
		}
	}
}
