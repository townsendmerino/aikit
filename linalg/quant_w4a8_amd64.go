//go:build amd64

package linalg

// dotW4A8FoldAVX2 returns the per-group-scaled f32 dot Σ_g scale[g]·(act·w)_g of
// one int4 weight row against the int8 activation row, via the fused AVX2 kernel
// in dot_w4a8_amd64.s. The f32 weight scales are folded IN-REGISTER (convert +
// FMA into an 8-lane accumulator, one reduce at the end) — no per-group int32
// scratch and no Go-side fold loop. Only safe when hasAVX2. group is fixed at 32;
// nGroups = K/32.
//
//go:noescape
func dotW4A8FoldAVX2(act *int8, packed *byte, scales *float32, nGroups int) float32

// dotW4A8 computes one W4A8 output (before the activation scale). The
// AVX-512 VNNI and AVX2 paths both fold the per-group weight scales inside
// the kernel and return the f32 dot directly; only a ragged final group
// (K % 32 ≠ 0) is mopped up in Go. Everything off the fast path falls back
// to the portable reference.
func dotW4A8(act []int8, packed []byte, scales []float32, group, K int) float32 {
	if group == 32 && K >= 32 {
		nFull := K / 32
		var total float32
		switch {
		case hasAVX512VNNIVL:
			total = dotW4A8FoldAVX512VNNI(&act[0], &packed[0], &scales[0], nFull)
		case hasAVX2:
			total = dotW4A8FoldAVX2(&act[0], &packed[0], &scales[0], nFull)
		default:
			return dotW4A8Scalar(act, packed, scales, group, K)
		}
		if done := nFull * 32; done < K {
			// Ragged final group (K not a multiple of 32): scalar, scales[nFull].
			var acc int32
			for k := done; k < K; k++ {
				b := packed[k>>1]
				nib := b & 0x0F
				if k&1 == 1 {
					nib = b >> 4
				}
				acc += int32(act[k]) * int32(int(nib)-8)
			}
			total += float32(acc) * scales[nFull]
		}
		return total
	}
	return dotW4A8Scalar(act, packed, scales, group, K)
}

// dotW4A8Fold2AccAVX2 is dotW4A8FoldAVX2 with the f32 fold split across two independent
// accumulator chains — a probe for whether the serial VFMADD231PS fold, rather than
// instruction count, is the AVX2 kernel's real ceiling (see dot_w4a8_amd64.s for the
// motivating measurement, and dotW4A8FoldSDOT2Acc for the arm64 case where the same
// change measured a real 1.4-1.75x).
//
// nGroups must be EVEN. NOT wired into dotW4A8's dispatch: the two chains sum groups in a
// different order, so the f32 result may differ from dotW4A8FoldAVX2's in the last ulp,
// and this repo's callers are bit-identity-gated. Harness-only until an A/B funds the
// switch-over and a parity decision is made explicitly.
//
//go:noescape
func dotW4A8Fold2AccAVX2(act *int8, packed *byte, scales *float32, nGroups int) float32

// dotW4A8SplitHalfAVX2 is dotW4A8FoldAVX2 over the split-half weight layout (byte i of a group
// carries weight i low, weight i+16 high), which deletes the two VPUNPCK shuffles per group.
// Item 3 of goinfer's task-w4a8-neon-bandwidth.md, ported to AVX2 — see dot_w4a8_amd64.s for why
// the shuffle port, not the accumulator chain, is the AVX2-specific candidate.
//
// packed MUST already be in split-half layout. NOT wired into dotW4A8's dispatch: the canonical
// packer and the .giw kind=3 zero-copy load path produce the interleaved layout, and changing
// those would silently misdecode existing bundles.
//
//go:noescape
func dotW4A8SplitHalfAVX2(act *int8, packed *byte, scales *float32, nGroups int) float32

// dotW4A8SplitHalfScaledAVX2 is dotW4A8SplitHalfAVX2 with a per-group activation scale multiplied into
// each group's scale (dot_w4a8_amd64.s): per-32 activation quantization (actgroup.go).
//
//go:noescape
func dotW4A8SplitHalfScaledAVX2(act *int8, packed *byte, scales *float32, aScales *float32, nGroups int) float32
