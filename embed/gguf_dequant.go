package embed

import (
	"encoding/binary"
	"math"
)

// GGML block dequantizers — one function per quantization type, moved out of
// gguf.go (which keeps the format-parsing and accessor surface) to keep each
// file's scope narrow, mirroring the ann/ convention of splitting one type's
// concerns across sibling files (flat_i8.go vs flat_i8_mmap.go vs
// flat_i8_persist.go) rather than one file per type.

// kvaluesIQ4NL is the 16-entry non-linear codebook shared by IQ4_NL and IQ4_XS:
// a 4-bit code indexes one of these int8 levels (ggml's kvalues_iq4nl), scaled by
// the block's f16 (and, for IQ4_XS, per-sub-block) scale. Unlike the linear
// Q4_* quants there is no `code-8` recentering — the codebook is the mapping.
var kvaluesIQ4NL = [16]int8{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}

// dequantRange dequantizes the block-aligned element range [start, start+len(dst))
// of a tensor's raw bytes into dst, dispatching to the per-block kernels. blockElems is the
// type's block size (from ggmlBlockElems). The caller validates alignment.
func dequantRange(typ uint32, raw []byte, start int, dst []float32, blockElems int) {
	switch typ {
	case ggmlTypeF32:
		for i := range dst {
			dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*(start+i):]))
		}
	case ggmlTypeF16:
		for i := range dst {
			dst[i] = halfBitsToF32(binary.LittleEndian.Uint16(raw[2*(start+i):]))
		}
	default:
		first := start / blockElems
		for k := 0; k*blockElems < len(dst); k++ {
			out := dst[k*blockElems : (k+1)*blockElems]
			switch typ {
			case ggmlTypeQ8_0:
				dequantQ8_0Block(raw, first+k, out)
			case ggmlTypeQ4_0:
				dequantQ4_0Block(raw, first+k, out)
			case ggmlTypeQ5_0:
				dequantQ5_0Block(raw, first+k, out)
			case ggmlTypeQ2_K:
				dequantQ2KBlock(raw, first+k, out)
			case ggmlTypeQ3_K:
				dequantQ3KBlock(raw, first+k, out)
			case ggmlTypeQ4_K:
				dequantQ4KBlock(raw, first+k, out)
			case ggmlTypeQ5_K:
				dequantQ5KBlock(raw, first+k, out)
			case ggmlTypeQ6_K:
				dequantQ6KBlock(raw, first+k, out)
			case ggmlTypeIQ4NL:
				dequantIQ4NLBlock(raw, first+k, out)
			case ggmlTypeMXFP4:
				dequantMXFP4Block(raw, first+k, out)
			case ggmlTypeIQ4XS:
				dequantIQ4XSBlock(raw, first+k, out)
			case ggmlTypeIQ2S:
				dequantIQ2SBlock(raw, first+k, out)
			case ggmlTypeIQ3S:
				dequantIQ3SBlock(raw, first+k, out)
			}
		}
	}
}

// dequantQ8_0Block dequantizes one 32-element Q8_0 block (b) into out[:32]: a
// f16 scale d then 32 int8 q; value = d*q.
func dequantQ8_0Block(raw []byte, blk int, out []float32) {
	base := blk * 34
	d := halfBitsToF32(binary.LittleEndian.Uint16(raw[base:]))
	qs := raw[base+2 : base+34]
	for i := range 32 {
		out[i] = float32(int8(qs[i])) * d
	}
}

// dequantQ4_0Block dequantizes one 32-element Q4_0 block (b) into out[:32]: a f16
// scale d then 16 packed bytes; low nibble of byte i is element i, high nibble is
// element i+16, each recentered by -8: value = d*(nibble-8).
func dequantQ4_0Block(raw []byte, blk int, out []float32) {
	base := blk * 18
	d := halfBitsToF32(binary.LittleEndian.Uint16(raw[base:]))
	qs := raw[base+2 : base+18]
	for i := range 16 {
		v := qs[i]
		out[i] = float32(int(v&0x0F)-8) * d
		out[i+16] = float32(int(v>>4)-8) * d
	}
}

// dequantQ5_0Block dequantizes one 32-element Q5_0 block (b) into out[:32]: a f16
// scale d, a 4-byte qh carrying each element's 5th (high) bit, then 16 packed low
// nibbles. For element j the code is (low nibble | high bit << 4) ∈ [0,31],
// recentered by -16: value = d*(code-16). Mirrors ggml's dequantize_row_q5_0.
func dequantQ5_0Block(raw []byte, blk int, out []float32) {
	base := blk * 22
	d := halfBitsToF32(binary.LittleEndian.Uint16(raw[base:]))
	qh := binary.LittleEndian.Uint32(raw[base+2:])
	qs := raw[base+6 : base+22]
	for j := range 16 {
		xh0 := byte(((qh >> uint(j)) << 4) & 0x10) // bit j → bit 4
		xh1 := byte((qh >> uint(j+12)) & 0x10)     // bit j+16 → bit 4
		q0 := int32((qs[j]&0x0F)|xh0) - 16
		q1 := int32((qs[j]>>4)|xh1) - 16
		out[j] = float32(q0) * d
		out[j+16] = float32(q1) * d
	}
}

// dequantQ6KBlock dequantizes one 256-element Q6_K super-block (sb) into out[:256].
// Layout (210 bytes): ql[128] (low 4 bits), qh[64] (high 2 bits), scales[16]
// (int8), d (f16 super-scale). Mirrors ggml's dequantize_row_q6_K: a 6-bit quant
// q∈[0,63] recentered by -32, scaled by its sub-block int8 scale and d.
func dequantQ6KBlock(raw []byte, sb int, out []float32) {
	base := sb * 210
	ql := raw[base : base+128]
	qh := raw[base+128 : base+192]
	sc := raw[base+192 : base+208] // int8 scales
	d := halfBitsToF32(binary.LittleEndian.Uint16(raw[base+208:]))
	for chunk := range 2 {
		n0 := chunk * 128
		qlo := ql[chunk*64:]
		qho := qh[chunk*32:]
		sco := sc[chunk*8:]
		for l := range 32 {
			is := l / 16
			q1 := int8((qlo[l]&0x0F)|(((qho[l]>>0)&3)<<4)) - 32
			q2 := int8((qlo[l+32]&0x0F)|(((qho[l]>>2)&3)<<4)) - 32
			q3 := int8((qlo[l]>>4)|(((qho[l]>>4)&3)<<4)) - 32
			q4 := int8((qlo[l+32]>>4)|(((qho[l]>>6)&3)<<4)) - 32
			out[n0+l+0] = d * float32(int8(sco[is+0])) * float32(q1)
			out[n0+l+32] = d * float32(int8(sco[is+2])) * float32(q2)
			out[n0+l+64] = d * float32(int8(sco[is+4])) * float32(q3)
			out[n0+l+96] = d * float32(int8(sco[is+6])) * float32(q4)
		}
	}
}

// dequantIQ4NLBlock dequantizes one 32-element IQ4_NL block (b) into out[:32].
// Layout (18 bytes, same size as Q4_0): a f16 scale d then 16 packed bytes; the
// low nibble of byte j indexes element j and the high nibble element j+16 — each
// nibble looked up in the kvaluesIQ4NL codebook: value = d·kvaluesIQ4NL[code].
func dequantIQ4NLBlock(raw []byte, blk int, out []float32) {
	base := blk * 18
	d := halfBitsToF32(binary.LittleEndian.Uint16(raw[base:]))
	qs := raw[base+2 : base+18]
	for j := range 16 {
		out[j] = d * float32(kvaluesIQ4NL[qs[j]&0x0F])
		out[j+16] = d * float32(kvaluesIQ4NL[qs[j]>>4])
	}
}

// mxfp4KValues is the e2m1 dequant table (the 16 representable FP4 values, doubled
// so the lookup stays integer — paired with the HALF-scale e8m0 below, d·half ·
// value·2 recovers the true product). ref: gguf/quants.py MXFP4.kvalues; OCP MX v1.0.
var mxfp4KValues = [16]int8{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}

// e8m0ToF32Half converts an 8-bit e8m0 block scale to float32, matching ggml's
// ggml_e8m0_to_fp32_half (gguf/quants.py MXFP4.e8m0_to_fp32_half): x<2 is a
// subnormal bit pattern, else the exponent field is x-1.
//
// The bit formula is used because it is exact by construction and needs no libm
// call — NOT because 2^(x-128) would be wrong. This comment used to say the
// subnormals at x∈{0,1} depend on it; measured 2026-09-06 over all 256 inputs,
// float32(math.Pow(2, x-128)) is bit-identical everywhere, because 2^-128 and
// 2^-127 are exactly representable as float32 subnormals (which reach 2^-149).
// The old wording would have told anyone simplifying this that they had broken
// something they had not, which is worse than saying nothing.
func e8m0ToF32Half(x uint8) float32 {
	var bits uint32
	if x < 2 {
		bits = uint32(0x00200000) << uint32(x)
	} else {
		bits = uint32(x-1) << 23
	}
	return math.Float32frombits(bits)
}

// dequantMXFP4Block dequantizes one 17-byte MXFP4 (OCP FP4, ggml type 39) block into
// out[:32]: byte 0 is the e8m0 power-of-two block scale, bytes 1..16 each pack two
// e2m1 4-bit values (element j in the low nibble, j+16 in the high nibble — matching
// gguf/quants.py's reshape((n,2,16)) split). Used by gpt-oss's MXFP4 expert weights.
func dequantMXFP4Block(raw []byte, blk int, out []float32) {
	base := blk * 17
	d := e8m0ToF32Half(raw[base])
	qs := raw[base+1 : base+17]
	for j := range 16 {
		out[j] = d * float32(mxfp4KValues[qs[j]&0x0F])
		out[j+16] = d * float32(mxfp4KValues[qs[j]>>4])
	}
}

// dequantIQ4XSBlock dequantizes one 256-element IQ4_XS super-block (sb) into
// out[:256]. Layout (136 bytes): d (f16 super-scale), scales_h (u16), scales_l[4],
// qs[128]. The super-block splits into eight 32-element sub-blocks; sub-block ib
// has a 6-bit scale ls assembled from scales_l (low 4 bits) and scales_h (high 2
// bits), recentered by −32: dl = d·(ls−32). Each nibble of qs indexes the shared
// kvaluesIQ4NL codebook (low → element j, high → j+16 within the sub-block).
func dequantIQ4XSBlock(raw []byte, sb int, out []float32) {
	base := sb * 136
	d := halfBitsToF32(binary.LittleEndian.Uint16(raw[base:]))
	scalesH := binary.LittleEndian.Uint16(raw[base+2:])
	scalesL := raw[base+4 : base+8]
	qs := raw[base+8 : base+136]
	for ib := range 8 { // eight 32-element sub-blocks
		ls := int((scalesL[ib/2]>>(4*(ib%2)))&0x0F) | int((scalesH>>(2*ib))&3)<<4
		dl := d * float32(ls-32)
		q := qs[ib*16 : ib*16+16]
		o := out[ib*32 : ib*32+32]
		for j := range 16 {
			o[j] = dl * float32(kvaluesIQ4NL[q[j]&0x0F])
			o[j+16] = dl * float32(kvaluesIQ4NL[q[j]>>4])
		}
	}
}

// dequantIQ2SBlock dequantizes one 256-element IQ2_S super-block (sb) into
// out[:256]. Layout (82 bytes): d (f16), qs[32] (low 8 bits of each 8-wide grid
// index), signs[32] (per-element sign bits), qh[8] (high 2 bits of each index),
// scales[8] (4-bit sub-scales). The super-block is 16 sub-blocks of 16, each a
// 4-bit scale and two 8-wide codebook lookups (iq2sGrid, 1024×8); the per-element
// sign comes from the packed sign bits. Mirrors ggml's dequantize_row_iq2_s.
func dequantIQ2SBlock(raw []byte, sb int, out []float32) {
	base := sb * 82
	d := halfBitsToF32(binary.LittleEndian.Uint16(raw[base:]))
	qs := raw[base+2 : base+34]
	signs := raw[base+34 : base+66]
	qh := raw[base+66 : base+74]
	scales := raw[base+74 : base+82]
	for sub := range 16 {
		sc := int((scales[sub/2] >> (4 * (sub & 1))) & 0x0F)
		db := d * (0.5 + float32(sc)) * 0.25
		for pair := range 2 {
			k := sub*2 + pair
			idx := int(qs[k]) | int((qh[k/4]>>(2*(k&3)))&3)<<8
			g := iq2sGrid[idx*8 : idx*8+8]
			sg := signs[k]
			o := out[sub*16+pair*8:]
			for j := range 8 {
				v := db * float32(g[j])
				if (sg>>j)&1 != 0 {
					v = -v
				}
				o[j] = v
			}
		}
	}
}

// dequantIQ3SBlock dequantizes one 256-element IQ3_S super-block (sb) into
// out[:256]. Layout (110 bytes): d (f16), qs[64] (low 8 bits of each 4-wide grid
// index), qh[8] (1 high bit per index), signs[32] (per-element sign bits),
// scales[4] (4-bit sub-scales). 8 sub-blocks of 32, each a 4-bit scale and eight
// 4-wide codebook lookups (iq3sGrid, 512×4); the sign comes from the packed sign
// bits. Mirrors ggml's dequantize_row_iq3_s.
func dequantIQ3SBlock(raw []byte, sb int, out []float32) {
	base := sb * 110
	d := halfBitsToF32(binary.LittleEndian.Uint16(raw[base:]))
	qs := raw[base+2 : base+66]
	qh := raw[base+66 : base+74]
	signs := raw[base+74 : base+106]
	scales := raw[base+106 : base+110]
	for p := range 256 {
		sub := p / 32
		sc := int((scales[sub/2] >> (4 * (sub & 1))) & 0x0F)
		db := d * float32(1+2*sc)
		m := p / 4 // grid-index number (0..63)
		idx := int(qs[m]) | int((qh[m/8]>>(m&7))&1)<<8
		v := db * float32(iq3sGrid[idx*4+p%4])
		if (signs[p/8]>>(p&7))&1 != 0 {
			v = -v
		}
		out[p] = v
	}
}

// dequantQ4KBlock dequantizes one 256-element Q4_K super-block (sb) into out[:256].
// Layout (144 bytes): d (f16), dmin (f16), scales[12] (6-bit scales+mins, packed),
// qs[128] (4-bit quants). Mirrors ggml's dequantize_row_q4_K: y = d·scale·q −
// dmin·min, with scale/min unpacked by get_scale_min_k4.
func dequantQ4KBlock(raw []byte, sb int, out []float32) {
	base := sb * 144
	d := halfBitsToF32(binary.LittleEndian.Uint16(raw[base:]))
	dmin := halfBitsToF32(binary.LittleEndian.Uint16(raw[base+2:]))
	scales := raw[base+4 : base+16]
	qs := raw[base+16 : base+144]
	yi := 0
	for j := range 4 { // four 64-element groups
		is := 2 * j
		sc1, m1 := q4kScaleMin(is+0, scales)
		sc2, m2 := q4kScaleMin(is+1, scales)
		d1, off1 := d*float32(sc1), dmin*float32(m1)
		d2, off2 := d*float32(sc2), dmin*float32(m2)
		q := qs[j*32 : j*32+32]
		for l := range 32 {
			out[yi] = d1*float32(q[l]&0x0F) - off1
			yi++
		}
		for l := range 32 {
			out[yi] = d2*float32(q[l]>>4) - off2
			yi++
		}
	}
}

// q4kScaleMin unpacks the j-th 6-bit scale and min from a Q4_K/Q5_K super-block's
// 12-byte scales array (ggml's get_scale_min_k4).
func q4kScaleMin(j int, q []byte) (scale, min uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	scale = (q[j+4] & 0x0F) | ((q[j-4] >> 6) << 4)
	min = (q[j+4] >> 4) | ((q[j] >> 6) << 4)
	return scale, min
}

// dequantQ5KBlock dequantizes one 256-element Q5_K super-block (sb) into out[:256].
// Layout (176 bytes): d (f16), dmin (f16), scales[12] (6-bit scales+mins, same
// packing as Q4_K), qh[32] (the 5th/high bit per element), qs[128] (low 4 bits).
// Mirrors ggml's dequantize_row_q5_K: y = d·sc·q − dmin·m with q a 5-bit code
// (low nibble | high bit << 4). The high bit for each 32-wide half is selected by
// a mask that walks the qh byte two bits at a time across the four 64-elem groups.
func dequantQ5KBlock(raw []byte, sb int, out []float32) {
	base := sb * 176
	d := halfBitsToF32(binary.LittleEndian.Uint16(raw[base:]))
	dmin := halfBitsToF32(binary.LittleEndian.Uint16(raw[base+2:]))
	scales := raw[base+4 : base+16]
	qh := raw[base+16 : base+48]
	qs := raw[base+48 : base+176]
	yi := 0
	u1, u2 := byte(1), byte(2)
	for j := range 4 { // four 64-element groups
		is := 2 * j
		sc1, m1 := q4kScaleMin(is+0, scales)
		sc2, m2 := q4kScaleMin(is+1, scales)
		d1, off1 := d*float32(sc1), dmin*float32(m1)
		d2, off2 := d*float32(sc2), dmin*float32(m2)
		ql := qs[j*32 : j*32+32]
		for l := range 32 {
			var h float32
			if qh[l]&u1 != 0 {
				h = 16
			}
			out[yi] = d1*(float32(ql[l]&0x0F)+h) - off1
			yi++
		}
		for l := range 32 {
			var h float32
			if qh[l]&u2 != 0 {
				h = 16
			}
			out[yi] = d2*(float32(ql[l]>>4)+h) - off2
			yi++
		}
		u1 <<= 2
		u2 <<= 2
	}
}

// dequantQ2KBlock dequantizes one 256-element Q2_K super-block (sb) into out[:256].
// Layout (84 bytes): scales[16] (each byte a 4-bit scale in the low nibble and a
// 4-bit min in the high nibble), qs[64] (2-bit quants), d (f16 super-scale), dmin
// (f16 super-min). Mirrors ggml's dequantize_row_q2_K: y = d·scale·q2 − dmin·min,
// q2 the 2-bit code. No high-bit mask (unlike Q3_K) — the coarsest K-quant.
func dequantQ2KBlock(raw []byte, sb int, out []float32) {
	base := sb * 84
	scales := raw[base : base+16]
	qs := raw[base+16 : base+80]
	d := halfBitsToF32(binary.LittleEndian.Uint16(raw[base+80:]))
	dmin := halfBitsToF32(binary.LittleEndian.Uint16(raw[base+82:]))

	yi, is := 0, 0
	for n := range 2 { // two 128-element halves
		qb := n * 32 // qs advances by 32 each half
		shift := uint(0)
		for range 4 {
			sc := scales[is]
			is++
			dl, ml := d*float32(sc&0x0F), dmin*float32(sc>>4)
			for l := range 16 {
				out[yi] = dl*float32((qs[qb+l]>>shift)&3) - ml
				yi++
			}
			sc = scales[is]
			is++
			dl, ml = d*float32(sc&0x0F), dmin*float32(sc>>4)
			for l := range 16 {
				out[yi] = dl*float32((qs[qb+l+16]>>shift)&3) - ml
				yi++
			}
			shift += 2
		}
	}
}

// dequantQ3KBlock dequantizes one 256-element Q3_K super-block (sb) into out[:256].
// Layout (110 bytes): hmask[32] (the 3rd/high bit per element), qs[64] (low 2
// bits), scales[12] (16 six-bit sub-block scales, bit-packed), d (f16). Mirrors
// ggml's dequantize_row_q3_K: the 12 scale bytes are unpacked (the aux dance) into
// 16 int8 scales recentered by −32, and each element is a 2-bit code lifted to
// [−4,3] by the hmask bit: y = d·scale·(q2 − (hmask_bit ? 0 : 4)).
func dequantQ3KBlock(raw []byte, sb int, out []float32) {
	const (
		kmask1 = 0x03030303
		kmask2 = 0x0f0f0f0f
	)
	base := sb * 110
	hm := raw[base : base+32]
	q := raw[base+32 : base+96]
	scRaw := raw[base+96 : base+108]
	dAll := halfBitsToF32(binary.LittleEndian.Uint16(raw[base+108:]))

	// Unpack the 16 six-bit sub-block scales: the 12 packed bytes are read as three
	// little-endian uint32s, recombined (ggml's bit dance), and laid back down as a
	// 16-byte int8 buffer. Each scale is recentered by −32 at use.
	a0 := binary.LittleEndian.Uint32(scRaw[0:])
	a1 := binary.LittleEndian.Uint32(scRaw[4:])
	tmp := binary.LittleEndian.Uint32(scRaw[8:])
	scaleWords := [4]uint32{
		(a0 & kmask2) | (((tmp >> 0) & kmask1) << 4),
		(a1 & kmask2) | (((tmp >> 2) & kmask1) << 4),
		((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4),
		((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4),
	}
	var sc [16]int8
	for i, v := range scaleWords {
		sc[4*i+0] = int8(v)
		sc[4*i+1] = int8(v >> 8)
		sc[4*i+2] = int8(v >> 16)
		sc[4*i+3] = int8(v >> 24)
	}

	yi, is := 0, 0
	m := byte(1)
	for n := range 2 { // two 128-element halves
		qb := n * 32 // q advances by 32 each half
		shift := uint(0)
		for range 4 {
			dl := dAll * float32(int(sc[is])-32)
			is++
			for l := range 16 {
				var sub float32 = 4
				if hm[l]&m != 0 {
					sub = 0
				}
				out[yi] = dl * (float32((q[qb+l]>>shift)&3) - sub)
				yi++
			}
			dl = dAll * float32(int(sc[is])-32)
			is++
			for l := range 16 {
				var sub float32 = 4
				if hm[l+16]&m != 0 {
					sub = 0
				}
				out[yi] = dl * (float32((q[qb+l+16]>>shift)&3) - sub)
				yi++
			}
			shift += 2
			m <<= 1
		}
	}
}
