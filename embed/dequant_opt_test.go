package embed

// Bit-identity tests for the branchless-sign / direct-index / BCE optimizations
// applied to dequantIQ2SBlock, dequantIQ3SBlock, dequantQ5KBlock, dequantQ2KBlock,
// and dequantQ3KBlock. Same pattern as q6k_hoist_test.go's audit M-24 gate: each
// *PreOpt function below is the exact pre-change body, and the test compares
// FLOAT32 BITS (not values) against many random blocks — bits because a value
// comparison can't distinguish +0 from -0, and this file's own sign-flip change
// is exactly the kind of edit that could introduce a wrong-sign zero.

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"testing"
)

// dequantIQ2SBlockPreOpt is the exact pre-optimization body: a branchy
// `if (sg>>j)&1 != 0 { v = -v }` instead of the branchless XOR.
func dequantIQ2SBlockPreOpt(raw []byte, sb int, out []float32) {
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

// dequantIQ3SBlockPreOpt is the exact pre-optimization body: a flat 256-iteration
// loop (recomputing db 8x more often and idx 4x more often than needed) with a
// branchy sign flip, instead of the restructured sub/gridChunk/i nesting.
func dequantIQ3SBlockPreOpt(raw []byte, sb int, out []float32) {
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

// dequantQ4KBlockPreOpt is the exact pre-optimization body: an accumulated yi
// counter instead of a direct index, and out not resliced to a constant cap.
// (Found and fixed after the initial pass — same yi pattern as Q5_K/Q2_K/Q3_K,
// missed the first time because it wasn't in the original suggestion list.)
func dequantQ4KBlockPreOpt(raw []byte, sb int, out []float32) {
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

// dequantQ5KBlockPreOpt is the exact pre-optimization body: an accumulated yi
// counter instead of a direct index, and out not resliced to a constant cap.
func dequantQ5KBlockPreOpt(raw []byte, sb int, out []float32) {
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

// dequantQ2KBlockPreOpt is the exact pre-optimization body.
func dequantQ2KBlockPreOpt(raw []byte, sb int, out []float32) {
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

// dequantQ3KBlockPreOpt is the exact pre-optimization body.
func dequantQ3KBlockPreOpt(raw []byte, sb int, out []float32) {
	const (
		kmask1 = 0x03030303
		kmask2 = 0x0f0f0f0f
	)
	base := sb * 110
	hm := raw[base : base+32]
	q := raw[base+32 : base+96]
	scRaw := raw[base+96 : base+108]
	dAll := halfBitsToF32(binary.LittleEndian.Uint16(raw[base+108:]))

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

// bitIdenticalCheck runs both the optimized and PreOpt kernel over `blocks`
// random super-blocks and fails on the first bit-level mismatch.
func bitIdenticalCheck(t *testing.T, name string, blockBytes, elems int, optFn, preOptFn func(raw []byte, sb int, out []float32)) {
	t.Helper()
	rng := rand.New(rand.NewPCG(1, 2))
	const blocks = 200
	raw := make([]byte, blocks*blockBytes)
	for i := range raw {
		raw[i] = byte(rng.UintN(256))
	}
	got := make([]float32, elems)
	want := make([]float32, elems)
	for sb := range blocks {
		optFn(raw, sb, got)
		preOptFn(raw, sb, want)
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s: block %d elem %d: optimized %v (bits %08x), pre-opt %v (bits %08x)",
					name, sb, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			}
		}
	}
}

func TestDequantIQ2SBlock_branchlessIsBitIdentical(t *testing.T) {
	bitIdenticalCheck(t, "IQ2_S", 82, 256, dequantIQ2SBlock, dequantIQ2SBlockPreOpt)
}

func TestDequantIQ3SBlock_restructureIsBitIdentical(t *testing.T) {
	bitIdenticalCheck(t, "IQ3_S", 110, 256, dequantIQ3SBlock, dequantIQ3SBlockPreOpt)
}

func TestDequantQ4KBlock_directIndexIsBitIdentical(t *testing.T) {
	bitIdenticalCheck(t, "Q4_K", 144, 256, dequantQ4KBlock, dequantQ4KBlockPreOpt)
}

func TestDequantQ5KBlock_directIndexIsBitIdentical(t *testing.T) {
	bitIdenticalCheck(t, "Q5_K", 176, 256, dequantQ5KBlock, dequantQ5KBlockPreOpt)
}

func TestDequantQ2KBlock_directIndexIsBitIdentical(t *testing.T) {
	bitIdenticalCheck(t, "Q2_K", 84, 256, dequantQ2KBlock, dequantQ2KBlockPreOpt)
}

func TestDequantQ3KBlock_directIndexIsBitIdentical(t *testing.T) {
	bitIdenticalCheck(t, "Q3_K", 110, 256, dequantQ3KBlock, dequantQ3KBlockPreOpt)
}
