package embed

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"testing"
)

// dequantQ6KBlockPreHoist is the exact pre-M-24 body: d·scale recomputed per
// element, as `d * float32(scale) * float32(q)`.
func dequantQ6KBlockPreHoist(raw []byte, sb int, out []float32) {
	base := sb * 210
	ql := raw[base : base+128]
	qh := raw[base+128 : base+192]
	sc := raw[base+192 : base+208]
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

// TestDequantQ6KBlock_hoistIsBitIdentical gates audit M-24's hoist. The claim
// is EQUALITY, not a tolerance: the original groups as (d*scale)*q by Go's
// left-to-right evaluation, so precomputing d*scale keeps the same two
// roundings in the same order. Any tolerance here would hide the one way this
// could go wrong.
func TestDequantQ6KBlock_hoistIsBitIdentical(t *testing.T) {
	rng := rand.New(rand.NewPCG(6, 6))
	const blocks = 64
	raw := make([]byte, blocks*210)
	for i := range raw {
		raw[i] = byte(rng.UintN(256))
	}
	got := make([]float32, 256)
	want := make([]float32, 256)
	for sb := range blocks {
		dequantQ6KBlock(raw, sb, got)
		dequantQ6KBlockPreHoist(raw, sb, want)
		for i := range want {
			// Compare BITS, not values. Random bytes give the f16 super-scale
			// NaN payloads, and NaN != NaN would fail a value comparison on
			// output that is in fact identical — which is exactly what the first
			// version of this test did. Bits are also the right check for a
			// bit-identity claim: they distinguish +0 from -0, which a value
			// comparison does not.
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("block %d elem %d: hoisted %v (bits %08x), reference %v (bits %08x)",
					sb, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			}
		}
	}
}
