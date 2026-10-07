package embed

import (
	"encoding/binary"
	"math"
	"testing"
)

// TestDequantBF16: a GGUF BF16 element is the top 16 bits of an f32, so it reads back as exactly that f32 with the low
// half zeroed, signs, zeros, infinities and a subnormal included; a range starting mid-tensor reads the right
// elements.
func TestDequantBF16(t *testing.T) {
	vals := []float32{0, -0, 1, -2.5, 3.140625, float32(math.Inf(1)), float32(math.Inf(-1)), 1e-40, -65504, 0.00392156862745098}
	raw := make([]byte, 2*len(vals))
	want := make([]float32, len(vals))
	for i, v := range vals {
		b := math.Float32bits(v) >> 16
		binary.LittleEndian.PutUint16(raw[2*i:], uint16(b))
		want[i] = math.Float32frombits(b << 16)
	}
	dst := make([]float32, len(vals))
	dequantRange(ggmlTypeBF16, raw, 0, dst, 1)
	for i := range dst {
		if math.Float32bits(dst[i]) != math.Float32bits(want[i]) {
			t.Errorf("element %d: %g (bits %08x), want %g (%08x)", i, dst[i], math.Float32bits(dst[i]), want[i], math.Float32bits(want[i]))
		}
	}
	part := make([]float32, 3)
	dequantRange(ggmlTypeBF16, raw, 4, part, 1)
	for i := range part {
		if math.Float32bits(part[i]) != math.Float32bits(want[4+i]) {
			t.Errorf("range from 4, element %d: %g, want %g", i, part[i], want[4+i])
		}
	}
	if n, ok := ggmlBlockElems(ggmlTypeBF16); !ok || n != 1 {
		t.Errorf("ggmlBlockElems(BF16) = %d, %v", n, ok)
	}
}
