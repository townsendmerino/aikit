package half

import (
	"math"
	"testing"
)

// reference decodes h by arithmetic rather than bit placement: (-1)^s · 2^(e-15) · (1 + m/1024) for a normal,
// (-1)^s · 2^-14 · m/1024 for a subnormal. Inf and NaN are placed by bits (a NaN has no arithmetic value).
func reference(h uint16) uint32 {
	s, e, m := h>>15, int(h>>10)&0x1f, float64(h&0x3ff)
	if e == 0x1f {
		return uint32(s)<<31 | 0x7f800000 | uint32(h&0x3ff)<<13
	}
	v := math.Ldexp(m/1024, -14)
	if e != 0 {
		v = math.Ldexp(1+m/1024, e-15)
	}
	if s == 1 {
		v = -v
	}
	return math.Float32bits(float32(v))
}

// TestToF32_allHalves: every one of the 65,536 bit patterns, bit for bit, against the arithmetic reference (which also
// keeps -0's sign and each NaN's payload).
func TestToF32_allHalves(t *testing.T) {
	bad := 0
	for i := range 1 << 16 {
		h := uint16(i)
		if got, want := math.Float32bits(ToF32(h)), reference(h); got != want {
			if bad++; bad <= 5 {
				t.Errorf("ToF32(%#04x) = %#08x, want %#08x", h, got, want)
			}
		}
	}
	if bad > 0 {
		t.Errorf("%d of 65536 halves differ", bad)
	}
}
