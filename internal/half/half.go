// Package half decodes IEEE-754 binary16 to float32: the one scalar decode linalg, its kernel packages and embed share,
// so the f16 weights, scales and caches they read widen identically everywhere. Every binary16 value is exactly
// representable in float32, so the decode is exact, and a NaN keeps its sign and payload (a signaling NaN stays
// signaling).
package half

import "math"

// ToF32 converts one binary16 bit pattern to float32. Subnormals are renormalized into a float32 normal; exp is signed
// because the normalization loop can drive it below zero before the rebias.
func ToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := int32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)

	var bits uint32
	switch {
	case exp == 0:
		if mant == 0 {
			bits = sign // ±0
			break
		}
		// Shift the mantissa left until the implicit leading 1 reaches bit 10, decrementing the exponent per shift,
		// then drop that bit and rebias (15 → 127).
		exp = 1
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		mant &= 0x3ff
		bits = sign | uint32(exp+(127-15))<<23 | mant<<13
	case exp == 0x1f:
		bits = sign | 0x7f800000 | mant<<13 // Inf (mant == 0) or NaN
	default:
		bits = sign | uint32(exp+(127-15))<<23 | mant<<13
	}
	return math.Float32frombits(bits)
}
