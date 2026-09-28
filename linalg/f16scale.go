package linalg

import "math"

// Int4 group scales are stored as IEEE binary16 (goinfer docs/tasks/task-cpu-decode-peer-gap-2026-09.md, L1):
// 2 bytes per 32 weights instead of 4, and the exact values every GPU backend already serves. Kernels widen
// them to f32 (exactly) before use.

// F32ToF16 converts f to the IEEE binary16 bit pattern int4 scales are stored in. It is goinfer's F16Bits
// rule, bit for bit, so a scale converted here and one a GPU backend converts are the same bits:
// round-half-up on the dropped mantissa bits (not ties-to-even), carry into the exponent, overflow to ±Inf,
// NaN to a quiet NaN, and gradual underflow.
func F32ToF16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 16) & 0x8000)
	e := int32((b>>23)&0xFF) - 112
	m := b & 0x7FFFFF
	switch {
	case (b>>23)&0xFF == 0xFF:
		if m != 0 {
			return sign | 0x7E00
		}
		return sign | 0x7C00
	case e >= 0x1F:
		return sign | 0x7C00
	case e <= 0:
		if e < -10 {
			return sign
		}
		m |= 0x800000
		sh := uint32(14 - e)
		return sign | uint16((m+(1<<(sh-1)))>>sh)
	default:
		half := sign | uint16(e<<10) | uint16(m>>13)
		if m&0x1000 != 0 {
			half++
		}
		return half
	}
}

// F16ToF32 widens an IEEE binary16 bit pattern to f32. Exact.
func F16ToF32(h uint16) float32 { return f16ToF32(h) }

// F32ToF16Slice converts src into dst[:len(src)] with F32ToF16.
func F32ToF16Slice(dst []uint16, src []float32) {
	_ = dst[len(src)-1:]
	for i, f := range src {
		dst[i] = F32ToF16(f)
	}
}

// F32ToF16Scales returns src converted with F32ToF16 in a new slice (nil for nil).
func F32ToF16Scales(src []float32) []uint16 {
	if src == nil {
		return nil
	}
	dst := make([]uint16, len(src))
	for i, f := range src {
		dst[i] = F32ToF16(f)
	}
	return dst
}

// F16ToF32Slice widens src into dst[:len(src)], exactly (SIMD where the CPU has it).
func F16ToF32Slice(dst []float32, src []uint16) {
	if len(src) == 0 {
		return
	}
	_ = dst[len(src)-1]
	widenF16(dst[:len(src)], src)
}

// widenScalesCopy returns src widened in a new slice (nil for nil).
func widenScalesCopy(src []uint16) []float32 {
	if src == nil {
		return nil
	}
	dst := make([]float32, len(src))
	F16ToF32Slice(dst, src)
	return dst
}
