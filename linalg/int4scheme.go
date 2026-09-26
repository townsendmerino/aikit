package linalg

import (
	"fmt"
	"math"
)

// INT4 WEIGHT-QUANTIZER CANDIDATES (goinfer docs/tasks/task-actquant-pergroup-2026-09.md, Track B).
//
// QuantizeGroupInt4Row's default sets scale = max|w|/7 and clamps codes to [-7, 7]: it uses 15 of
// the 16 levels a nibble holds (code -8 is never produced) and searches nothing. The candidates here
// keep the format exactly (decode (nibble-8)*scale, same packing, same group) and change only how the
// scale and codes are chosen, so no kernel changes. Experimental, behind SetInt4WeightScheme; the
// default "" leaves QuantizeGroupInt4Row bit-identical.
//
//   - "fullrange": scale = (signed element of largest magnitude) / -8, llama.cpp Q4_0's rule. That
//     element becomes code -8, the rest round into [-8, 7]. The scale may be negative.
//   - "mse": per group, scales +/-max|w|/d for d on a grid over [6.5, 8.5], codes rounded and
//     clamped to [-8, 7], keeping the scale with the least squared reconstruction error.

var int4WeightScheme string

// SetInt4WeightScheme selects the int4 weight quantizer: "" (the default max/7 rule), "fullrange" or
// "mse". It affects only quantization done after the call; set it before loading a model.
func SetInt4WeightScheme(scheme string) {
	switch scheme {
	case "", "fullrange", "mse":
		int4WeightScheme = scheme
	default:
		panic(fmt.Sprintf("linalg: SetInt4WeightScheme(%q): want \"\", \"fullrange\" or \"mse\"", scheme))
	}
}

// Int4WeightScheme reports the scheme SetInt4WeightScheme selected.
func Int4WeightScheme() string { return int4WeightScheme }

// int4Codes rounds row[ks:ke]/s into [-8, 7] and returns the squared reconstruction error.
func int4Codes(row []float32, ks, ke int, s float32, codes []int8) float64 {
	inv := 1 / s
	var e float64
	for k := ks; k < ke; k++ {
		q := math.Round(float64(row[k] * inv))
		q = max(-8, min(7, q))
		codes[k-ks] = int8(q)
		d := float64(row[k]) - q*float64(s)
		e += d * d
	}
	return e
}

// quantizeInt4GroupAlt quantizes row[ks:ke] with int4WeightScheme into packed (nibble k/2, low
// nibble even k) and returns the group's scale. An all-zero group gets scale 1 and code 0, as the
// default does.
func quantizeInt4GroupAlt(row []float32, ks, ke int, packed []byte) float32 {
	var maxAbs, signed float32
	for k := ks; k < ke; k++ {
		if a := float32(math.Abs(float64(row[k]))); a > maxAbs {
			maxAbs, signed = a, row[k]
		}
	}
	codes := make([]int8, ke-ks)
	s := float32(1)
	if maxAbs > 0 {
		switch int4WeightScheme {
		case "fullrange":
			s = signed / -8
			int4Codes(row, ks, ke, s, codes)
		case "mse":
			best := math.Inf(1)
			trial := make([]int8, ke-ks)
			for _, sign := range []float32{1, -1} {
				for d := float32(6.5); d <= 8.5001; d += 0.25 {
					c := sign * maxAbs / d
					if e := int4Codes(row, ks, ke, c, trial); e < best {
						best, s = e, c
						copy(codes, trial)
					}
				}
			}
		}
	}
	for k := ks; k < ke; k++ {
		nib := byte(codes[k-ks]+8) & 0x0F
		if k&1 == 0 {
			packed[k/2] = (packed[k/2] &^ 0x0F) | nib
		} else {
			packed[k/2] = (packed[k/2] &^ 0xF0) | nib<<4
		}
	}
	return s
}
