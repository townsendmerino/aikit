package embed

import (
	"math"
	"testing"
)

// le16 packs a 16-bit value little-endian (the safetensors byte order).
func le16(v uint16) []byte { return []byte{byte(v), byte(v >> 8)} }

// TestBFloat16sToF32 decodes a hand-built BF16 byte slice. bf16 is the top
// 16 bits of the f32 pattern, so each expected value is just f32's high
// halfword. Covers a positive, a negative, a zero, and a fraction.
func TestBFloat16sToF32(t *testing.T) {
	cases := []struct {
		name string
		hi16 uint16  // high 16 bits of the float32 pattern
		want float32 // exact (bf16 of these is lossless)
	}{
		{"one", 0x3f80, 1.0},
		{"neg_two", 0xc000, -2.0},
		{"zero", 0x0000, 0.0},
		{"onehalf", 0x3fc0, 1.5},
		{"neg_onehalf", 0xbfc0, -1.5},
	}
	var raw []byte
	for _, c := range cases {
		raw = append(raw, le16(c.hi16)...)
	}
	tn := Tensor{Name: "t", DType: "BF16", Shape: []int{len(cases)}, raw: raw}
	got, err := tn.BFloat16sToF32()
	if err != nil {
		t.Fatalf("BFloat16sToF32: %v", err)
	}
	if len(got) != len(cases) {
		t.Fatalf("got %d elems, want %d", len(got), len(cases))
	}
	for i, c := range cases {
		if got[i] != c.want {
			t.Errorf("%s: got %v want %v", c.name, got[i], c.want)
		}
	}

	// Guards.
	if _, err := (Tensor{Name: "t", DType: "F32"}).BFloat16sToF32(); err == nil {
		t.Error("expected DType-mismatch error")
	}
	if _, err := (Tensor{Name: "t", DType: "BF16", raw: []byte{0x00}}).BFloat16sToF32(); err == nil {
		t.Error("expected odd-length error")
	}
}

// TestFloat16sToF32 decodes a hand-built F16 byte slice covering the three
// decode paths: normal (1.0, 0.5, -2.0), zero, and a subnormal. The
// subnormal 0x0001 is the smallest positive half = 2^-24.
func TestFloat16sToF32(t *testing.T) {
	cases := []struct {
		name string
		bits uint16
		want float32
	}{
		{"one", 0x3c00, 1.0},
		{"half", 0x3800, 0.5},
		{"neg_two", 0xc000, -2.0}, // negative
		{"zero", 0x0000, 0.0},     // zero
		{"two", 0x4000, 2.0},
		{"min_subnormal", 0x0001, float32(math.Ldexp(1, -24))}, // f16 subnormal = 2^-24
		{"max_subnormal", 0x03ff, float32(math.Ldexp(1023, -24))},
	}
	var raw []byte
	for _, c := range cases {
		raw = append(raw, le16(c.bits)...)
	}
	tn := Tensor{Name: "t", DType: "F16", Shape: []int{len(cases)}, raw: raw}
	got, err := tn.Float16sToF32()
	if err != nil {
		t.Fatalf("Float16sToF32: %v", err)
	}
	for i, c := range cases {
		if got[i] != c.want {
			t.Errorf("%s: got %v (bits %#x) want %v", c.name, got[i], c.bits, c.want)
		}
	}

	// Inf / NaN carry over.
	inf, _ := (Tensor{Name: "t", DType: "F16", raw: le16(0x7c00)}).Float16sToF32()
	if !math.IsInf(float64(inf[0]), 1) {
		t.Errorf("0x7c00 should be +Inf, got %v", inf[0])
	}
	nan, _ := (Tensor{Name: "t", DType: "F16", raw: le16(0x7e00)}).Float16sToF32()
	if !math.IsNaN(float64(nan[0])) {
		t.Errorf("0x7e00 should be NaN, got %v", nan[0])
	}

	// Guards.
	if _, err := (Tensor{Name: "t", DType: "BF16"}).Float16sToF32(); err == nil {
		t.Error("expected DType-mismatch error")
	}
	if _, err := (Tensor{Name: "t", DType: "F16", raw: []byte{0x00}}).Float16sToF32(); err == nil {
		t.Error("expected odd-length error")
	}
}
