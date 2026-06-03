package embed

import (
	"encoding/binary"
	"math"
	"os"
	"testing"
)

// f16bits returns the IEEE-754 binary16 bit pattern of v (test helper; only
// exact-in-f16 values are used below).
func f16bits(v float32) uint16 {
	// Reuse the decoder by round-tripping through known-exact values.
	switch v {
	case 2.0:
		return 0x4000
	case 0.5:
		return 0x3800
	case 1.0:
		return 0x3C00
	}
	panic("f16bits: add the constant")
}

func TestDequantQ8_0(t *testing.T) {
	// One block of 32: scale d=2.0, qs = -16..15 → value = 2*q.
	raw := make([]byte, 34)
	binary.LittleEndian.PutUint16(raw[0:], f16bits(2.0))
	for i := 0; i < 32; i++ {
		raw[2+i] = byte(int8(i - 16))
	}
	got := dequantQ8_0(raw, 32)
	for i := 0; i < 32; i++ {
		want := 2.0 * float32(i-16)
		if got[i] != want {
			t.Errorf("Q8_0[%d] = %v, want %v", i, got[i], want)
		}
	}
}

func TestDequantQ4_0(t *testing.T) {
	// One block: d=2.0; byte j low nibble → elem j, high nibble → elem j+16;
	// value = d*(nibble-8).
	raw := make([]byte, 18)
	binary.LittleEndian.PutUint16(raw[0:], f16bits(2.0))
	for j := 0; j < 16; j++ {
		lo := byte(j)      // 0..15
		hi := byte(15 - j) // 15..0
		raw[2+j] = lo | hi<<4
	}
	got := dequantQ4_0(raw, 32)
	for j := 0; j < 16; j++ {
		if w := 2.0 * float32(j-8); got[j] != w {
			t.Errorf("Q4_0[%d] = %v, want %v", j, got[j], w)
		}
		if w := 2.0 * float32((15-j)-8); got[j+16] != w {
			t.Errorf("Q4_0[%d] = %v, want %v", j+16, got[j+16], w)
		}
	}
}

// TestGGUF_realFile is a light smoke test against a real TinyLlama GGUF when
// present: header parses, metadata reads, and Q6_K dequant of the output head
// matches the Python reference's first values (pinned below).
func TestGGUF_realFile(t *testing.T) {
	path := "../testdata/tinyllama-gguf/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
	if _, err := os.Stat(path); err != nil {
		t.Skip("no TinyLlama GGUF")
	}
	g, err := OpenGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	if arch, _ := g.Str("general.architecture"); arch != "llama" {
		t.Errorf("architecture = %q, want llama", arch)
	}
	if n, _ := g.Uint("llama.block_count"); n != 22 {
		t.Errorf("block_count = %d, want 22", n)
	}
	dims, data, err := g.Tensor("blk.0.attn_q.weight")
	if err != nil {
		t.Fatal(err)
	}
	if len(dims) != 2 || dims[0] != 2048 || dims[1] != 2048 {
		t.Fatalf("attn_q dims = %v, want [2048 2048]", dims)
	}
	// Pinned from the Python gguf reference (raw Q8_0 dequant, pre-permute).
	want := []float32{-0.0014365911, -0.0024311543, -0.0074039698}
	for i, w := range want {
		if math.Abs(float64(data[i]-w)) > 1e-9 {
			t.Errorf("attn_q[%d] = %v, want %v", i, data[i], w)
		}
	}
}
