package embed

import (
	"encoding/binary"
	"fmt"
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

func TestDequantQ5_0(t *testing.T) {
	// One block: d=2.0; f16 scale, 4-byte qh (high bit per element), 16 bytes of
	// low nibbles. code = (lowNibble | highBit<<4) ∈ [0,31]; value = d*(code-16).
	// elem j low nibble = j&0xF, high bit = qh bit j; elem j+16 low nibble =
	// (15-j)&0xF, high bit = qh bit (j+16). Set qh so all high bits are 1.
	raw := make([]byte, 22)
	binary.LittleEndian.PutUint16(raw[0:], f16bits(2.0))
	binary.LittleEndian.PutUint32(raw[2:], 0xFFFFFFFF) // every element's 5th bit set
	for j := 0; j < 16; j++ {
		raw[6+j] = byte(j) | byte(15-j)<<4
	}
	got := dequantQ5_0(raw, 32)
	for j := 0; j < 16; j++ {
		if w := 2.0 * float32((j|0x10)-16); got[j] != w {
			t.Errorf("Q5_0[%d] = %v, want %v", j, got[j], w)
		}
		if w := 2.0 * float32(((15-j)|0x10)-16); got[j+16] != w {
			t.Errorf("Q5_0[%d] = %v, want %v", j+16, got[j+16], w)
		}
	}
	// And with all high bits 0 (qh=0): code = low nibble only.
	binary.LittleEndian.PutUint32(raw[2:], 0)
	got = dequantQ5_0(raw, 32)
	if w := 2.0 * float32(0-16); got[0] != w {
		t.Errorf("Q5_0 qh=0 [0] = %v, want %v", got[0], w)
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

// TestGGUFMmap_matchesHeap: OpenGGUFMmap must parse identically to OpenGGUF —
// same metadata and bit-identical dequantized tensors — so the mmap path is a
// pure memory optimization, not a behavior change. After Close the mapping is
// released (the weights were copied out, so nothing dangles).
func TestGGUFMmap_matchesHeap(t *testing.T) {
	path := "../testdata/tinyllama-gguf/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
	if _, err := os.Stat(path); err != nil {
		t.Skip("no TinyLlama GGUF")
	}
	heap, err := OpenGGUF(path)
	if err != nil {
		t.Fatalf("OpenGGUF: %v", err)
	}
	mm, err := OpenGGUFMmap(path)
	if err != nil {
		t.Fatalf("OpenGGUFMmap: %v", err)
	}

	// Metadata: same key set and same scalar values on a representative few.
	if len(mm.Metadata) != len(heap.Metadata) {
		t.Errorf("metadata count: mmap %d, heap %d", len(mm.Metadata), len(heap.Metadata))
	}
	for _, k := range []string{"general.architecture", "llama.block_count", "llama.embedding_length"} {
		if a, b := fmt.Sprint(mm.Metadata[k]), fmt.Sprint(heap.Metadata[k]); a != b {
			t.Errorf("metadata[%q]: mmap %s, heap %s", k, a, b)
		}
	}

	// Tensors: dequant must be bit-identical on a few of each layout.
	for _, name := range []string{"token_embd.weight", "blk.0.attn_q.weight", "blk.0.ffn_down.weight", "output_norm.weight"} {
		_, hd, herr := heap.Tensor(name)
		_, md, merr := mm.Tensor(name)
		if herr != nil || merr != nil {
			t.Fatalf("Tensor(%q): heap err=%v mmap err=%v", name, herr, merr)
		}
		if len(md) != len(hd) {
			t.Fatalf("Tensor(%q): len mmap %d, heap %d", name, len(md), len(hd))
		}
		for i := range hd {
			if md[i] != hd[i] {
				t.Fatalf("Tensor(%q)[%d]: mmap %v != heap %v", name, i, md[i], hd[i])
			}
		}
	}

	if err := mm.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
	if err := mm.Close(); err != nil { // idempotent
		t.Errorf("second Close: %v", err)
	}
}
