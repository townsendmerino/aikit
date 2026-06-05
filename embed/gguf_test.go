package embed

import (
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"math/rand"
	"os"
	"sort"
	"testing"
)

// TestIQDequant_matchesReference parity-gates the IQ4_NL / IQ4_XS dequant against
// llama.cpp's gguf Python reference: the committed golden holds deterministic raw
// super-blocks (raw_hex) and the reference dequantization (expected); dequantRange
// must reproduce them. Codebook quants have no convenient small-model f32 oracle,
// so this pins the kernel directly (every value, not just a forward cosine).
// Regenerate: .venv/bin/python scripts/pin_iq_dequant.py
func TestIQDequant_matchesReference(t *testing.T) {
	raw, err := os.ReadFile("../testdata/iq_dequant_golden.json")
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no IQ golden — regenerate with scripts/pin_iq_dequant.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g struct {
		Cases []struct {
			Type     string    `json:"type"`
			GGMLType uint32    `json:"ggml_type"`
			Elems    int       `json:"elems"`
			RawHex   string    `json:"raw_hex"`
			Expected []float32 `json:"expected"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if len(g.Cases) == 0 {
		t.Fatal("golden has no cases")
	}
	for _, c := range g.Cases {
		t.Run(c.Type, func(t *testing.T) {
			blk, err := hex.DecodeString(c.RawHex)
			if err != nil {
				t.Fatalf("decode raw_hex: %v", err)
			}
			bs, ok := ggmlBlockElems(c.GGMLType)
			if !ok {
				t.Fatalf("type %d (%s) not supported", c.GGMLType, c.Type)
			}
			got := make([]float32, c.Elems)
			dequantRange(c.GGMLType, blk, 0, got, bs)
			var maxErr float64
			for i := range got {
				// Same float32 ops as the reference, so the only slack is mantissa
				// rounding; a logic bug (wrong codebook entry / scale bit) is off by
				// far more than this relative bound.
				d := math.Abs(float64(got[i] - c.Expected[i]))
				if d > maxErr {
					maxErr = d
				}
				if tol := 1e-4 * (1 + math.Abs(float64(c.Expected[i]))); d > tol {
					t.Fatalf("%s[%d] = %v, want %v (Δ%.2e > %.2e)", c.Type, i, got[i], c.Expected[i], d, tol)
				}
			}
			t.Logf("%s: %d values match gguf reference, maxΔ=%.3e", c.Type, c.Elems, maxErr)
		})
	}
}

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
	for i := range 32 {
		raw[2+i] = byte(int8(i - 16))
	}
	got := make([]float32, 32)
	dequantQ8_0Block(raw, 0, got)
	for i := range 32 {
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
	for j := range 16 {
		raw[6+j] = byte(j) | byte(15-j)<<4
	}
	got := make([]float32, 32)
	dequantQ5_0Block(raw, 0, got)
	for j := range 16 {
		if w := 2.0 * float32((j|0x10)-16); got[j] != w {
			t.Errorf("Q5_0[%d] = %v, want %v", j, got[j], w)
		}
		if w := 2.0 * float32(((15-j)|0x10)-16); got[j+16] != w {
			t.Errorf("Q5_0[%d] = %v, want %v", j+16, got[j+16], w)
		}
	}
	// And with all high bits 0 (qh=0): code = low nibble only.
	binary.LittleEndian.PutUint32(raw[2:], 0)
	dequantQ5_0Block(raw, 0, got)
	if w := 2.0 * float32(0-16); got[0] != w {
		t.Errorf("Q5_0 qh=0 [0] = %v, want %v", got[0], w)
	}
}

func TestDequantQ4_0(t *testing.T) {
	// One block: d=2.0; byte j low nibble → elem j, high nibble → elem j+16;
	// value = d*(nibble-8).
	raw := make([]byte, 18)
	binary.LittleEndian.PutUint16(raw[0:], f16bits(2.0))
	for j := range 16 {
		lo := byte(j)      // 0..15
		hi := byte(15 - j) // 15..0
		raw[2+j] = lo | hi<<4
	}
	got := make([]float32, 32)
	dequantQ4_0Block(raw, 0, got)
	for j := range 16 {
		if w := 2.0 * float32(j-8); got[j] != w {
			t.Errorf("Q4_0[%d] = %v, want %v", j, got[j], w)
		}
		if w := 2.0 * float32((15-j)-8); got[j+16] != w {
			t.Errorf("Q4_0[%d] = %v, want %v", j+16, got[j+16], w)
		}
	}
}

// TestDequantRange_streamMatchesWhole: dequantizing a tensor in block-aligned
// sub-ranges (the load-time streaming path, RowDequantizer) must be bit-identical
// to dequantizing it all at once (Tensor) — same kernels, only the start offset
// differs. Random raw bytes (their f16 scales may be NaN/Inf, so compare bit
// patterns, not values) over several blocks of each supported ggml type.
func TestDequantRange_streamMatchesWhole(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	types := []struct {
		name    string
		typ     uint32
		blkByte int
	}{
		{"F32", ggmlTypeF32, 4}, {"F16", ggmlTypeF16, 2},
		{"Q8_0", ggmlTypeQ8_0, 34}, {"Q4_0", ggmlTypeQ4_0, 18}, {"Q5_0", ggmlTypeQ5_0, 22},
		{"Q2_K", ggmlTypeQ2_K, 84}, {"Q3_K", ggmlTypeQ3_K, 110}, {"Q4_K", ggmlTypeQ4_K, 144},
		{"Q5_K", ggmlTypeQ5_K, 176}, {"Q6_K", ggmlTypeQ6_K, 210},
		{"IQ4_NL", ggmlTypeIQ4NL, 18}, {"IQ4_XS", ggmlTypeIQ4XS, 136},
		{"IQ2_S", ggmlTypeIQ2S, 82}, {"IQ3_S", ggmlTypeIQ3S, 110},
	}
	for _, tc := range types {
		bs, _ := ggmlBlockElems(tc.typ)
		const nBlocks = 5
		n := nBlocks * bs
		raw := make([]byte, nBlocks*tc.blkByte)
		for i := range raw {
			raw[i] = byte(rng.Intn(256))
		}
		whole := make([]float32, n)
		dequantRange(tc.typ, raw, 0, whole, bs)

		// Re-dequant in chunks of 1, 2, then the rest of the blocks, at successive
		// block-aligned starts — exercising the offset arithmetic.
		streamed := make([]float32, n)
		for start, step := 0, bs; start < n; start += step {
			if start == 2*bs {
				step = (nBlocks - 2) * bs // jump to the tail in one go
			}
			end := min(start+step, n)
			dequantRange(tc.typ, raw, start, streamed[start:end], bs)
		}
		for i := range whole {
			if math.Float32bits(whole[i]) != math.Float32bits(streamed[i]) {
				t.Errorf("%s[%d]: streamed %x != whole %x", tc.name, i, math.Float32bits(streamed[i]), math.Float32bits(whole[i]))
				break
			}
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

// TestOpenGGUFBytes_matchesMmap: OpenGGUFBytes parses an in-memory slice
// identically to OpenGGUFMmap — same tensor name set and bit-identical
// dequantized tensors — so the bytes entry point is a pure no-filesystem
// convenience, not a behavior change. Close is a no-op on the bytes file
// (nothing is mapped).
func TestOpenGGUFBytes_matchesMmap(t *testing.T) {
	path := "../testdata/tinyllama-gguf/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
	if _, err := os.Stat(path); err != nil {
		t.Skip("no TinyLlama GGUF")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	bytesF, err := OpenGGUFBytes(raw)
	if err != nil {
		t.Fatalf("OpenGGUFBytes: %v", err)
	}
	mm, err := OpenGGUFMmap(path)
	if err != nil {
		t.Fatalf("OpenGGUFMmap: %v", err)
	}
	defer mm.Close()

	// Same tensor name set (Names is map-order, so sort before comparing).
	bn, mn := bytesF.Names(), mm.Names()
	if len(bn) != len(mn) {
		t.Fatalf("Names count: bytes %d, mmap %d", len(bn), len(mn))
	}
	sort.Strings(bn)
	sort.Strings(mn)
	for i := range bn {
		if bn[i] != mn[i] {
			t.Fatalf("Names[%d]: bytes %q, mmap %q", i, bn[i], mn[i])
		}
	}

	// Bit-identical dims + dequant on a representative few of each layout.
	for _, name := range []string{"token_embd.weight", "blk.0.attn_q.weight", "blk.0.ffn_down.weight", "output_norm.weight"} {
		bd, bdata, berr := bytesF.Tensor(name)
		md, mdata, merr := mm.Tensor(name)
		if berr != nil || merr != nil {
			t.Fatalf("Tensor(%q): bytes err=%v mmap err=%v", name, berr, merr)
		}
		if len(bd) != len(md) {
			t.Fatalf("Tensor(%q) dims: bytes %v, mmap %v", name, bd, md)
		}
		for i := range bd {
			if bd[i] != md[i] {
				t.Fatalf("Tensor(%q) dim[%d]: bytes %d != mmap %d", name, i, bd[i], md[i])
			}
		}
		if len(bdata) != len(mdata) {
			t.Fatalf("Tensor(%q) len: bytes %d, mmap %d", name, len(bdata), len(mdata))
		}
		for i := range bdata {
			if bdata[i] != mdata[i] {
				t.Fatalf("Tensor(%q)[%d]: bytes %v != mmap %v", name, i, bdata[i], mdata[i])
			}
		}
	}

	// Close on the bytes-backed file is a safe no-op (nothing mapped).
	if err := bytesF.Close(); err != nil {
		t.Errorf("OpenGGUFBytes Close should be nil, got %v", err)
	}
}
