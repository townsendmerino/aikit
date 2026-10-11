package embed

import (
	"encoding/binary"
	"hash/crc32"
	"math"
	"math/rand"
	"os"
	"sort"
	"testing"
)

// q4kSame runs dequantQ4KBlock and dequantQ4KBlockScalar on block sb of raw and compares the 256 weights by bits.
func q4kSame(t *testing.T, what string, raw []byte, sb int) bool {
	t.Helper()
	var got, want [256]float32
	for i := range got {
		got[i], want[i] = float32(math.NaN()), float32(math.NaN()) // every weight must be written
	}
	dequantQ4KBlockScalar(raw, sb, want[:])
	dequantQ4KBlock(raw, sb, got[:])
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Errorf("%s: weight %d is %g (bits %08x), the scalar's %g (%08x)", what, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			return false
		}
	}
	return true
}

// TestDequantQ4KBlock_matchesScalar: dequantQ4KBlock gives the scalar's weights, bit for bit, on this architecture:
// random blocks; every half-float as d and as dmin (zeros, subnormals, both infinities and the NaNs among them) over
// random scales and nibbles; every byte value in every scales position; and blocks whose nibble bytes run through all
// 256 values in every lane. A block read at an offset checks the block index.
func TestDequantQ4KBlock_matchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(20261010))
	raw := make([]byte, 3*144)
	fill := func() {
		for i := range raw {
			raw[i] = byte(rng.Intn(256))
		}
	}
	for range 20000 {
		fill()
		if !q4kSame(t, "a random block", raw, rng.Intn(3)) {
			return
		}
	}
	for h := range 1 << 16 {
		fill()
		binary.LittleEndian.PutUint16(raw[144:], uint16(h))
		if !q4kSame(t, "every d", raw, 1) {
			return
		}
		fill()
		binary.LittleEndian.PutUint16(raw[144+2:], uint16(h))
		if !q4kSame(t, "every dmin", raw, 1) {
			return
		}
	}
	for _, pair := range [][2]uint16{{0x7C00, 0x7C00}, {0x7E00, 0x7E01}, {0xFE00, 0x7C00}, {0x0000, 0x8000}, {0x0001, 0x0001}, {0x7C00, 0x0000}} {
		fill()
		binary.LittleEndian.PutUint16(raw[144:], pair[0])
		binary.LittleEndian.PutUint16(raw[146:], pair[1])
		if !q4kSame(t, "d and dmin both special", raw, 1) {
			return
		}
	}
	for pos := range 12 {
		for v := range 256 {
			fill()
			raw[144+4+pos] = byte(v)
			if !q4kSame(t, "every scales byte", raw, 1) {
				return
			}
		}
	}
	for shift := range 256 {
		fill()
		for i := range 128 {
			raw[144+16+i] = byte(i + shift)
		}
		if !q4kSame(t, "every nibble byte in every lane", raw, 1) {
			return
		}
	}
}

// TestDequantQ4KBlock_realFile holds the kernel to the scalar on every Q4_K block of a real GGUF, named by
// AIKIT_GGUF_ORACLE. It skips without one: no real checkpoint lives in this repository.
func TestDequantQ4KBlock_realFile(t *testing.T) {
	path := os.Getenv("AIKIT_GGUF_ORACLE")
	if path == "" {
		t.Skip("set AIKIT_GGUF_ORACLE to a Q4_K GGUF")
	}
	g, err := OpenGGUFMmap(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	tensors, blocks := 0, 0
	for name, info := range g.tensors {
		if info.typ != ggmlTypeQ4_K {
			continue
		}
		n := 1
		for _, d := range info.dims {
			n *= int(d)
		}
		raw, err := g.tensorBytes(info, n)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		for sb := 0; sb < n/256; sb++ {
			if !q4kSame(t, name, raw, sb) {
				return
			}
			blocks++
		}
		tensors++
	}
	if blocks == 0 {
		t.Fatalf("%s holds no Q4_K block: nothing was compared", path)
	}
	t.Logf("%d Q4_K tensors, %d blocks (%d weights): the kernel's weights are the scalar's", tensors, blocks, blocks*256)
	// One checksum of every weight's bits, to set beside another architecture's: the products are exact, so a file with
	// finite scales dequantizes to the same weights everywhere.
	names := make([]string, 0, len(g.tensors))
	for name, info := range g.tensors {
		if info.typ == ggmlTypeQ4_K {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	h := crc32.NewIEEE()
	var w [256]float32
	buf := make([]byte, 4*256)
	for _, name := range names {
		info := g.tensors[name]
		n := 1
		for _, d := range info.dims {
			n *= int(d)
		}
		raw, _ := g.tensorBytes(info, n)
		for sb := 0; sb < n/256; sb++ {
			dequantQ4KBlock(raw, sb, w[:])
			for i, x := range w {
				binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(x))
			}
			h.Write(buf)
		}
	}
	t.Logf("crc32 of every Q4_K weight's bits, tensors in name order: %08x", h.Sum32())
}
