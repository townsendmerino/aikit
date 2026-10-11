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

// q6kSame runs dequantQ6KBlock and dequantQ6KBlockScalar on block sb of raw and compares the 256 weights by bits.
func q6kSame(t *testing.T, what string, raw []byte, sb int) bool {
	t.Helper()
	var got, want [256]float32
	for i := range got {
		got[i], want[i] = float32(math.NaN()), float32(math.NaN()) // every weight must be written
	}
	dequantQ6KBlockScalar(raw, sb, want[:])
	dequantQ6KBlock(raw, sb, got[:])
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Errorf("%s: weight %d is %g (bits %08x), the scalar's %g (%08x)", what, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			return false
		}
	}
	return true
}

// TestDequantQ6KBlock_matchesScalar: dequantQ6KBlock gives the scalar's weights, bit for bit: random blocks; every
// half-float as d (zeros, subnormals, both infinities and the NaNs among them) over random scales and quants; every
// byte value in every scale position; and blocks whose ql and qh bytes run through all 256 values in every lane. A
// block read at an offset checks the block index.
func TestDequantQ6KBlock_matchesScalar(t *testing.T) {
	if !dequantKernelsActive() {
		t.Skip("no vector kernel runs on this CPU: this would compare the scalar with itself (under Rosetta, set ROSETTA_ADVERTISE_AVX=1)")
	}
	rng := rand.New(rand.NewSource(20261010))
	raw := make([]byte, 3*210)
	fill := func() {
		for i := range raw {
			raw[i] = byte(rng.Intn(256))
		}
	}
	for range 20000 {
		fill()
		if !q6kSame(t, "a random block", raw, rng.Intn(3)) {
			return
		}
	}
	for h := range 1 << 16 {
		fill()
		binary.LittleEndian.PutUint16(raw[210+208:], uint16(h))
		if !q6kSame(t, "every d", raw, 1) {
			return
		}
	}
	for pos := range 16 {
		for v := range 256 {
			fill()
			raw[210+192+pos] = byte(v)
			if !q6kSame(t, "every scale byte", raw, 1) {
				return
			}
		}
	}
	for shift := range 256 {
		fill()
		for i := range 192 {
			raw[210+i] = byte(i + shift)
		}
		if !q6kSame(t, "every ql and qh byte in every lane", raw, 1) {
			return
		}
	}
}

// TestDequantQ6KBlock_realFile holds the kernel to the scalar on every Q6_K block of a real GGUF, named by
// AIKIT_GGUF_ORACLE, and logs one checksum of every weight's bits. It skips without one: no real checkpoint lives in
// this repository.
func TestDequantQ6KBlock_realFile(t *testing.T) {
	if !dequantKernelsActive() {
		t.Skip("no vector kernel runs on this CPU: this would compare the scalar with itself (under Rosetta, set ROSETTA_ADVERTISE_AVX=1)")
	}
	path := os.Getenv("AIKIT_GGUF_ORACLE")
	if path == "" {
		t.Skip("set AIKIT_GGUF_ORACLE to a GGUF with Q6_K tensors")
	}
	g, err := OpenGGUFMmap(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	var names []string
	for name, info := range g.tensors {
		if info.typ == ggmlTypeQ6_K {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	h := crc32.NewIEEE()
	var w [256]float32
	buf := make([]byte, 4*256)
	blocks := 0
	for _, name := range names {
		info := g.tensors[name]
		n := 1
		for _, d := range info.dims {
			n *= int(d)
		}
		raw, err := g.tensorBytes(info, n)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		for sb := 0; sb < n/256; sb++ {
			if !q6kSame(t, name, raw, sb) {
				return
			}
			dequantQ6KBlock(raw, sb, w[:])
			for i, x := range w {
				binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(x))
			}
			h.Write(buf)
			blocks++
		}
	}
	if blocks == 0 {
		t.Fatalf("%s holds no Q6_K block: nothing was compared", path)
	}
	t.Logf("%d Q6_K tensors, %d blocks (%d weights): the kernel's weights are the scalar's", len(names), blocks, blocks*256)
	t.Logf("crc32 of every Q6_K weight's bits, tensors in name order: %08x", h.Sum32())
}
