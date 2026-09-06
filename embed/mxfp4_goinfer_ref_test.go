package embed

import (
	"math"
	"testing"
)

// M2 of docs/task-goinfer-kernel-moves.md. THIS FILE IS THE GATE, AND IT IS WRITTEN BEFORE THE
// CODE IT GATES — the move's stated order: land in aikit with a reference implementation and a
// raw-bit gate, then bump goinfer and delete its copy. Never the other way.
//
// What moves: goinfer's decoder/mxfp4.go holds an MXFP4 (OCP FP4, ggml type 39) dequantizer whose
// e2m1 table and e8m0 scale conversion are already byte-identical to this package's, plus ONE
// thing this package does not have — the SAFETENSORS layout, where the packed nibbles and the
// block scales arrive as two separate tensors and the intra-block nibble order is different.
//
// THE TRAP THIS GATE EXISTS TO PIN. The two layouts share the block size, the scale encoding and
// the value table, so they look like the same function with different addressing. They are not:
//
//	GGML (contiguous 17-byte blocks): byte j packs elements j and j+16
//	safetensors (split tensors):      byte j packs elements 2j and 2j+1
//
// goinfer measured this rather than assuming it, and its assumption had been wrong: dequantizing a
// real gpt-oss expert both ways and diffing against the same weight read through the validated
// GGUF path gave cosine 0.081 for GGML order and 1.000000 for sequential. Routing the split layout
// through the GGML core produces finite, plausibly-scaled, completely wrong weights. So the gate
// below asserts BOTH that each order matches its reference bit-for-bit AND that the two orders
// genuinely disagree — because the tidying that merges them is the failure mode.

// ---- Frozen reference: transcribed from goinfer decoder/mxfp4.go at 4f5da73c. -----------------
// These are the bodies being replaced. They are frozen HERE so the gate keeps comparing against
// what goinfer actually shipped, not against whatever this package's own implementation drifts to.

var refMXFP4KValues = [16]int8{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}

func refE8M0ToF32Half(x uint8) float32 {
	var bits uint32
	if x < 2 {
		bits = uint32(0x00200000) << uint32(x)
	} else {
		bits = uint32(x-1) << 23
	}
	return math.Float32frombits(bits)
}

// refDequantSplitInto is goinfer's mxfp4DequantSplitInto, arithmetic unchanged.
func refDequantSplitInto(blocks, scales []byte, nBlocks int, dst []float32) {
	for b := range nBlocks {
		d := refE8M0ToF32Half(scales[b])
		qs := blocks[b*16 : b*16+16]
		row := dst[b*32 : (b+1)*32]
		for j := range 16 {
			v := qs[j]
			row[2*j] = d * float32(refMXFP4KValues[v&0x0F])
			row[2*j+1] = d * float32(refMXFP4KValues[v>>4])
		}
	}
}

// refDequantBlocks is goinfer's mxfp4Dequant/mxfp4DequantBlock (GGML contiguous order).
func refDequantBlocks(raw []byte, nBlocks int, dst []float32) {
	for i := range nBlocks {
		base := i * 17
		d := refE8M0ToF32Half(raw[base])
		qs := raw[base+1 : base+17]
		out := dst[i*32 : (i+1)*32]
		for j := range 16 {
			out[j] = d * float32(refMXFP4KValues[qs[j]&0x0F])
			out[j+16] = d * float32(refMXFP4KValues[qs[j]>>4])
		}
	}
}

// ---- Vectors. aikit has no Python, so every fixture is a committed Go test vector. -------------

// mxfp4Vectors builds nBlocks of packed nibbles plus their scale bytes, deterministically and
// without an RNG so the bytes are reproducible from the source alone. Every one of the 256 scale
// bytes appears (including the x<2 SUBNORMALS, which are the only cases where the exact bit
// formula differs from a naive 2^(x-128)), and every one of the 256 byte values appears as a
// nibble pair, so all 16 e2m1 codes are exercised in both halves.
func mxfp4Vectors(nBlocks int) (blocks, scales []byte) {
	blocks = make([]byte, nBlocks*16)
	scales = make([]byte, nBlocks)
	for b := range nBlocks {
		scales[b] = byte(b * 37 % 256) // strides the whole 0..255 range, subnormals included
		for j := range 16 {
			blocks[b*16+j] = byte((b*16 + j) % 256)
		}
	}
	return blocks, scales
}

func bitsDiff(t *testing.T, what string, got, want []float32) int {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	bad := 0
	for i := range got {
		// RAW BITS, never a tolerance: −0 vs +0 and a NaN payload both compare equal under `==`
		// and are exactly the kind of difference a move is supposed to prove absent.
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			if bad < 5 {
				t.Errorf("%s: [%d] = %08x (%v), want %08x (%v)",
					what, i, math.Float32bits(got[i]), got[i], math.Float32bits(want[i]), want[i])
			}
			bad++
		}
	}
	return bad
}

// ---- The gate ---------------------------------------------------------------------------------

// TestMXFP4Scale_bitIdenticalToGoinferRef covers all 256 e8m0 bytes, which is the whole domain.
func TestMXFP4Scale_bitIdenticalToGoinferRef(t *testing.T) {
	for x := 0; x < 256; x++ {
		got, want := MXFP4Scale(byte(x)), refE8M0ToF32Half(byte(x))
		if math.Float32bits(got) != math.Float32bits(want) {
			t.Errorf("MXFP4Scale(%d) = %08x, want %08x", x, math.Float32bits(got), math.Float32bits(want))
		}
	}
}

func TestDequantMXFP4Split_bitIdenticalToGoinferRef(t *testing.T) {
	// 1 and 2 blocks catch an off-by-one in the block stride that a large count averages away;
	// 256 makes every scale byte appear at least once.
	for _, n := range []int{1, 2, 3, 17, 256} {
		blocks, scales := mxfp4Vectors(n)
		got := make([]float32, n*32)
		if err := DequantMXFP4Split(blocks, scales, n, got); err != nil {
			t.Fatalf("n=%d: %v", n, err)
		}
		want := make([]float32, n*32)
		refDequantSplitInto(blocks, scales, n, want)
		if bad := bitsDiff(t, "split n="+itoa(n), got, want); bad > 0 {
			t.Errorf("n=%d: %d of %d values differ", n, bad, n*32)
		}
	}
}

func TestDequantMXFP4Blocks_bitIdenticalToGoinferRef(t *testing.T) {
	for _, n := range []int{1, 2, 3, 17, 256} {
		raw := make([]byte, n*17)
		for i := range raw {
			raw[i] = byte(i * 31 % 256)
		}
		got := make([]float32, n*32)
		if err := DequantMXFP4Blocks(raw, n, got); err != nil {
			t.Fatalf("n=%d: %v", n, err)
		}
		want := make([]float32, n*32)
		refDequantBlocks(raw, n, want)
		if bad := bitsDiff(t, "blocks n="+itoa(n), got, want); bad > 0 {
			t.Errorf("n=%d: %d of %d values differ", n, bad, n*32)
		}
	}
}

// TestMXFP4Orders_areNotInterchangeable is the anti-tidying gate. If someone later notices the two
// functions "differ only in addressing" and unifies them, this goes red — with the measurement
// that settled it in the failure text, so the next reader does not have to rediscover cosine 0.081.
func TestMXFP4Orders_areNotInterchangeable(t *testing.T) {
	const n = 8
	blocks, scales := mxfp4Vectors(n)

	// Feed the SAME payload to both orders by materializing the contiguous form from the split one.
	raw := make([]byte, n*17)
	for b := range n {
		raw[b*17] = scales[b]
		copy(raw[b*17+1:], blocks[b*16:(b+1)*16])
	}
	seq := make([]float32, n*32)
	ggml := make([]float32, n*32)
	if err := DequantMXFP4Split(blocks, scales, n, seq); err != nil {
		t.Fatal(err)
	}
	if err := DequantMXFP4Blocks(raw, n, ggml); err != nil {
		t.Fatal(err)
	}

	same := 0
	for i := range seq {
		if math.Float32bits(seq[i]) == math.Float32bits(ggml[i]) {
			same++
		}
	}
	if same == len(seq) {
		t.Fatal("the split (safetensors) and contiguous (GGML) orders produced IDENTICAL output on " +
			"the same payload — they must not. byte j packs elements j and j+16 in GGML and 2j and " +
			"2j+1 in safetensors; goinfer measured cosine 0.081 for GGML order on safetensors data " +
			"against 1.000000 for sequential. If these were just unified, this is the gate that " +
			"should have stopped it")
	}
	t.Logf("orders agree on %d of %d values, as expected (coincidences only)", same, len(seq))
}

// The error contract is part of the move: these are two independently-shaped tensors and a
// mismatch between them is exactly the corruption worth refusing loudly rather than panicking.
func TestDequantMXFP4Split_refusesMismatchedShapes(t *testing.T) {
	blocks, scales := mxfp4Vectors(4)
	dst := make([]float32, 4*32)
	for _, tc := range []struct {
		name           string
		blocks, scales []byte
		n              int
		dst            []float32
	}{
		{"short scales", blocks, scales[:3], 4, dst},
		{"short blocks", blocks[:48], scales, 4, dst},
		{"short dst", blocks, scales, 4, dst[:64]},
		{"negative n", blocks, scales, -1, dst},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := DequantMXFP4Split(tc.blocks, tc.scales, tc.n, tc.dst); err == nil {
				t.Error("accepted a mismatched shape instead of refusing")
			}
		})
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}
