//go:build arm64 || amd64

package linalg

import (
	"math"
	"sync"
	"sync/atomic"
	"testing"
)

// TestQuantizeInt4Block_matchesScalar: the vector kernel packs clamp(roundInt4(x*inv), -7, 7) + 8 for every lane,
// as quantizeInt4GroupScalar does. With inv = 1 the product is x itself, so the sweep puts each float32 bit pattern
// through the kernel's rounding, conversion and clamp directly: a stride of all 2^32 by default, every one with
// AIKIT_ROUND_EXHAUSTIVE=1, in every lane position of a block. A second pass uses other inverses, a zero and an
// infinite one among them, on the patterns near the rounding and conversion edges.
func TestQuantizeInt4Block_matchesScalar(t *testing.T) {
	nib := func(x, inv float32) byte {
		q := roundInt4(x * inv)
		if q > 7 {
			q = 7
		} else if q < -7 {
			q = -7
		}
		return byte(q+8) & 0x0F
	}
	var bad atomic.Int64
	check := func(row []float32, packed []byte, inv float32) {
		for i := range packed {
			packed[i] = 0xA5
		}
		quantizeInt4Block(row, packed, inv)
		for k, x := range row {
			got := packed[k/2] >> (4 * (k & 1)) & 0x0F
			if want := nib(x, inv); got != want && bad.Add(1) <= 5 {
				t.Errorf("x = %g (bits %08x), inv %g, lane %d: nibble %d, the scalar's %d", x, math.Float32bits(x), inv, k, got, want)
			}
		}
	}

	const block = 4 * int4QuantBlock
	stride := uint64(1543)
	if roundExhaustive() {
		stride = 1
	}
	var wg sync.WaitGroup
	const parts = 64
	for p := range parts {
		wg.Add(1)
		go func() {
			defer wg.Done()
			row, packed := make([]float32, block), make([]byte, block/2)
			n := 0
			for b := uint64(p)<<26 + uint64(p)%stride; b < uint64(p+1)<<26; b += stride {
				row[n] = math.Float32frombits(uint32(b))
				if n++; n == block {
					check(row, packed, 1)
					n = 0
				}
			}
			for ; n < block; n++ { // a last partial block, filled out with a value already checked
				row[n] = 0
			}
			check(row, packed, 1)
		}()
	}
	wg.Wait()

	// The edges, each with its neighbours and both signs, under several inverses.
	var edges []float32
	for _, v := range []float32{0, math.SmallestNonzeroFloat32, 1e-39, 0.25, 0.49999997, 0.5, 1, 1.5, 2.5, 6.4999995, 6.5, 7, 7.4999995, 7.5, 8, 127.5,
		8388607.5, 16777216, 2147483520, 2147483648, 4294967296, 9223371487098961920, 9223372036854775808, 1e30, 3.4028235e38,
		float32(math.Inf(1)), float32(math.NaN())} {
		for _, b := range []uint32{math.Float32bits(v), math.Float32bits(v) - 1, math.Float32bits(v) + 1} {
			edges = append(edges, math.Float32frombits(b), math.Float32frombits(b^0x80000000))
		}
	}
	for len(edges)%block != 0 {
		edges = append(edges, 0)
	}
	packed := make([]byte, block/2)
	for _, inv := range []float32{1, 0, 0.5, 3, 7 / 0.02, 1e-30, 1e30, float32(math.Inf(1)), float32(math.Copysign(0, -1)), float32(math.NaN())} {
		for at := 0; at < len(edges); at += block {
			check(edges[at:at+block], packed, inv)
		}
	}
	if n := bad.Load(); n != 0 {
		t.Fatalf("%d lanes differ from the scalar (stride %d)", n, stride)
	}
}
