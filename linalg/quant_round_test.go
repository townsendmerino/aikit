package linalg

import (
	"bytes"
	"math"
	"math/rand"
	"os"
	"sync"
	"sync/atomic"
	"testing"
)

// roundExhaustive reports whether the tests below sweep every float32 (AIKIT_ROUND_EXHAUSTIVE=1) or a sample.
func roundExhaustive() bool { return os.Getenv("AIKIT_ROUND_EXHAUSTIVE") == "1" }

// TestRoundInt4_matchesMathRound: roundInt4(x) is int(math.Round(float64(x))) on this architecture for the
// half-integers and their neighbours, zeros, subnormals, the infinities, NaNs, values past the int64 range, and a
// stride through every float32 bit pattern; with AIKIT_ROUND_EXHAUSTIVE=1, for all 2^32 patterns.
func TestRoundInt4_matchesMathRound(t *testing.T) {
	ref := func(x float32) int { return int(math.Round(float64(x))) }
	var bad atomic.Int64
	check := func(bits uint32) {
		x := math.Float32frombits(bits)
		if got, want := roundInt4(x), ref(x); got != want && bad.Add(1) <= 5 {
			t.Errorf("roundInt4(%g, bits %08x) = %d, want %d", x, bits, got, want)
		}
	}
	for _, v := range []float32{0, float32(math.Copysign(0, -1)), math.SmallestNonzeroFloat32, 1e-30, 0.25, 0.49999997, 0.5, 0.50000006,
		1, 1.5, 2.5, 6.4999995, 6.5, 6.5000005, 7, 7.0000005, 7.5, 8, 8388607.5, 16777216, 1e18, 1e19, 3e38,
		float32(math.Inf(1)), float32(math.NaN())} {
		for _, b := range []uint32{math.Float32bits(v), math.Float32bits(v) - 1, math.Float32bits(v) + 1} {
			check(b)
			check(b ^ 0x80000000)
		}
	}
	check(0x7FC00001) // a NaN with a payload
	check(0xFFFFFFFF)
	stride := uint64(4099)
	if roundExhaustive() {
		stride = 1
	}
	var wg sync.WaitGroup
	const parts = 64
	for p := range parts {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for b := uint64(p)<<26 + uint64(p)%stride; b < uint64(p+1)<<26; b += stride {
				check(uint32(b))
			}
		}()
	}
	wg.Wait()
	if n := bad.Load(); n != 0 {
		t.Fatalf("%d mismatches (stride %d)", n, stride)
	}
}

// quantizeGroupInt4RowReference is QuantizeGroupInt4Row's default scheme with its rounding written as
// int(math.Round(float64(x))), kept here as the oracle the shipped kernel must equal byte for byte.
func quantizeGroupInt4RowReference(row []float32, cols, group int, packed []byte, scales []float32) {
	nGroups := (cols + group - 1) / group
	for g := range nGroups {
		ks := g * group
		ke := min(ks+group, cols)
		var maxAbs float32
		for k := ks; k < ke; k++ {
			v := row[k]
			if v < 0 {
				v = -v
			}
			if v > maxAbs {
				maxAbs = v
			}
		}
		s := float32(1)
		if maxAbs > 0 {
			s = maxAbs / 7
		}
		scales[g] = s
		inv := 1.0 / s
		for k := ks; k < ke; k++ {
			q := int(math.Round(float64(row[k] * inv)))
			if q > 7 {
				q = 7
			} else if q < -7 {
				q = -7
			}
			nib := byte(q+8) & 0x0F
			if k&1 == 0 {
				packed[k/2] = (packed[k/2] &^ 0x0F) | nib
			} else {
				packed[k/2] = (packed[k/2] &^ 0xF0) | (nib << 4)
			}
		}
	}
}

// TestQuantizeGroupInt4Row_matchesReference: the kernel writes the reference's packed bytes and scales, bit for bit,
// for random rows at scales from 1e-6 to 1e6, groups that are all zero, all equal or hold one huge outlier, signed
// zeros, subnormals, infinities and NaN, at even and odd widths and several group sizes, into buffers that start out
// non-zero (so the pad nibble's contract is part of the comparison); and for single-element groups over a stride of
// the float32s in [-8, 8] (all of them with AIKIT_ROUND_EXHAUSTIVE=1).
func TestQuantizeGroupInt4Row_matchesReference(t *testing.T) {
	if int4WeightScheme != "" {
		t.Skip("an alternative int4 scheme is selected; the reference is the default scheme's")
	}
	same := func(name string, row []float32, cols, group int) {
		t.Helper()
		nGroups, bpr := groupsFor(cols, group)
		pw, pg := bytes.Repeat([]byte{0xA5}, bpr), bytes.Repeat([]byte{0xA5}, bpr)
		sw, sg := make([]float32, nGroups), make([]float32, nGroups)
		quantizeGroupInt4RowReference(row, cols, group, pw, sw)
		QuantizeGroupInt4Row(row, cols, group, pg, sg)
		if !bytes.Equal(pw, pg) {
			t.Fatalf("%s (cols %d, group %d): packed differs from the reference", name, cols, group)
		}
		for g := range sw {
			if math.Float32bits(sw[g]) != math.Float32bits(sg[g]) {
				t.Fatalf("%s (cols %d, group %d): scale %d is %g, the reference's %g", name, cols, group, g, sg[g], sw[g])
			}
		}
	}
	rng := rand.New(rand.NewSource(20261010))
	inf, nan := float32(math.Inf(1)), float32(math.NaN())
	for _, cols := range []int{1, 2, 31, 32, 33, 64, 95, 257, 1536} {
		for _, group := range []int{1, 7, 32, 64} {
			row := make([]float32, cols)
			for _, scale := range []float64{1e-6, 1e-3, 0.02, 1, 1e3, 1e6} {
				for i := range row {
					row[i] = float32(rng.NormFloat64() * scale)
				}
				same("normal", row, cols, group)
				row[rng.Intn(cols)] = float32(scale * 1e6)
				same("one huge outlier", row, cols, group)
			}
			clear(row)
			same("all zero", row, cols, group)
			for i := range row {
				row[i] = float32(math.Copysign(0, -1))
			}
			same("all negative zero", row, cols, group)
			for i := range row {
				row[i] = -0.75
			}
			same("all equal", row, cols, group)
			for i := range row {
				row[i] = float32(i%15-7) * 0.5 // every half-integer multiple of the scale when the group holds a 3.5
			}
			same("ties", row, cols, group)
			for i := range row {
				row[i] = math.SmallestNonzeroFloat32 * float32(i%5)
			}
			same("subnormals", row, cols, group)
			for _, special := range []float32{inf, -inf, nan} {
				for i := range row {
					row[i] = float32(rng.NormFloat64())
				}
				row[cols/2] = special
				same("a non-finite element", row, cols, group)
				for i := range row {
					row[i] = special
				}
				same("all non-finite", row, cols, group)
			}
		}
	}
	// Single-element groups: the scale is |x|/7 and the product x*inv is rounded on its own.
	stride := uint32(1021)
	if roundExhaustive() {
		stride = 1
	}
	var one [1]float32
	var pw, pg [1]byte
	var sw, sg [1]float32
	for b := uint32(0); b <= math.Float32bits(8); b += stride {
		for _, sign := range []uint32{0, 0x80000000} {
			one[0] = math.Float32frombits(b | sign)
			pw[0], pg[0] = 0xA5, 0xA5
			quantizeGroupInt4RowReference(one[:], 1, 1, pw[:], sw[:])
			QuantizeGroupInt4Row(one[:], 1, 1, pg[:], sg[:])
			if pw != pg || math.Float32bits(sw[0]) != math.Float32bits(sg[0]) {
				t.Fatalf("single element %g (bits %08x): packed %02x scale %g, the reference's %02x and %g", one[0], b|sign, pg[0], sg[0], pw[0], sw[0])
			}
		}
	}
}
