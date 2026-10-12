package linalg

import (
	"math"
	"testing"
)

// widenBF16 is the 16-bit WeightMat's inner loop on every architecture (AVX2 on amd64, NEON on arm64, scalar elsewhere). Widening is a shift,
// so the contract is exact: every output is the input pattern in the high half of the f32, every bit, NaN payloads and subnormals included.
func scalarBF16(u uint16) float32 { return math.Float32frombits(uint32(u) << 16) }

// All 65,536 patterns, as one long vector (the vector body, with a ragged tail) and from every start offset 0..7 (a misaligned load and store).
func TestWidenBF16_exhaustive(t *testing.T) {
	src := make([]uint16, 1<<16)
	for i := range src {
		src[i] = uint16(i)
	}
	for off := 0; off < 8; off++ {
		s := src[off:]
		dst := make([]float32, len(s)+off)
		widenBF16(dst[off:], s)
		for i, u := range s {
			if got, want := math.Float32bits(dst[off+i]), math.Float32bits(scalarBF16(u)); got != want {
				t.Fatalf("offset %d, pattern %#04x: widened to %#08x, want %#08x", off, u, got, want)
			}
		}
	}
}

// Every length across the code's boundaries (under eight, one block, one vector pair, ragged tails), with sentinels on both sides of dst that
// the routine must not touch (the overlapping last block writes only inside dst[:n]).
func TestWidenBF16_everyLengthAndNoOverrun(t *testing.T) {
	const guard = 16
	pat := make([]uint16, 200)
	for i := range pat {
		pat[i] = uint16(i*257 + 3)
	}
	for n := 0; n <= 70; n++ {
		buf := make([]float32, n+2*guard)
		for i := range buf {
			buf[i] = math.Float32frombits(0xDEADBEEF)
		}
		widenBF16(buf[guard:guard+n], pat[:n])
		for i := range buf {
			var want uint32 = 0xDEADBEEF
			if i >= guard && i < guard+n {
				want = math.Float32bits(scalarBF16(pat[i-guard]))
			}
			if got := math.Float32bits(buf[i]); got != want {
				t.Fatalf("n=%d: dst[%d] = %#08x, want %#08x (guard=%d)", n, i, got, want, guard)
			}
		}
	}
}

// dst longer than src: only dst[:len(src)] is written.
func TestWidenBF16_dstLongerThanSrc(t *testing.T) {
	src := []uint16{0x3f80, 0x4000, 0xbf80, 0x7fc0, 0x0001, 0x8000, 0x7f80, 0xff80, 0x4049, 0x1234}
	dst := make([]float32, 14)
	for i := range dst {
		dst[i] = 1234
	}
	widenBF16(dst, src)
	for i, u := range src {
		if math.Float32bits(dst[i]) != math.Float32bits(scalarBF16(u)) {
			t.Errorf("dst[%d] = %v, want %v", i, dst[i], scalarBF16(u))
		}
	}
	for i := len(src); i < len(dst); i++ {
		if dst[i] != 1234 {
			t.Errorf("dst[%d] (past src) was written: %v", i, dst[i])
		}
	}
}

func BenchmarkWidenBF16(b *testing.B) {
	const n = 32 * 4096 // one 32-row tile of a 4096-wide matrix, the shape matmulHalf16 widens
	src := make([]uint16, n)
	for i := range src {
		src[i] = uint16(i)
	}
	dst := make([]float32, n)
	b.SetBytes(int64(n) * 2)
	for b.Loop() {
		widenBF16(dst, src)
	}
}

func BenchmarkWidenBF16_scalarLoop(b *testing.B) {
	const n = 32 * 4096
	src := make([]uint16, n)
	for i := range src {
		src[i] = uint16(i)
	}
	dst := make([]float32, n)
	b.SetBytes(int64(n) * 2)
	for b.Loop() {
		for i, u := range src {
			dst[i] = math.Float32frombits(uint32(u) << 16)
		}
	}
}
