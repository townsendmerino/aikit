//go:build arm64

package linalg

import (
	"math/rand/v2"
	"testing"
)

// TestDotW4A8SplitHalf4RowFoldF16_matchesF32 is gate 2 of the L1 arm64 fix for the fused kernel (goinfer
// docs/tasks/task-cpu-decode-peer-gap-2026-09.md, "L1 arm64 fix — pre-registered"): over binary16 scales it
// equals dotW4A8SplitHalf4RowFold fed the widened scales, exactly, for every nGroups 1..64, including subnormal
// scales. Two mutation checks follow.
//   - One ulp up on a single scale: the fused and f32 kernels must still agree exactly, and most such changes
//     move the output (more than half is required; the count is logged, 224 of 256 when written). Not every
//     one can: the group's product changes by about dot·ulp(scale), which can round away in the f32 sum of
//     the other groups.
//   - The same scale set to ±128, dominant over every other: this must move exactly the row that owns it, which
//     shows each row reads its own lane of the widened quad.
//
// The per-path one-ulp mutation of the pre-registration is TestW4A8F16_everyPathMatchesF32's.
func TestDotW4A8SplitHalf4RowFoldF16_matchesF32(t *testing.T) {
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	rng := rand.New(rand.NewPCG(28, 9))
	var ulpTried, ulpMoved int
	for nGroups := 1; nGroups <= 64; nGroups++ {
		act := make([]int8, 32*nGroups)
		for i := range act {
			act[i] = int8(rng.IntN(256) - 128)
		}
		packed4 := make([]byte, 4*16*nGroups)
		for i := range packed4 {
			packed4[i] = byte(rng.IntN(256))
		}
		s16 := make([]uint16, 4*nGroups)
		for i := range s16 {
			switch rng.IntN(8) {
			case 0: // subnormal
				s16[i] = uint16(1 + rng.IntN(0x3FF))
			default: // normal, a scale's usual magnitude (about 2^-12 .. 2^2)
				s16[i] = uint16(0x0C00 + rng.IntN(0x4400-0x0C00))
			}
			if rng.IntN(2) == 0 {
				s16[i] |= 0x8000
			}
		}
		corr := make([]int32, 4*nGroups)
		w4a8LaneCorrNeg8(&act[0], &corr[0], nGroups)

		want := fold4F32(act, corr, packed4, s16, nGroups)
		var got [4]float32
		dotW4A8SplitHalf4RowFoldF16(&act[0], &corr[0], &packed4[0], &s16[0], &got[0], nGroups)
		if got != want {
			t.Fatalf("nGroups=%d: fused f16 %v != f32 kernel on the widened scales %v", nGroups, got, want)
		}

		// Mutations: each row's scale in a random group.
		g := rng.IntN(nGroups)
		for r := 0; r < 4; r++ {
			m16 := append([]uint16(nil), s16...)
			m16[4*g+r]++
			var gm [4]float32
			dotW4A8SplitHalf4RowFoldF16(&act[0], &corr[0], &packed4[0], &m16[0], &gm[0], nGroups)
			if wm := fold4F32(act, corr, packed4, m16, nGroups); gm != wm {
				t.Fatalf("nGroups=%d one-ulp row %d: fused %v != f32 %v", nGroups, r, gm, wm)
			}
			ulpTried++
			if gm[r] != got[r] {
				ulpMoved++
			}

			d16 := append([]uint16(nil), s16...)
			d16[4*g+r] = s16[4*g+r]&0x8000 | 0x5800 // ±128
			var gd [4]float32
			dotW4A8SplitHalf4RowFoldF16(&act[0], &corr[0], &packed4[0], &d16[0], &gd[0], nGroups)
			if wd := fold4F32(act, corr, packed4, d16, nGroups); gd != wd {
				t.Fatalf("nGroups=%d dominant row %d: fused %v != f32 %v", nGroups, r, gd, wd)
			}
			for rr := 0; rr < 4; rr++ {
				if rr != r && gd[rr] != got[rr] {
					t.Fatalf("nGroups=%d: changing row %d's scale moved row %d (lane mix-up)", nGroups, r, rr)
				}
			}
			if gd[r] == got[r] && g16dot(act, packed4, corr, g, r) != 0 {
				t.Fatalf("nGroups=%d: row %d's scale set dominant but its output did not move", nGroups, r)
			}
		}
	}
	// Non-vacuity only: which one-ulp changes round away depends on the data (a subnormal scale's ulp is tiny
	// against the other groups). The discriminating check is the dominant mutation above.
	if ulpMoved*2 <= ulpTried {
		t.Fatalf("only %d of %d one-ulp scale changes moved the output", ulpMoved, ulpTried)
	}
	t.Logf("one-ulp scale changes that moved the output: %d of %d", ulpMoved, ulpTried)
}

// fold4F32 is dotW4A8SplitHalf4RowFold on s16 widened by the scalar f16ToF32 (independent of widenF16's asm).
func fold4F32(act []int8, corr []int32, packed4 []byte, s16 []uint16, nGroups int) [4]float32 {
	s32 := make([]float32, len(s16))
	for i, h := range s16 {
		s32[i] = f16ToF32(h)
	}
	var out [4]float32
	dotW4A8SplitHalf4RowFold(&act[0], &corr[0], &packed4[0], &s32[0], &out[0], nGroups)
	return out
}

// g16dot is row r's integer dot product in group g, as the kernel forms it (corr lanes included): zero means
// a scale change cannot move that group's contribution.
func g16dot(act []int8, packed4 []byte, corr []int32, g, r int) int32 {
	blk := packed4[g*64+r*16 : g*64+r*16+16]
	var s int32
	for l := 0; l < 4; l++ {
		lane := corr[4*g+l]
		for i := 0; i < 4; i++ {
			b := blk[4*l+i]
			lane += int32(b&0x0F)*int32(act[32*g+4*l+i]) + int32(b>>4)*int32(act[32*g+16+4*l+i])
		}
		s += lane
	}
	return s
}
