//go:build arm64

package linalg

import (
	"math/rand"
	"testing"
)

// docs/prompts/w4a8-item3-harness.md (goinfer) — scalar oracles for the
// split-half / 4-row-interleave layouts. The repack functions themselves
// (canonicalNibble, repackSplitHalfRow, splitHalfNibble,
// repackSplitHalf4RowBlock, interleaveScales4Row) were promoted to
// production in repack_w4a8_row4_arm64.go once the harness grid recorded GO
// — only the oracles and their tests stay here.

// dotW4A8SplitHalfScalar is dotW4A8Scalar's split-half-layout twin. Same
// per-k values, same group/k iteration order, same int32 accumulator type —
// so this must be BIT-IDENTICAL to dotW4A8Scalar for the same logical
// weights, and the test below asserts exactly that (== not tolerance).
func dotW4A8SplitHalfScalar(act []int8, packedSH []byte, scales []float32, group, K int) float32 {
	var total float32
	nGroups := (K + group - 1) / group
	for g := range nGroups {
		ks := g * group
		ke := min(ks+group, K)
		var acc int32
		for k := ks; k < ke; k++ {
			nib := splitHalfNibble(packedSH, k)
			acc += int32(act[k]) * int32(int(nib)-8)
		}
		total += float32(acc) * scales[g]
	}
	return total
}

func TestRepackSplitHalf_roundtripsCanonical(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for trial := range 50 {
		nGroups := 1 + rng.Intn(20)
		K := nGroups * 32
		w := make([]float32, K)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		packed, _ := QuantizeGroupsInt4(w, 1, K, 32)
		sh := repackSplitHalfRow(packed, K)
		for k := range K {
			want := canonicalNibble(packed, k)
			got := splitHalfNibble(sh, k)
			if got != want {
				t.Fatalf("trial %d K=%d k=%d: canonical nibble %d, split-half roundtrip gave %d", trial, K, k, want, got)
			}
		}
	}
}

// dotW4A8SplitHalf4RowScalar is the scalar oracle for the interleaved
// 4-row-block layout: computes all 4 rows' dots, must match
// dotW4A8SplitHalfScalar called separately per row (same values, same
// group/k order, same per-row accumulator — this only changes memory
// layout, not arithmetic), which the test below asserts exactly.
func dotW4A8SplitHalf4RowScalar(act []int8, packed4 []byte, scales4 []float32, group, K int) [4]float32 {
	nGroups := (K + group - 1) / group
	bpr := K / 2
	var out [4]float32
	for row := range 4 {
		var total float32
		for g := range nGroups {
			ks := g * group
			ke := min(ks+group, K)
			rowBytes := packed4[g*64+row*16 : g*64+row*16+16]
			var acc int32
			for k := ks; k < ke; k++ {
				kl := k - ks // 0..31 within this group
				var nib byte
				if kl < 16 {
					nib = rowBytes[kl] & 0x0F
				} else {
					nib = (rowBytes[kl-16] >> 4) & 0x0F
				}
				acc += int32(act[k]) * int32(int(nib)-8)
			}
			total += float32(acc) * scales4[4*g+row]
		}
		_ = bpr
		out[row] = total
	}
	return out
}

func TestRepackSplitHalf4RowBlock_matchesPerRow(t *testing.T) {
	rng := rand.New(rand.NewSource(37))
	for trial := range 30 {
		nGroups := 1 + rng.Intn(15)
		K := nGroups * 32
		rows := make([][]byte, 4)
		scales := make([][]float32, 4)
		acts := make([]int8, K)
		for i := range acts {
			acts[i] = int8(rng.Intn(255) - 128)
		}
		for r := range 4 {
			w := make([]float32, K)
			for i := range w {
				w[i] = float32(rng.NormFloat64())
			}
			p, s := QuantizeGroupsInt4(w, 1, K, 32)
			rows[r] = p
			scales[r] = s
		}
		packed4 := repackSplitHalf4RowBlock(rows[0], rows[1], rows[2], rows[3], K)
		scales4 := interleaveScales4Row(scales[0], scales[1], scales[2], scales[3], nGroups)

		got := dotW4A8SplitHalf4RowScalar(acts, packed4, scales4, 32, K)
		for r := range 4 {
			want := dotW4A8Scalar(acts, rows[r], scales[r], 32, K)
			if got[r] != want {
				t.Fatalf("trial %d K=%d row=%d: got %v want %v", trial, K, r, got[r], want)
			}
		}
	}
}

func TestDotW4A8SplitHalfScalar_matchesCanonical_exact(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for trial := range 200 {
		K := 32 * (1 + rng.Intn(20))
		act := make([]int8, K)
		for i := range act {
			act[i] = int8(rng.Intn(255) - 128)
		}
		w := make([]float32, K)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		packed, scales := QuantizeGroupsInt4(w, 1, K, 32)
		sh := repackSplitHalfRow(packed, K)

		want := dotW4A8Scalar(act, packed, scales, 32, K)
		got := dotW4A8SplitHalfScalar(act, sh, scales, 32, K)
		if got != want {
			t.Fatalf("trial %d K=%d: got %v want %v (bit mismatch)", trial, K, got, want)
		}
	}
}
