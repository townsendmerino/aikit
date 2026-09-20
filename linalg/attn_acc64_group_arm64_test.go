//go:build arm64

package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// The G=6 NEON kernel against the Go kernel it replaces, called directly so
// no dispatch decision can hide a defect — same adversarial-data convention
// as attn_acc64_arm64_test.go (raw bits, not ==, spanning ~60 binades).

// avGoBlockG is the Go definition of one 8-dim block for G queries, written
// as the reference loop-nest (key-outer, dim-inner, query-inner — matching
// MatmulAVAcc64Group's own loop nesting so this is a block-for-block pin of
// the SAME accumulation order, not an independent implementation).
func avGoBlockG(scores, vals []float32, G, nKeys, scoresStride, rowStride, off int, out []float32, outStride int) {
	acc := make([]float64, G*8)
	for i := range acc {
		acc[i] = 0
	}
	for s := range nKeys {
		vrow := vals[off+s*rowStride : off+s*rowStride+8]
		for d := range 8 {
			vd := float64(vrow[d])
			for g := range G {
				acc[d*G+g] += float64(scores[g*scoresStride+s]) * vd
			}
		}
	}
	for g := range G {
		for d := range 8 {
			out[g*outStride+d] = float32(acc[d*G+g])
		}
	}
}

func TestAVAcc64NEON8G6_matchesGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5113))
	const G = 6
	for _, nKeys := range []int{1, 2, 3, 7, 16, 17, 130, 131, 1000} {
		for _, rowStride := range []int{8, 16, 128, 130} {
			for _, off := range []int{0, 8, rowStride - 8} {
				if off+8 > rowStride || off < 0 {
					continue
				}
				scores := adversarialF32(rng, G*nKeys)
				vals := adversarialF32(rng, nKeys*rowStride)
				const outStride = 24 // deliberately != 8, to prove dstRowStrideBytes is honored
				want := make([]float32, G*outStride)
				avGoBlockG(scores, vals, G, nKeys, nKeys, rowStride, off, want, outStride)

				got := make([]float32, G*outStride)
				for i := range got {
					got[i] = float32(math.NaN())
				}
				avAcc64NEON8G6(&scores[0], nKeys*4, &vals[off], nKeys, rowStride*4, &got[0], outStride*4)

				for g := range G {
					for d := range 8 {
						i := g*outStride + d
						if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
							t.Fatalf("nKeys=%d rowStride=%d off=%d g=%d d=%d: NEON %v (%08x) Go %v (%08x)",
								nKeys, rowStride, off, g, d, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
						}
					}
				}
			}
		}
	}
}

// The comparison must be able to fail: dropping the last key must change the
// output on adversarial data. Guards against a vacuous oracle.
func TestAVAcc64NEON8G6_mutationDetected(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5114))
	const G, nKeys, rowStride = 6, 37, 64
	scores := adversarialF32(rng, G*nKeys)
	vals := adversarialF32(rng, nKeys*rowStride)
	const outStride = 8
	got := make([]float32, G*outStride)
	avAcc64NEON8G6(&scores[0], nKeys*4, &vals[0], nKeys, rowStride*4, &got[0], outStride*4)

	mut := make([]float32, G*outStride)
	avGoBlockG(scores, vals, G, nKeys-1, nKeys, rowStride, 0, mut, outStride)

	same := 0
	for i := range got {
		if math.Float32bits(got[i]) == math.Float32bits(mut[i]) {
			same++
		}
	}
	if same == len(got) {
		t.Fatal("AV G6: dropping the last key changed no output bits — the oracle is vacuous")
	}
}

// qkGoBlockG is the Go definition of one block of 8 keys for G queries,
// written as the reference loop-nest (d-outer, key-inner, query-inner —
// matching MatmulQKAcc64Group's own nesting so this pins the SAME
// accumulation order, not an independent implementation).
func qkGoBlockG(q, rows []float32, G, K, qStride, rowStride int, out []float32, outStride int) {
	acc := make([]float64, G*8)
	for i := range acc {
		acc[i] = 0
	}
	for d := range K {
		for j := range 8 {
			rd := float64(rows[j*rowStride+d])
			for g := range G {
				acc[g*8+j] += float64(q[g*qStride+d]) * rd
			}
		}
	}
	for g := range G {
		for j := range 8 {
			out[g*outStride+j] = float32(acc[g*8+j])
		}
	}
}

func TestQKAcc64NEON8G6_matchesGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5116))
	const G = 6
	for _, K := range []int{4, 8, 12, 64, 80, 96, 128, 256} {
		for _, nBlocks := range []int{1, 2, 3, 9} {
			for _, rowStride := range []int{K, K + 4, 2 * K} {
				q := adversarialF32(rng, G*K)
				rows := adversarialF32(rng, nBlocks*8*rowStride)
				outStride := nBlocks*8 + 8 // deliberately != nBlocks*8, to prove dstRowStrideBytes is honored
				want := make([]float32, G*outStride)
				for b := range nBlocks {
					blk := make([]float32, G*8)
					qkGoBlockG(q, rows[b*8*rowStride:], G, K, K, rowStride, blk, 8)
					for g := range G {
						copy(want[g*outStride+b*8:g*outStride+b*8+8], blk[g*8:g*8+8])
					}
				}

				got := make([]float32, G*outStride)
				for i := range got {
					got[i] = float32(math.NaN())
				}
				qkAcc64NEON8G6(&q[0], K*4, &rows[0], rowStride*4, K/4, nBlocks, &got[0], outStride*4)

				for g := range G {
					for j := range nBlocks * 8 {
						i := g*outStride + j
						if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
							t.Fatalf("K=%d nBlocks=%d rowStride=%d g=%d key=%d: NEON %v (%08x) Go %v (%08x)",
								K, nBlocks, rowStride, g, j, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
						}
					}
				}
			}
		}
	}
}

// The comparison must be able to fail: dropping the last dim must change the
// output on adversarial data. Guards against a vacuous oracle.
func TestQKAcc64NEON8G6_mutationDetected(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5117))
	const G, K, rowStride = 6, 128, 130
	q := adversarialF32(rng, G*K)
	rows := adversarialF32(rng, 8*rowStride+K)
	const outStride = 8
	got := make([]float32, G*outStride)
	qkAcc64NEON8G6(&q[0], K*4, &rows[0], rowStride*4, K/4, 1, &got[0], outStride*4)

	mut := make([]float32, G*outStride)
	qkGoBlockG(q, rows, G, K-4, K, rowStride, mut, outStride) // drop the last d-quad

	same := 0
	for i := range got {
		if math.Float32bits(got[i]) == math.Float32bits(mut[i]) {
			same++
		}
	}
	if same == len(got) {
		t.Fatal("QK G6: dropping the last d-quad changed no output bits — the oracle is vacuous")
	}
}

// End to end through MatmulQKAcc64Group on shapes that mix NEON blocks with
// the Go remainder (N%8), confirming G=6 vs the ungrouped kernel agrees
// exactly with the NEON dispatch actually firing (G=6, K%4==0, N>=8).
func TestQKAcc64NEON8G6_publicKernelMixedShapes(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5118))
	const G = 6
	for _, sh := range []struct{ N, K int }{
		{8, 64}, {9, 64}, {16, 128}, {17, 128}, {100, 96}, {257, 128}, {40, 256}, {130, 128},
	} {
		const nKV = 3
		kvh := 1
		kvDim := nKV * sh.K
		bOff := kvh * sh.K

		a := adversarialF32(rng, G*sh.K)
		bMat := adversarialF32(rng, sh.N*kvDim)

		want := make([]float32, G*sh.N)
		for g := range G {
			MatmulQKAcc64(a[g*sh.K:(g+1)*sh.K], bMat, want[g*sh.N:(g+1)*sh.N], 1, sh.K, sh.N, bOff, kvDim)
		}

		got := make([]float32, G*sh.N)
		MatmulQKAcc64Group(a, bMat, got, G, sh.K, sh.N, bOff, kvDim)

		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%+v idx %d: grouped %08x vs ungrouped %08x", sh, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
			}
		}
	}
}

// End to end through MatmulAVAcc64Group on shapes that mix NEON blocks with
// the Go remainder (hd%8), and confirm G=6 vs the ungrouped kernel agrees
// exactly, same as TestAVAcc64Group_matchesUngrouped but forcing the NEON
// dispatch to actually fire (G=6, nKeys>0).
func TestAVAcc64NEON8G6_publicKernelMixedShapes(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5115))
	const G = 6
	for _, sh := range []struct{ nKeys, hd int }{
		{1, 8}, {1, 16}, {1, 12}, {17, 128}, {100, 96}, {257, 64}, {40, 20}, {70, 12},
	} {
		const nKV = 3
		kvh := 1
		rowStride := nKV * sh.hd
		headOff := kvh * sh.hd

		scores := adversarialF32(rng, G*sh.nKeys)
		vals := adversarialF32(rng, sh.nKeys*rowStride)

		want := make([]float32, G*sh.hd)
		wantAcc := make([]float64, sh.hd)
		for g := range G {
			MatmulAVAcc64(scores[g*sh.nKeys:(g+1)*sh.nKeys], vals, want[g*sh.hd:(g+1)*sh.hd], wantAcc, 1, sh.nKeys, sh.hd, headOff, rowStride)
		}

		got := make([]float32, G*sh.hd)
		gotAcc := make([]float64, G*sh.hd)
		MatmulAVAcc64Group(scores, vals, got, gotAcc, G, sh.nKeys, sh.hd, headOff, rowStride)

		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%+v idx %d: grouped %08x vs ungrouped %08x", sh, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
			}
		}
	}
}
