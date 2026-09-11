//go:build amd64

package linalg

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestAccAcc64AVX_isLoadBearing is the first thing to check about a new kernel:
// that it RUNS. Both dispatchers return 0 on several decline paths (no AVX2,
// no keys, K%4 != 0, too few blocks), and a kernel that silently declined
// would leave every acc64 test passing against the pure-Go path while proving
// nothing about the assembly.
func TestAccAcc64AVX_isLoadBearing(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2 on this CPU — the kernels decline by design")
	}
	const nKeys, hd = 6, 64
	rng := rand.New(rand.NewPCG(41, 43))
	rf := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	srow := rf(nKeys)
	vals := rf(nKeys * hd)
	drow := make([]float32, hd)
	if got := avAcc64Blocks(srow, vals, drow, nKeys, hd, 0, hd); got != 64 {
		t.Errorf("avAcc64Blocks covered %d dims, want 64 — the AV kernel did not engage", got)
	}

	const K, N = 8, 12
	q := rf(K)
	b := rf(N * K)
	out := make([]float32, N)
	if got := qkAcc64Keys(q, b, out, K, N, 0, K); got != 12 {
		t.Errorf("qkAcc64Keys covered %d keys, want 12 — the QK kernel did not engage", got)
	}
}

// TestAvAcc64AVX32_matchesGo pins the AV kernel against the Go block it
// replaces, EXACTLY. The products are exact in f64 (both operands come from
// float32, so the 48-bit product fits float64's 53), which is what lets the
// kernel use FMA where the Go reference does a separate multiply and add — so
// equality, not a tolerance, is the right assertion.
func TestAvAcc64AVX32_matchesGo(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2 on this CPU")
	}
	rng := rand.New(rand.NewPCG(7, 11))
	for _, nKeys := range []int{1, 2, 7, 33} {
		const hd = 32
		srow := make([]float32, nKeys)
		for i := range srow {
			srow[i] = float32(rng.NormFloat64())
		}
		vals := make([]float32, nKeys*hd)
		for i := range vals {
			vals[i] = float32(rng.NormFloat64())
		}
		got := make([]float32, hd)
		avAcc64AVX32(&srow[0], &vals[0], nKeys, hd*4, &got[0])

		want := make([]float32, hd)
		for d := range hd {
			var acc float64
			for s := range nKeys { // key-ascending, one accumulator per output dim
				acc += float64(srow[s]) * float64(vals[s*hd+d])
			}
			want[d] = float32(acc)
		}
		for d := range hd {
			if math.Float32bits(got[d]) != math.Float32bits(want[d]) {
				t.Fatalf("nKeys=%d d=%d: kernel %v (%08x) != Go %v (%08x)",
					nKeys, d, got[d], math.Float32bits(got[d]), want[d], math.Float32bits(want[d]))
			}
		}
	}
}

// TestQkAcc64AVX4_matchesGo is the QK sibling: one key per f64 lane, dims
// ascending, compared bit-for-bit against the scalar f64 dot.
func TestQkAcc64AVX4_matchesGo(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2 on this CPU")
	}
	rng := rand.New(rand.NewPCG(13, 17))
	for _, K := range []int{4, 8, 64, 128} {
		for _, nBlocks := range []int{1, 3} {
			N := nBlocks * 4
			q := make([]float32, K)
			for i := range q {
				q[i] = float32(rng.NormFloat64())
			}
			rows := make([]float32, N*K)
			for i := range rows {
				rows[i] = float32(rng.NormFloat64())
			}
			got := make([]float32, N)
			qkAcc64AVX4(&q[0], &rows[0], K*4, K/4, nBlocks, &got[0])

			for n := range N {
				var acc float64
				for d := range K { // dim-ascending, one accumulator per key
					acc += float64(q[d]) * float64(rows[n*K+d])
				}
				want := float32(acc)
				if math.Float32bits(got[n]) != math.Float32bits(want) {
					t.Fatalf("K=%d N=%d key=%d: kernel %v (%08x) != Go %v (%08x)",
						K, N, n, got[n], math.Float32bits(got[n]), want, math.Float32bits(want))
				}
			}
		}
	}
}
