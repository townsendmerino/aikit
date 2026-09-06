//go:build arm64

package linalg

import (
	"math/rand"
	"testing"
)

// TestMatmulBTW4A8Row4PrefetchInto_bitIdenticalToRow4 proves the PRFM remedy
// (docs/task-w4a8-neon-bandwidth.md's cold-fix harness pass) changes no
// numeric result at any swept distance — PRFM never faults and never
// touches a register any SDOT/FMLA reads, so this should hold trivially, but
// it's proven the same way every other kernel variant in this file is,
// across every (K,N) shape MatmulBTW4A8Row4Into itself is tested against
// (every residue this campaign's harness convention checks), not assumed
// from the assembly alone.
func TestMatmulBTW4A8Row4PrefetchInto_bitIdenticalToRow4(t *testing.T) {
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	const group = 32
	shapes := []struct{ K, N int }{
		{32, 4}, {64, 8}, {1536, 8960}, {8960, 1536}, {96, 20},
	}
	distances := []int{0, 64, 128, 256, 512, 4096}
	rng := rand.New(rand.NewSource(83))
	for _, sh := range shapes {
		K, N := sh.K, sh.N
		a := make([]float32, K)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		w := make([]float32, N*K)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		packed, scales := QuantizeGroupsInt4(w, N, K, group)
		packed4 := RepackW4A8Row4(packed, N, K, group)
		scales4 := RepackW4A8Row4Scales(scales, N, K, group)

		var ws Workspace
		want := make([]float32, N)
		MatmulBTW4A8Row4Into(&ws, a, packed4, scales4, want, 1, K, N, group)

		for _, dist := range distances {
			var ws2 Workspace
			got := make([]float32, N)
			MatmulBTW4A8Row4PrefetchInto(&ws2, a, packed4, scales4, got, 1, K, N, group, dist)
			for j := range N {
				if got[j] != want[j] {
					t.Fatalf("K=%d N=%d dist=%d j=%d: Prefetch %v != Row4 %v (bit mismatch)", K, N, dist, j, got[j], want[j])
				}
			}
		}
	}
}

// TestMatmulBTW4A8Row4DesharedInto_bitIdenticalToRow4 proves the de-sharing
// remedy is bit-identical to the production row4 kernel — same per-row math,
// only which pointer each row reads from differs (the same bit-identity bar
// the original row4 layout change was held to:
// TestDotW4A8SplitHalf4Row_bitIdenticalToCanonical).
func TestMatmulBTW4A8Row4DesharedInto_bitIdenticalToRow4(t *testing.T) {
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	const group = 32
	shapes := []struct{ K, N int }{
		{32, 4}, {64, 8}, {1536, 8960}, {8960, 1536}, {96, 20},
	}
	rng := rand.New(rand.NewSource(89))
	for _, sh := range shapes {
		K, N := sh.K, sh.N
		a := make([]float32, K)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		w := make([]float32, N*K)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		packed, scales := QuantizeGroupsInt4(w, N, K, group)
		packed4 := RepackW4A8Row4(packed, N, K, group)
		scales4 := RepackW4A8Row4Scales(scales, N, K, group)

		var ws Workspace
		want := make([]float32, N)
		MatmulBTW4A8Row4Into(&ws, a, packed4, scales4, want, 1, K, N, group)

		w0, w1, w2, w3 := RepackW4A8Row4Deshared(packed, N, K, group)
		s0, s1, s2, s3 := RepackW4A8Row4DesharedScales(scales, N, K, group)

		var ws2 Workspace
		got := make([]float32, N)
		MatmulBTW4A8Row4DesharedInto(&ws2, a, w0, w1, w2, w3, s0, s1, s2, s3, got, 1, K, N, group)
		for j := range N {
			if got[j] != want[j] {
				t.Fatalf("K=%d N=%d j=%d: Deshared %v != Row4 %v (bit mismatch)", K, N, j, got[j], want[j])
			}
		}
	}
}
