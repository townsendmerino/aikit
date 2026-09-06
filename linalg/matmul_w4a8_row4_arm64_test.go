//go:build arm64

package linalg

import (
	"math/rand"
	"testing"
)

// TestMatmulBTW4A8Row4Into_bitIdenticalToMatmulBTW4A8Into is the production
// entry point's end-to-end proof, one level above
// TestDotW4A8SplitHalf4Row_bitIdenticalToCanonical's raw-kernel check:
// RepackW4A8Row4/RepackW4A8Row4Scales followed by MatmulBTW4A8Row4Into must
// produce EXACTLY the same dst as MatmulBTW4A8Into on the unrepacked
// canonical data, for the same logical weight matrix. Covers several
// (N, K) shapes, all satisfying the entry point's N%4==0/K%32==0 contract,
// including N not a multiple of the parallel-fanout width so the serial and
// parallel code paths both get exercised.
func TestMatmulBTW4A8Row4Into_bitIdenticalToMatmulBTW4A8Into(t *testing.T) {
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	const group = 32
	shapes := []struct{ K, N int }{
		{32, 4}, {64, 8}, {1536, 8960}, {8960, 1536}, {96, 20},
	}
	rng := rand.New(rand.NewSource(53))
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
		MatmulBTW4A8Into(&ws, a, packed, scales, want, 1, K, N, group)

		var ws2 Workspace
		got := make([]float32, N)
		MatmulBTW4A8Row4Into(&ws2, a, packed4, scales4, got, 1, K, N, group)

		for j := range N {
			if got[j] != want[j] {
				t.Fatalf("K=%d N=%d j=%d: MatmulBTW4A8Row4Into %v != MatmulBTW4A8Into %v (bit mismatch)", K, N, j, got[j], want[j])
			}
		}
	}
}

// TestMatmulBTW4A8Row4Into_parallelMatchesSerial checks that forcing the
// parallel fan-out (ws.SetThreshold(0)) gives the identical result to the
// serial fast path, at a real shape with several worker counts — the
// parallel branch partitions output QUADS, not individual rows, so this is
// a genuinely different code path from MatmulBTW4A8Into's own
// column-partitioned parallel branch and needs its own proof.
func TestMatmulBTW4A8Row4Into_parallelMatchesSerial(t *testing.T) {
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	const (
		K     = 1536
		N     = 8960
		group = 32
	)
	rng := rand.New(rand.NewSource(59))
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

	var wsSerial Workspace
	wsSerial.SetThreshold(1 << 30) // force serial
	serial := make([]float32, N)
	MatmulBTW4A8Row4Into(&wsSerial, a, packed4, scales4, serial, 1, K, N, group)

	for _, workers := range []int{2, 4, 6, 8} {
		var ws Workspace
		ws.SetWorkers(workers)
		ws.SetThreshold(0)
		got := make([]float32, N)
		MatmulBTW4A8Row4Into(&ws, a, packed4, scales4, got, 1, K, N, group)
		for j := range N {
			if got[j] != serial[j] {
				t.Fatalf("workers=%d j=%d: parallel %v != serial %v (bit mismatch)", workers, j, got[j], serial[j])
			}
		}
	}
}
