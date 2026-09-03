//go:build arm64

package linalg

import (
	"math/rand/v2"
	"strings"
	"testing"
)

// TestMatmulBTW4A8Row4TileInto_bitIdenticalToCanonical is the S-01 tile's own
// gate, held against the canonical M>1 path (MatmulBTW4A8Into over the SAME
// logical weights in their canonical packing) rather than against the row4 M=1
// kernel — so it proves the thing callers care about, which is that swapping the
// dispatch changes no bit of any logit.
//
// The M sweep is deliberately dense from 1 to 13: the tile consumes activation
// rows four at a time and hands M%4 to the shipped per-row kernel, so every
// residue needs a case, and 1..3 are the shapes where the tile body never runs
// at all.
func TestMatmulBTW4A8Row4TileInto_bitIdenticalToCanonical(t *testing.T) {
	if !hasDotProd {
		t.Skip("no FEAT_DotProd on this core; the row4 kernels do not dispatch")
	}
	const group = 32
	shapes := []struct{ K, N int }{
		{1536, 8960}, // production qkv/gate/up
		{8960, 1536}, // production down-proj
		{64, 8},      // minimum interesting: 2 groups, 2 quads
		{32, 4},      // single group, single quad — nQuads<2, forces the serial span
		{256, 12},    // 3 quads: an odd quad count against a 4-row activation block
	}
	rng := rand.New(rand.NewPCG(0x71, 0x1e))
	for _, sh := range shapes {
		K, N := sh.K, sh.N
		w := make([]float32, N*K)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		q4, q4s := QuantizeGroupsInt4(w, N, K, group)
		row4 := RepackW4A8Row4(q4, N, K, group)
		row4s := RepackW4A8Row4Scales(q4s, N, K, group)

		for _, M := range []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13} {
			a := make([]float32, M*K)
			for i := range a {
				a[i] = float32(rng.NormFloat64())
			}
			var wsWant, wsGot Workspace
			want := make([]float32, M*N)
			got := make([]float32, M*N)
			MatmulBTW4A8Into(&wsWant, a, q4, q4s, want, M, K, N, group)
			MatmulBTW4A8Row4TileInto(&wsGot, a, row4, row4s, got, M, K, N, group)
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("K=%d N=%d M=%d idx=%d (row %d, col %d): tile %v != canonical %v (diff %v)",
						K, N, M, i, i/N, i%N, got[i], want[i], got[i]-want[i])
				}
			}
		}
	}
}

// TestMatmulBTW4A8Row4TileInto_parallelMatchesSerial pins that the tile's quad
// partitioning is width-inert, the same contract TestParallelWidth_bitIdentical
// holds for the rest of the package: the fan-out splits output quads, never a
// reduction, so any worker count must give the same bits.
func TestMatmulBTW4A8Row4TileInto_parallelMatchesSerial(t *testing.T) {
	if !hasDotProd {
		t.Skip("no FEAT_DotProd on this core; the row4 kernels do not dispatch")
	}
	const group, K, N, M = 32, 512, 256, 7
	rng := rand.New(rand.NewPCG(0x9a, 0x77))
	w := make([]float32, N*K)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	q4, q4s := QuantizeGroupsInt4(w, N, K, group)
	row4 := RepackW4A8Row4(q4, N, K, group)
	row4s := RepackW4A8Row4Scales(q4s, N, K, group)

	var wsSerial Workspace
	wsSerial.SetThreshold(1 << 62) // never parallelize
	serial := make([]float32, M*N)
	MatmulBTW4A8Row4TileInto(&wsSerial, a, row4, row4s, serial, M, K, N, group)

	for _, workers := range []int{1, 2, 3, 4, 8} {
		var ws Workspace
		ws.SetThreshold(1)
		ws.SetWorkers(workers)
		got := make([]float32, M*N)
		MatmulBTW4A8Row4TileInto(&ws, a, row4, row4s, got, M, K, N, group)
		for i := range got {
			if got[i] != serial[i] {
				t.Fatalf("workers=%d idx=%d: %v != serial %v", workers, i, got[i], serial[i])
			}
		}
	}
}

// TestMatmulBTW4A8Row4TileInto_zeroActivationRow covers the aScale==0 shortcut
// that scatterRow4 mirrors from w4a8Span: a row that quantizes to all zeros must
// store a literal 0, and it must do so for the tiled rows and the M%4 remainder
// rows alike. The zero row is placed at index 2 so it lands inside the first
// 4-row tile, and at index 5 for a second shape so it lands in the remainder.
func TestMatmulBTW4A8Row4TileInto_zeroActivationRow(t *testing.T) {
	if !hasDotProd {
		t.Skip("no FEAT_DotProd on this core; the row4 kernels do not dispatch")
	}
	const group, K, N = 32, 256, 16
	rng := rand.New(rand.NewPCG(0xbee, 0x5))
	w := make([]float32, N*K)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	q4, q4s := QuantizeGroupsInt4(w, N, K, group)
	row4 := RepackW4A8Row4(q4, N, K, group)
	row4s := RepackW4A8Row4Scales(q4s, N, K, group)

	for _, zeroRow := range []int{2, 5} {
		const M = 7
		a := make([]float32, M*K)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		for k := range K {
			a[zeroRow*K+k] = 0
		}
		var wsWant, wsGot Workspace
		want := make([]float32, M*N)
		got := make([]float32, M*N)
		MatmulBTW4A8Into(&wsWant, a, q4, q4s, want, M, K, N, group)
		MatmulBTW4A8Row4TileInto(&wsGot, a, row4, row4s, got, M, K, N, group)
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("zeroRow=%d idx=%d (row %d): tile %v != canonical %v", zeroRow, i, i/N, got[i], want[i])
			}
		}
		for j := range N {
			if v := got[zeroRow*N+j]; v != 0 {
				t.Fatalf("zeroRow=%d col %d: got %v, want a literal 0", zeroRow, j, v)
			}
		}
	}
}

// TestMatmulBTW4A8Row4_rejectsTooSmallKAtEntry pins the shape error for K < group
// on both row4 entry points. K=0 satisfies the K%group==0 check that guards them,
// so before this it reached the span with nGroups=0 and died as an
// index-out-of-range on &blk[0] — a panic naming a kernel internal rather than
// the caller's mistake, which is the same distinction
// TestWrapInt4Row4_rejectsBadShapeAtWrap was written to hold elsewhere.
//
// The assertion is on the MESSAGE, not merely that a panic happens, because
// "it panics either way" is exactly the failure mode: what changed is WHICH
// panic, and only a message check can tell the two apart.
func TestMatmulBTW4A8Row4_rejectsTooSmallKAtEntry(t *testing.T) {
	if !hasDotProd {
		t.Skip("no FEAT_DotProd on this core; the row4 kernels do not dispatch")
	}
	const group, N = 32, 4
	for _, tc := range []struct {
		name string
		call func(ws *Workspace)
		want string
	}{
		{"tile_M4", func(ws *Workspace) {
			MatmulBTW4A8Row4TileInto(ws, nil, nil, nil, make([]float32, 4*N), 4, 0, N, group)
		}, "MatmulBTW4A8Row4TileInto requires K >= group=32"},
		{"row4_M1", func(ws *Workspace) {
			MatmulBTW4A8Row4Into(ws, nil, nil, nil, make([]float32, N), 1, 0, N, group)
		}, "MatmulBTW4A8Row4Into requires K >= group=32"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("K=0 did not panic at all")
				}
				msg, _ := r.(string)
				if msg == "" {
					if e, ok := r.(error); ok {
						msg = e.Error()
					}
				}
				if !strings.Contains(msg, tc.want) {
					t.Fatalf("panic names the wrong thing:\n got: %v\nwant it to contain: %s", r, tc.want)
				}
			}()
			var ws Workspace
			tc.call(&ws)
		})
	}
}
