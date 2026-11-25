//go:build arm64

package linalg

import (
	"fmt"
	"math/rand/v2"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/internal/cpufeat"
)

// TestW4A8TileVsCanonicalAB is docs/task-simd-audit.md S-01's decision-rule
// harness: the canonical M>1 path (MatmulBTW4A8Into — a GEMV per activation
// row per weight row) against the register-blocked tile
// (matmulBTW4A8Row4TileInto) at the production projection shape, over the batch
// sizes prefill and speculative verify actually run at.
//
// PAIRED AND INTERLEAVED, three passes, per the campaign rules: the two arms are
// timed back to back inside each pass and the passes alternate, so a drift in
// machine state during the run shows up as spread between passes rather than as
// a fake win for whichever arm ran first. The reported ratio is the median pass.
//
// Serial by default (SetThreshold at max), because this measures THE KERNEL. The
// fan-out is S-02's subject and has its own finding; mixing the two is how a
// kernel win gets credited to a scheduler or vice versa. Set W4A8_TILE_AB_PAR=1
// to run the same grid on the default parallel dispatch instead.
//
// Not a gate — it asserts nothing about speed, only that the two arms agree
// bit-for-bit at every cell, which they must (see
// TestMatmulBTW4A8Row4TileInto_bitIdenticalToCanonical). Read the numbers, do
// not let CI vote on them.
func TestW4A8TileVsCanonicalAB(t *testing.T) {
	harnessOnly(t)
	if !cpufeat.HasDotProd {
		t.Skip("no FEAT_DotProd on this core; the row4 kernels do not dispatch")
	}
	const (
		group = 32
		K     = 1536 // Qwen2.5-Coder-1.5B gate/up projection
		N     = 8960
	)
	parallel := os.Getenv("W4A8_TILE_AB_PAR") == "1"
	Ms := []int{1, 2, 4, 8, 16, 64}

	start := time.Now()
	fmt.Fprintf(os.Stderr, "[tile-ab] start %s — K=%d N=%d, Ms=%v, parallel=%v, 3 interleaved passes\n",
		start.Format("15:04:05"), K, N, Ms, parallel)

	rng := rand.New(rand.NewPCG(0x5117, 0xace))
	w := make([]float32, N*K)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	q4, q4s := QuantizeGroupsInt4(w, N, K, group)
	row4 := RepackW4A8Row4(q4, N, K, group)
	row4s := RepackW4A8Row4Scales(q4s, N, K, group)
	fmt.Fprintf(os.Stderr, "[tile-ab] weights ready (%.1f MB packed + %.1f MB scales) at +%v\n",
		float64(len(row4))/(1<<20), float64(len(row4s)*4)/(1<<20), time.Since(start).Round(time.Millisecond))

	newWS := func() *Workspace {
		ws := &Workspace{}
		if !parallel {
			ws.SetThreshold(1 << 62)
		}
		return ws
	}

	type cell struct {
		M           int
		canon, tile float64
		ratio       float64
	}
	results := make(map[int][]cell)

	for pass := 1; pass <= 3; pass++ {
		for _, M := range Ms {
			a := make([]float32, M*K)
			for i := range a {
				a[i] = float32(rng.NormFloat64())
			}
			dstA := make([]float32, M*N)
			dstB := make([]float32, M*N)
			wsA, wsB := newWS(), newWS()

			// Warm both workspaces' scratch so neither arm pays a first-call
			// allocation inside the timed region.
			MatmulBTW4A8Into(wsA, a, q4, q4s, dstA, M, K, N, group)
			matmulBTW4A8Row4TileInto(wsB, a, row4, row4s, dstB, M, K, N, group)
			for i := range dstA {
				if dstA[i] != dstB[i] {
					t.Fatalf("M=%d idx=%d: tile %v != canonical %v — the A/B arms disagree, "+
						"the timing below would be meaningless", M, i, dstB[i], dstA[i])
				}
			}

			canon := float64(testing.Benchmark(func(b *testing.B) {
				for b.Loop() {
					MatmulBTW4A8Into(wsA, a, q4, q4s, dstA, M, K, N, group)
				}
			}).NsPerOp())
			tile := float64(testing.Benchmark(func(b *testing.B) {
				for b.Loop() {
					matmulBTW4A8Row4TileInto(wsB, a, row4, row4s, dstB, M, K, N, group)
				}
			}).NsPerOp())
			results[M] = append(results[M], cell{M: M, canon: canon, tile: tile, ratio: canon / tile})
			fmt.Fprintf(os.Stderr, "[tile-ab] pass %d/3 M=%-3d canonical %10.0f ns  tile %10.0f ns  %.2fx  (+%v)\n",
				pass, M, canon, tile, canon/tile, time.Since(start).Round(time.Second))
		}
	}

	macs := func(M int) float64 { return float64(M) * float64(N) * float64(K) }
	t.Logf("S-01 tile vs canonical, K=%d N=%d, parallel=%v, median of 3 interleaved passes", K, N, parallel)
	t.Logf("%-5s %14s %14s %8s %10s %10s", "M", "canonical ns", "tile ns", "ratio", "canon GM/s", "tile GM/s")
	for _, M := range Ms {
		cs := results[M]
		med := func(pick func(cell) float64) float64 {
			v := []float64{pick(cs[0]), pick(cs[1]), pick(cs[2])}
			if v[0] > v[1] {
				v[0], v[1] = v[1], v[0]
			}
			if v[1] > v[2] {
				v[1], v[2] = v[2], v[1]
			}
			if v[0] > v[1] {
				v[0], v[1] = v[1], v[0]
			}
			return v[1]
		}
		c, tl := med(func(x cell) float64 { return x.canon }), med(func(x cell) float64 { return x.tile })
		t.Logf("%-5d %14.0f %14.0f %7.2fx %10.1f %10.1f",
			M, c, tl, c/tl, macs(M)/c, macs(M)/tl)
	}
}
