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

// TestW8A8TileVsSpanAB is docs/task-simd-audit.md S-01b's decision-rule harness:
// the pre-tile W8A8 span (w8a8SpanRows over the whole rectangle — literally the
// code that shipped before dotI8Tile4x4 existed) against w8a8Span, which now
// routes the largest 4x4 rectangle through the tile. Both arms run serially on
// one core, so this measures the kernel and not the fan-out.
//
// THE SHAPE GRID IS THE POINT, not an afterthought. w8a8Span's own comment
// records why: the deleted dotI8Cols8 was measured at ONE shape where B was
// cache-resident, shipped on that, and lost badly wherever B streamed. The tile
// walks four weight streams too — inside one contiguous 4*K block rather than
// eight scattered ones, and with four times the arithmetic per weight byte — but
// that is an argument, and the argument is exactly what was wrong last time. So
// the grid deliberately straddles the LLC: two cache-resident shapes and three
// that stream, tagged in the output, and the S-01b verdict is not allowed to rest
// on the resident ones.
func TestW8A8TileVsSpanAB(t *testing.T) {
	harnessOnly(t)
	if !cpufeat.HasDotProd {
		t.Skip("no FEAT_DotProd on this core; the tile does not dispatch")
	}
	shapes := []struct {
		K, N int
		note string
	}{
		{1536, 8960, "resident — Qwen2.5-Coder-1.5B gate/up, the prefill cell"},
		{768, 8192, "resident — the shape dotI8Cols8 was shipped on"},
		{1536, 18944, "streamed — the worst case dotI8Cols8 saw"},
		{3584, 18944, "streamed — a 7B model's FFN"},
		{768, 100000, "streamed — aikit's own ANN scan, B far past the LLC"},
	}
	Ms := []int{4, 8, 16}

	start := time.Now()
	fmt.Fprintf(os.Stderr, "[w8a8-tile-ab] start %s — %d shapes x %d Ms x 2 arms x 3 passes, serial\n",
		start.Format("15:04:05"), len(shapes), len(Ms))

	type key struct{ s, M int }
	ratios := map[key][]float64{}
	gm := map[key][2]float64{}

	for pass := 1; pass <= 3; pass++ {
		for si, sh := range shapes {
			K, N := sh.K, sh.N
			rng := rand.New(rand.NewPCG(uint64(0xa11+si), 0x8a8))
			bq := make([]int8, K*N)
			for i := range bq {
				bq[i] = int8(rng.IntN(255) - 127)
			}
			bs := make([]float32, N)
			for i := range bs {
				bs[i] = 0.01
			}
			for _, M := range Ms {
				aq := make([]int8, M*K)
				for i := range aq {
					aq[i] = int8(rng.IntN(255) - 127)
				}
				as := make([]float32, M)
				for i := range as {
					as[i] = 0.01
				}
				base := make([]float32, M*N)
				tiled := make([]float32, M*N)

				w8a8SpanRows(aq, as, bq, bs, base, K, N, 0, M, 0, N)
				w8a8Span(aq, as, bq, bs, tiled, M, K, N, 0, N)
				for i := range base {
					if base[i] != tiled[i] {
						t.Fatalf("K=%d N=%d M=%d idx=%d: tile %v != pre-tile span %v — "+
							"the A/B arms disagree, the timing would be meaningless",
							K, N, M, i, tiled[i], base[i])
					}
				}

				bns := float64(testing.Benchmark(func(b *testing.B) {
					for b.Loop() {
						w8a8SpanRows(aq, as, bq, bs, base, K, N, 0, M, 0, N)
					}
				}).NsPerOp())
				tns := float64(testing.Benchmark(func(b *testing.B) {
					for b.Loop() {
						w8a8Span(aq, as, bq, bs, tiled, M, K, N, 0, N)
					}
				}).NsPerOp())
				k := key{si, M}
				ratios[k] = append(ratios[k], bns/tns)
				macs := float64(M) * float64(N) * float64(K)
				gm[k] = [2]float64{macs / bns, macs / tns}
				fmt.Fprintf(os.Stderr, "[w8a8-tile-ab] pass %d/3 K%-5d N%-6d M%-3d  pre-tile %6.1f GMAC/s  tile %6.1f  %.2fx  (+%v)\n",
					pass, K, N, M, macs/bns, macs/tns, bns/tns, time.Since(start).Round(time.Second))
			}
			bq = nil
		}
	}

	med3 := func(v []float64) float64 {
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
	t.Logf("S-01b W8A8 tile vs pre-tile span, serial, median of 3 interleaved passes")
	t.Logf("%-6s %-7s %-4s %11s %11s %8s   %s", "K", "N", "M", "pre GMAC/s", "tile GMAC/s", "ratio", "regime")
	for si, sh := range shapes {
		for _, M := range Ms {
			k := key{si, M}
			g := gm[k]
			t.Logf("%-6d %-7d %-4d %11.1f %11.1f %7.2fx   %s", sh.K, sh.N, M, g[0], g[1], med3(ratios[k]), sh.note)
		}
	}
}
