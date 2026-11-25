//go:build amd64

package linalg

import (
	"fmt"
	"math/rand/v2"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/internal/cpufeat"
)

// abShapes is the grid both amd64 tile harnesses sweep. It straddles the 3700X's
// 32 MB L3 deliberately: a kernel that blocks over multiple weight rows can be a
// win while B is cache-resident and a loss once B streams, which is exactly how a
// deleted eight-column W8A8 kernel got shipped and then reverted (see w8a8Span's
// comment). One resident shape is not a measurement.
var abShapes = []struct {
	K, N int
	note string
}{
	{1536, 8960, "Qwen2.5-Coder-1.5B gate/up — the prefill cell"},
	{768, 8192, "small K, resident"},
	{1536, 18944, "a 7B model's FFN width"},
	{3584, 18944, "large K, streamed"},
	{768, 100000, "aikit's ANN scan, B far past the L3"},
}

var abMs = []int{4, 8, 16}

func abRandInt8(rng *rand.Rand, n int) []int8 {
	v := make([]int8, n)
	for i := range v {
		v[i] = int8(rng.IntN(255) - 127)
	}
	return v
}

func med3(v []float64) float64 {
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

// TestW4A8TileVsSpanAB_amd64 is docs/task-simd-audit.md S-01's amd64 decision
// rule: the pre-tile canonical span (w4a8SpanRows over every row — the code that
// shipped before dotW4A8Tile4RowAVX2) against w4a8Span, which now hands the first
// M&^3 activation rows to the tile. Serial, one core: this measures the kernel,
// not the fan-out.
//
// Pre-registered expectation, written before the first run: the tile costs 13
// instructions per 32 MACs per row against the span's 25, and the VPMADDWD count
// per MAC is unchanged, so the Zen 2 single-port floor (~67 GMAC/s at 4.2 GHz) is
// far away and does not bind. Roughly 2x is the number to beat; below 1.2x the
// finding is parked.
func TestW4A8TileVsSpanAB_amd64(t *testing.T) {
	harnessOnly(t)
	if !cpufeat.HasAVX2 {
		t.Skip("no AVX2 on this core; the W4A8 tile does not dispatch")
	}
	const group = 32
	start := time.Now()
	fmt.Fprintf(os.Stderr, "[w4a8-amd64-ab] start %s — %d shapes x %d Ms x 2 arms x 3 passes\n",
		start.Format("15:04:05"), len(abShapes), len(abMs))

	type key struct{ s, M int }
	ratios := map[key][]float64{}
	gm := map[key][2]float64{}

	for si, sh := range abShapes {
		K, N := sh.K, sh.N
		nGroups, bpr := groupsFor(K, group)
		rng := rand.New(rand.NewPCG(uint64(0x4a8+si), 0xab))
		w4 := make([]byte, N*bpr)
		for i := range w4 {
			w4[i] = byte(rng.UintN(256))
		}
		ws := make([]float32, N*nGroups)
		for i := range ws {
			ws[i] = 0.01 + rng.Float32()*0.01
		}
		for _, M := range abMs {
			aq := abRandInt8(rng, M*K)
			as := make([]float32, M)
			for i := range as {
				as[i] = 0.01
			}
			base := make([]float32, M*N)
			tiled := make([]float32, M*N)
			w4a8SpanRows(aq, as, w4, ws, base, K, N, group, nGroups, bpr, 0, M, 0, N)
			w4a8Span(aq, as, w4, ws, tiled, M, K, N, group, nGroups, bpr, 0, N)
			for i := range base {
				if base[i] != tiled[i] {
					t.Fatalf("K=%d N=%d M=%d idx=%d: tile %v != pre-tile span %v — arms disagree, timing meaningless",
						K, N, M, i, tiled[i], base[i])
				}
			}
			for pass := range 3 {
				bns := float64(testing.Benchmark(func(b *testing.B) {
					for b.Loop() {
						w4a8SpanRows(aq, as, w4, ws, base, K, N, group, nGroups, bpr, 0, M, 0, N)
					}
				}).NsPerOp())
				tns := float64(testing.Benchmark(func(b *testing.B) {
					for b.Loop() {
						w4a8Span(aq, as, w4, ws, tiled, M, K, N, group, nGroups, bpr, 0, N)
					}
				}).NsPerOp())
				k := key{si, M}
				ratios[k] = append(ratios[k], bns/tns)
				macs := float64(M) * float64(N) * float64(K)
				gm[k] = [2]float64{macs / bns, macs / tns}
				fmt.Fprintf(os.Stderr, "[w4a8-amd64-ab] pass %d/3 K%-5d N%-6d M%-3d  pre %5.1f GMAC/s  tile %5.1f  %.2fx  (+%v)\n",
					pass+1, K, N, M, macs/bns, macs/tns, bns/tns, time.Since(start).Round(time.Second))
			}
		}
	}
	t.Logf("S-01 amd64: W4A8 tile vs pre-tile span, serial, median of 3 interleaved passes")
	t.Logf("%-6s %-7s %-4s %11s %11s %8s   %s", "K", "N", "M", "pre GMAC/s", "tile GMAC/s", "ratio", "note")
	for si, sh := range abShapes {
		for _, M := range abMs {
			k := key{si, M}
			g := gm[k]
			t.Logf("%-6d %-7d %-4d %11.1f %11.1f %7.2fx   %s", sh.K, sh.N, M, g[0], g[1], med3(ratios[k]), sh.note)
		}
	}
}

// TestW8A8TileVsSpanAB_amd64 is S-01b's amd64 decision rule, same construction.
//
// Pre-registered expectation, again written before the first run and deliberately
// modest: the tile cuts instructions per MAC from 0.25 to 0.172, but VPMADDWD
// stays at one per 16 MACs and issues on a SINGLE Zen 2 port, so ~67 GMAC/s is a
// hard ceiling. dotI8AVX2 already sits at 48.3-51.3, i.e. 72-76% of it — so the
// most this can return is about 1.35x, and the audit's "2-4x" for S-01b is an
// arm64 number that does not transfer. If the measurement lands far ABOVE 1.35x,
// the port model is wrong and that is the finding.
func TestW8A8TileVsSpanAB_amd64(t *testing.T) {
	harnessOnly(t)
	if !cpufeat.HasAVX2 {
		t.Skip("no AVX2 on this core; the W8A8 tile does not dispatch")
	}
	start := time.Now()
	fmt.Fprintf(os.Stderr, "[w8a8-amd64-ab] start %s\n", start.Format("15:04:05"))

	type key struct{ s, M int }
	ratios := map[key][]float64{}
	gm := map[key][2]float64{}

	for si, sh := range abShapes {
		K, N := sh.K, sh.N
		rng := rand.New(rand.NewPCG(uint64(0x8a+si), 0xcd))
		bq := abRandInt8(rng, K*N)
		bs := make([]float32, N)
		for i := range bs {
			bs[i] = 0.01
		}
		for _, M := range abMs {
			aq := abRandInt8(rng, M*K)
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
					t.Fatalf("K=%d N=%d M=%d idx=%d: tile %v != pre-tile span %v — arms disagree",
						K, N, M, i, tiled[i], base[i])
				}
			}
			for pass := range 3 {
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
				fmt.Fprintf(os.Stderr, "[w8a8-amd64-ab] pass %d/3 K%-5d N%-6d M%-3d  pre %5.1f GMAC/s  tile %5.1f  %.2fx  (+%v)\n",
					pass+1, K, N, M, macs/bns, macs/tns, bns/tns, time.Since(start).Round(time.Second))
			}
		}
	}
	t.Logf("S-01b amd64: W8A8 tile vs pre-tile span, serial, median of 3 interleaved passes")
	t.Logf("%-6s %-7s %-4s %11s %11s %8s   %s", "K", "N", "M", "pre GMAC/s", "tile GMAC/s", "ratio", "note")
	for si, sh := range abShapes {
		for _, M := range abMs {
			k := key{si, M}
			g := gm[k]
			t.Logf("%-6d %-7d %-4d %11.1f %11.1f %7.2fx   %s", sh.K, sh.N, M, g[0], g[1], med3(ratios[k]), sh.note)
		}
	}
}
