//go:build arm64

package linalg

import (
	"fmt"
	"math/rand/v2"
	"os"
	"testing"
	"time"
)

// TestW4A8TilePanelTraffic is step 0.2 of the CPU-prefill-remainder brief: it
// sizes the S-01 tile at large M on both production shapes, and answers the one
// question that decides whether M-blocking (the brief's step 2) is worth
// building at all.
//
// THE DISCRIMINATOR IS PER-MAC COST AGAINST M, not throughput against M. The
// tile sweeps quads in the outer loop and activation rows in the inner one, so
// for each quad it walks the whole quantised activation panel. If that panel
// stops fitting in cache as M grows, it is re-streamed from L2 per quad and the
// per-MAC cost RISES with M — step 2 is live. If per-MAC cost is flat, the panel
// is either resident or the streaming is free, and step 2 is dead. Reported and
// reasoned about, never asserted: a threshold here would be a guess about one
// machine's cache hierarchy.
//
// K=8960 is the shape to read, because that is the one with the wide activation
// row (a panel row is K bytes after quantisation, so 8960 B/row against 1536).
func TestW4A8TilePanelTraffic(t *testing.T) {
	harnessOnly(t)
	if !hasDotProd {
		t.Skip("no FEAT_DotProd; the row4 tile does not dispatch")
	}
	const group = 32
	shapes := []struct {
		name string
		K, N int
	}{
		{"K1536_N8960_gate_up", 1536, 8960},
		{"K8960_N1536_down", 8960, 1536},
	}
	start := time.Now()
	fmt.Fprintf(os.Stderr, "\n[panel-probe] start %s — S-01 tile at large M, %s\n",
		start.Format("15:04:05"), "per-MAC cost is the step-2 discriminator")

	rng := rand.New(rand.NewPCG(0x9a, 0x7e))
	for _, sh := range shapes {
		K, N := sh.K, sh.N
		w := make([]float32, N*K)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		q4, q4s := QuantizeGroupsInt4(w, N, K, group)
		row4 := RepackW4A8Row4(q4, N, K, group)
		row4s := RepackW4A8Row4Scales(q4s, N, K, group)

		fmt.Fprintf(os.Stderr, "\n  %s  (weights %.1f MB)\n", sh.name, float64(len(row4)+4*len(row4s))/(1<<20))
		fmt.Fprintf(os.Stderr, "    %-5s %-8s %12s %12s %14s\n", "M", "workers", "ns/op", "GMAC/s", "ps per MAC")
		for _, workers := range []int{1, 6} {
			var basePerMAC float64
			for _, M := range []int{8, 64, 128, 512} {
				a := make([]float32, M*K)
				for i := range a {
					a[i] = float32(rng.NormFloat64())
				}
				dst := make([]float32, M*N)
				ws := &Workspace{}
				ws.SetWorkers(workers)
				if workers == 1 {
					ws.SetThreshold(1 << 62)
				} else {
					ws.SetThreshold(0)
				}
				MatmulBTW4A8Row4TileInto(ws, a, row4, row4s, dst, M, K, N, group)
				best := 0.0
				for rep := 0; rep < 3; rep++ {
					r := testing.Benchmark(func(b *testing.B) {
						for b.Loop() {
							MatmulBTW4A8Row4TileInto(ws, a, row4, row4s, dst, M, K, N, group)
						}
					})
					ns := float64(r.NsPerOp())
					if best == 0 || ns < best {
						best = ns
					}
				}
				macs := float64(M) * float64(N) * float64(K)
				perMAC := best / macs * 1000 // picoseconds per MAC
				if M == 8 {
					basePerMAC = perMAC
				}
				panelMB := float64(M*K) / (1 << 20) // the int8 activation panel
				fmt.Fprintf(os.Stderr, "    %-5d %-8d %12.0f %12.1f %14.2f   (panel %.2f MB, %.3fx vs M=8)\n",
					M, workers, best, macs/best, perMAC, panelMB, perMAC/basePerMAC)
				t.Logf("%s workers=%d M=%d: %.0f ns, %.1f GMAC/s, %.2f ps/MAC, panel %.2f MB (%.3fx vs M=8)",
					sh.name, workers, M, best, macs/best, perMAC, panelMB, perMAC/basePerMAC)
			}
		}
	}
}
