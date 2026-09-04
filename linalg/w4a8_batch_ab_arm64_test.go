//go:build arm64

package linalg

import (
	"fmt"
	"math/rand/v2"
	"os"
	"testing"
	"time"
)

// TestW4A8BatchVsPerOpAB is MatmulBTW4A8Batch's decision-rule harness. The
// do-nothing arm is deliberately NOT MatmulBTW4A8Into: goinfer's decode reaches
// these projections through WeightMat.MatmulBTW4A8Into, which at M=1 dispatches
// the row4 kernel, so comparing against the canonical kernel would credit the
// batch with a layout win it did not earn. The arm here is N separate
// MatmulBTW4A8Row4Into calls — exactly what ships today.
//
// Shapes are the two real fusions on the 1.5B: q‖k‖v (12 q heads and 2 kv heads
// at head dim 128) and gate‖up.
// medOf3 is a local median-of-three; the amd64 A/B file has its own med3 behind
// a build tag, and a shared one would have to move to a portable file for no
// other reason.
func medOf3(v []float64) float64 {
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

func TestW4A8BatchVsPerOpAB(t *testing.T) {
	harnessOnly(t)
	if !hasDotProd {
		t.Skip("DotProd required")
	}
	const (
		group = 32
		K     = 1536
		M     = 1
	)
	fusions := []struct {
		name string
		Ns   []int
	}{
		{"qkv", []int{1536, 256, 256}},
		{"gate_up", []int{8960, 8960}},
	}
	rng := rand.New(rand.NewPCG(0xba7, 0xab))

	start := time.Now()
	fmt.Fprintf(os.Stderr, "\n[w4a8-batch-ab] start %s — K=%d M=%d, arms interleaved, median of 3\n",
		start.Format("15:04:05"), K, M)

	for _, workers := range []int{6, 8} {
		for _, f := range fusions {
			a := make([]float32, M*K)
			for i := range a {
				a[i] = float32(rng.NormFloat64())
			}
			ops := make([]W4A8Op, len(f.Ns))
			for i, N := range f.Ns {
				w := make([]float32, N*K)
				for j := range w {
					w[j] = float32(rng.NormFloat64())
				}
				q4, q4s := QuantizeGroupsInt4(w, N, K, group)
				wm := WrapInt4(q4, q4s, N, K, group)
				if !wm.RepackInt4Row4() {
					t.Fatalf("N=%d did not repack to row4 — the arm would not be representative", N)
				}
				ops[i] = W4A8Op{
					W4: q4, Scales: q4s,
					Row4: wm.q4Row4, Row4Scales: wm.q4Row4Scales,
					Dst: make([]float32, M*N), N: N,
				}
			}
			perOpDst := make([][]float32, len(ops))
			for i, op := range ops {
				perOpDst[i] = make([]float32, M*op.N)
			}

			newWS := func() *Workspace {
				ws := &Workspace{}
				ws.SetWorkers(workers)
				ws.SetThreshold(0)
				return ws
			}
			wsA, wsB := newWS(), newWS()

			perOp := func() {
				for i, op := range ops {
					MatmulBTW4A8Row4Into(wsA, a, op.Row4, op.Row4Scales, perOpDst[i], M, K, op.N, group)
				}
			}
			batch := func() { MatmulBTW4A8Batch(wsB, a, M, K, group, ops) }

			perOp()
			batch()
			for i, op := range ops {
				for k := range op.Dst {
					if op.Dst[k] != perOpDst[i][k] {
						t.Fatalf("%s op %d idx %d: batch %v != per-op row4 %v — arms disagree, timing meaningless",
							f.name, i, k, op.Dst[k], perOpDst[i][k])
					}
				}
			}

			var rp, rb []float64
			for range 3 {
				rp = append(rp, float64(testing.Benchmark(func(b *testing.B) {
					for b.Loop() {
						perOp()
					}
				}).NsPerOp()))
				rb = append(rb, float64(testing.Benchmark(func(b *testing.B) {
					for b.Loop() {
						batch()
					}
				}).NsPerOp()))
			}
			p, bb := medOf3(rp), medOf3(rb)
			fmt.Fprintf(os.Stderr, "  workers=%d %-8s per-op %8.0f ns   batch %8.0f ns   %.3fx   (%d ops)\n",
				workers, f.name, p, bb, p/bb, len(ops))
			t.Logf("workers=%d %s: per-op(row4) %.0f ns, batch %.0f ns, %.3fx", workers, f.name, p, bb, p/bb)
		}
	}
}
