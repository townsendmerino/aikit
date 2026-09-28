//go:build arm64

package linalg

import (
	"fmt"
	"math/rand"
	"sort"
	"testing"
	"time"
)

// BenchmarkW4A8Row4F16Fix is the L1 arm64 fix's pre-registered A-or-B instrument (goinfer
// docs/tasks/task-cpu-decode-peer-gap-2026-09.md, "L1 arm64 fix — pre-registered"): the M=1 row4 decode
// matmul over a cold, DRAM-streamed bank, through the parallel workspace, on four model shapes. Four arms see
// identical weights:
//   - f32: the f32-scale kernel (the pre-v1.50.0 decode);
//   - scalar: binary16 scales widened per quad by the scalar f16ToF32 loop (v1.50.0's arm64 path, reproduced
//     here, since widenF16 is now NEON);
//   - neon: binary16 scales widened per quad by the NEON widenF16, then the f32 kernel (candidate A);
//   - fused: dotW4A8SplitHalf4RowFoldF16, widening in-kernel (candidate B).
//
// Run with -count 8. -count repeats each arm back-to-back (it does not round-robin), so
// BenchmarkW4A8Row4F16FixInterleaved below is the drift check.
func BenchmarkW4A8Row4F16Fix(b *testing.B) {
	if !hasDotProd || !row4Usable() {
		b.Skip("DotProd / row4 required")
	}
	shapes := []struct {
		name string
		K, N int
		bank int
	}{
		{"0.5B-gateup-896x4864", 896, 4864, 40},
		{"1.5B-gateup-1536x8960", 1536, 8960, 12},
		{"1.5B-down-8960x1536", 8960, 1536, 12},
		{"7B-gateup-3584x18944", 3584, 18944, 3},
	}
	for _, sh := range shapes {
		bk := f16FixBank(sh.K, sh.N, sh.bank)
		a := make([]float32, sh.K)
		rng := rand.New(rand.NewSource(7))
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		dst := make([]float32, sh.N)
		var ws Workspace
		bytesPer := float64(sh.K*sh.N/2) + float64(sh.K/32*sh.N)*2
		run := func(arm string, f func(m *f16FixMat)) {
			b.Run(fmt.Sprintf("%s/%s", sh.name, arm), func(b *testing.B) {
				b.ResetTimer()
				for i := range b.N {
					f(&bk[i%len(bk)])
				}
				sinkW4A8F32ARM64 = dst[0]
				b.ReportMetric(bytesPer*float64(b.N)/b.Elapsed().Seconds()/1e9, "GB/s")
			})
		}
		run("f32", func(m *f16FixMat) { MatmulBTW4A8Row4Into(&ws, a, m.row4, m.s32, dst, 1, sh.K, sh.N, 32) })
		run("scalar", func(m *f16FixMat) { matmulRow4F16ScalarWiden(&ws, a, m.row4, m.s16, dst, sh.K, sh.N) })
		run("neon", func(m *f16FixMat) {
			w4a8Row4FusedF16 = false
			matmulBTW4A8Row4F16Into(&ws, a, m.row4, m.s16, dst, sh.K, sh.N, 32)
			w4a8Row4FusedF16 = true
		})
		run("fused", func(m *f16FixMat) { matmulBTW4A8Row4F16Into(&ws, a, m.row4, m.s16, dst, sh.K, sh.N, 32) })
	}
}

type f16FixMat struct {
	row4 []byte
	s16  []uint16  // interleaved binary16 scales
	s32  []float32 // the same, widened
}

var f16FixBanks = map[[2]int][]f16FixMat{}

// f16FixBank builds (once per shape, shared by every -count round) a bank of distinct matrices sized past the
// LLC, so each matmul streams its weights from DRAM as decode does.
func f16FixBank(K, N, n int) []f16FixMat {
	if bk, ok := f16FixBanks[[2]int{K, N}]; ok {
		return bk
	}
	rng := rand.New(rand.NewSource(int64(K*31 + N)))
	bk := make([]f16FixMat, n)
	for i := range bk {
		w := make([]float32, N*K)
		for j := range w {
			w[j] = float32(rng.NormFloat64()) * 0.02
		}
		p, s := QuantizeGroupsInt4(w, N, K, 32)
		s16 := RepackW4A8Row4ScalesF16(F32ToF16Scales(s), N, K, 32)
		s32 := make([]float32, len(s16))
		for j, h := range s16 {
			s32[j] = f16ToF32(h)
		}
		bk[i] = f16FixMat{row4: RepackW4A8Row4(p, N, K, 32), s16: s16, s32: s32}
	}
	f16FixBanks[[2]int{K, N}] = bk
	return bk
}

// matmulRow4F16ScalarWiden is matmulBTW4A8Row4F16Into's folded M=1 path as aikit v1.50.0 ran it on arm64:
// each quad's scales widened by the scalar f16ToF32 into a buffer, then the f32 kernel.
func matmulRow4F16ScalarWiden(ws *Workspace, a []float32, w4Row4 []byte, s4 []uint16, dst []float32, K, N int) {
	nGroups, bpr := groupsFor(K, 32)
	aq := ws.int8Buf(K)
	aScale := quantizeRowInt8(a[:K], aq)
	corr := ws.int32Buf(4 * nGroups)
	w4a8LaneCorrNeg8(&aq[0], &corr[0], nGroups)
	ws.parallel(N/4, func(q0, q1 int) {
		sblk := make([]float32, 4*nGroups)
		var out [4]float32
		for q := q0; q < q1; q++ {
			blk := w4Row4[q*4*bpr : q*4*bpr+4*bpr]
			for i, h := range s4[q*4*nGroups : q*4*nGroups+4*nGroups] {
				sblk[i] = f16ToF32(h)
			}
			dotW4A8SplitHalf4RowFold(&aq[0], &corr[0], &blk[0], &sblk[0], &out[0], nGroups)
			dst[q*4] = out[0] * aScale
			dst[q*4+1] = out[1] * aScale
			dst[q*4+2] = out[2] * aScale
			dst[q*4+3] = out[3] * aScale
		}
	})
}

// BenchmarkW4A8Row4F16FixInterleaved is BenchmarkW4A8Row4F16Fix's drift check. -count repeats each arm
// back-to-back, so a thermal or background drift between arms would land on their ratio. This one runs 8 rounds,
// each timing every arm of a shape in an order rotated by one per round, and logs each arm's median. Run it once:
// -run '^$' -bench W4A8Row4F16FixInterleaved -benchtime 1x.
func BenchmarkW4A8Row4F16FixInterleaved(b *testing.B) {
	if !hasDotProd || !row4Usable() {
		b.Skip("DotProd / row4 required")
	}
	shapes := []struct {
		name       string
		K, N, bank int
		iters      int
	}{
		{"0.5B-gateup-896x4864", 896, 4864, 40, 2000},
		{"1.5B-gateup-1536x8960", 1536, 8960, 12, 600},
		{"1.5B-down-8960x1536", 8960, 1536, 12, 600},
		{"7B-gateup-3584x18944", 3584, 18944, 3, 300},
	}
	arms := []string{"f32", "scalar", "neon", "fused"}
	for _, sh := range shapes {
		bk := f16FixBank(sh.K, sh.N, sh.bank)
		a := make([]float32, sh.K)
		rng := rand.New(rand.NewSource(7))
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		dst := make([]float32, sh.N)
		var ws Workspace
		call := func(arm string, m *f16FixMat) {
			switch arm {
			case "f32":
				MatmulBTW4A8Row4Into(&ws, a, m.row4, m.s32, dst, 1, sh.K, sh.N, 32)
			case "scalar":
				matmulRow4F16ScalarWiden(&ws, a, m.row4, m.s16, dst, sh.K, sh.N)
			case "neon":
				w4a8Row4FusedF16 = false
				matmulBTW4A8Row4F16Into(&ws, a, m.row4, m.s16, dst, sh.K, sh.N, 32)
				w4a8Row4FusedF16 = true
			case "fused":
				matmulBTW4A8Row4F16Into(&ws, a, m.row4, m.s16, dst, sh.K, sh.N, 32)
			}
		}
		ns := map[string][]float64{}
		for round := 0; round < 8; round++ {
			for k := range arms {
				arm := arms[(k+round)%len(arms)]
				for i := 0; i < sh.iters/10; i++ { // warm this arm's code path
					call(arm, &bk[i%len(bk)])
				}
				t0 := time.Now()
				for i := 0; i < sh.iters; i++ {
					call(arm, &bk[i%len(bk)])
				}
				ns[arm] = append(ns[arm], float64(time.Since(t0).Nanoseconds())/float64(sh.iters))
			}
		}
		med := func(v []float64) float64 {
			s := append([]float64(nil), v...)
			sort.Float64s(s)
			return (s[3] + s[4]) / 2
		}
		m := map[string]float64{}
		for _, arm := range arms {
			m[arm] = med(ns[arm])
		}
		b.Logf("%-24s f32 %8.0f  scalar %8.0f  neon %8.0f  fused %8.0f ns | scalar/f32 %.3f  neon/f32 %.3f  fused/f32 %.3f  fused/neon %.3f",
			sh.name, m["f32"], m["scalar"], m["neon"], m["fused"], m["scalar"]/m["f32"], m["neon"]/m["f32"], m["fused"]/m["f32"], m["fused"]/m["neon"])
	}
	sinkW4A8F32ARM64 = 0
}
