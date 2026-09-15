//go:build amd64

package linalg

import (
	"math/rand/v2"
	"testing"
)

// w4a8TileRowsIndexHoisted — MEASURED NEGATIVE, not wired in.
//
// Tried: replacing w4a8TileRows's per-m dst[(i+m)*N+j] (a fresh multiply every m)
// with a hoisted i*N plus a running += N. Confirmed via
// -gcflags="-d=ssa/check_bce/debug=1" that this genuinely eliminates the dst
// write's bounds check (the highest-frequency one in the function — it fires
// once per output element, M*N times) that the original form defeats. Bit-
// identical to w4a8TileRows (TestW4A8TileRowsIndexHoisted_isBitIdentical) — pure
// index arithmetic, no reassociation of any accumulated float value.
//
// Performance: measured negative. AMD Ryzen 7 3700X, quiet box (load 0.00-0.02),
// benchstat n=5, order-alternated: qwen2.5-1.5b gate/up shape (K=1536,N=8960)
// +1.07% SLOWER (p=0.016, real, not noise); 7B FFN shape (K=1536,N=18944) and
// the ANN-scan shape (K=768,N=100000) both a statistical wash. The eliminated
// bounds check was real but apparently free at this call frequency — it sits
// behind dotW4A8Tile4RowAVX2's own K/32-deep SIMD work, which swamps a handful
// of scalar CMP+branch instructions in the 4-wide finishing loop. Removing it
// bought nothing and, on one shape, cost a touch (plausibly a scheduling
// side-effect of the extra idx/aIdx bookkeeping, not the bounds check itself).
//
// Kept here, not wired into w4a8TileRows, per this codebase's own convention:
// a documented negative is worth something, a silently-not-tried one is not.
func w4a8TileRowsIndexHoisted(aq []int8, aScales []float32, w4 []byte, wScales, dst []float32, M, K, N, group, nGroups, bpr, j0, j1 int) int {
	if !hasAVX2 || hasAVX512VNNIVL || M < 4 || group != 32 || K < 32 {
		return 0
	}
	mFull := M &^ 3
	nFull := K / 32
	done := nFull * 32
	var out [4]float32
	for j := j0; j < j1; j++ {
		prow := w4[j*bpr : j*bpr+bpr]
		srow := wScales[j*nGroups : j*nGroups+nGroups]
		for i := 0; i < mFull; i += 4 {
			dotW4A8Tile4RowAVX2(&aq[i*K], K, &prow[0], &srow[0], &out[0], nFull)
			idx := i*N + j
			aIdx := i
			for m := range 4 {
				aScale := aScales[aIdx]
				aIdx++
				if aScale == 0 {
					dst[idx] = 0
					idx += N
					continue
				}
				total := out[m]
				if done < K {
					arow := aq[(i+m)*K : (i+m)*K+K]
					var acc int32
					for k := done; k < K; k++ {
						b := prow[k>>1]
						nib := b & 0x0F
						if k&1 == 1 {
							nib = b >> 4
						}
						acc += int32(arow[k]) * int32(int(nib)-8)
					}
					total += float32(acc) * srow[nFull]
				}
				dst[idx] = total * aScale
				idx += N
			}
		}
	}
	return mFull
}

// w4a8TileRowsCase holds one random (M,K,N) case's inputs, built once so both
// kernels run on IDENTICAL data.
type w4a8TileRowsCase struct {
	aq                  []int8
	aScales             []float32
	w4                  []byte
	wScales             []float32
	M, K, N             int
	group, nGroups, bpr int
}

func newW4A8TileRowsCase(rng *rand.Rand, M, K, N int) w4a8TileRowsCase {
	const group = 32
	nGroups, bpr := groupsFor(K, group)
	aq := make([]int8, M*K)
	for i := range aq {
		aq[i] = int8(rng.IntN(255) - 127)
	}
	aScales := make([]float32, M)
	for i := range aScales {
		if i%7 == 0 {
			aScales[i] = 0 // exercise the aScale==0 early-out path too
		} else {
			aScales[i] = 0.01 + rng.Float32()*0.01
		}
	}
	w4 := make([]byte, N*bpr)
	for i := range w4 {
		w4[i] = byte(rng.UintN(256))
	}
	wScales := make([]float32, N*nGroups)
	for i := range wScales {
		wScales[i] = 0.01 + rng.Float32()*0.01
	}
	return w4a8TileRowsCase{aq, aScales, w4, wScales, M, K, N, group, nGroups, bpr}
}

func (c w4a8TileRowsCase) run(fn func(aq []int8, aScales []float32, w4 []byte, wScales, dst []float32, M, K, N, group, nGroups, bpr, j0, j1 int) int) []float32 {
	dst := make([]float32, c.M*c.N)
	fn(c.aq, c.aScales, c.w4, c.wScales, dst, c.M, c.K, c.N, c.group, c.nGroups, c.bpr, 0, c.N)
	return dst
}

func TestW4A8TileRowsIndexHoisted_isBitIdentical(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2 on this core; w4a8TileRows never dispatches")
	}
	rng := rand.New(rand.NewPCG(48, 8))
	// K values include both 32-aligned (no ragged tail) and ragged (K%32 != 0,
	// exercising the scalar mop-up branch) cases; N values are small and large.
	shapes := []struct{ M, K, N int }{
		{8, 32, 8}, {8, 64, 1},
		{8, 1000, 16},  // 1000%32 = 8, ragged
		{16, 4304, 64}, // so400m FFN width, aligned
		{8, 1537, 3},   // ragged, odd tail through the k&1 branch too
	}
	for _, sh := range shapes {
		c := newW4A8TileRowsCase(rng, sh.M, sh.K, sh.N)
		want := c.run(w4a8TileRows) // real production kernel
		got := c.run(w4a8TileRowsIndexHoisted)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("M=%d K=%d N=%d idx=%d: hoisted %v != production %v",
					sh.M, sh.K, sh.N, i, got[i], want[i])
			}
		}
	}
}

func benchW4A8TileRows(b *testing.B, fn func(aq []int8, aScales []float32, w4 []byte, wScales, dst []float32, M, K, N, group, nGroups, bpr, j0, j1 int) int, M, K, N int) {
	if !hasAVX2 {
		b.Skip("no AVX2 on this core; w4a8TileRows never dispatches")
	}
	const group = 32
	nGroups, bpr := groupsFor(K, group)
	rng := rand.New(rand.NewPCG(1, 2))
	aq := make([]int8, M*K)
	for i := range aq {
		aq[i] = int8(rng.IntN(255) - 127)
	}
	aScales := make([]float32, M)
	for i := range aScales {
		aScales[i] = 0.01
	}
	w4 := make([]byte, N*bpr)
	for i := range w4 {
		w4[i] = byte(rng.UintN(256))
	}
	wScales := make([]float32, N*nGroups)
	for i := range wScales {
		wScales[i] = 0.01
	}
	dst := make([]float32, M*N)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fn(aq, aScales, w4, wScales, dst, M, K, N, group, nGroups, bpr, 0, N)
	}
}

// Shapes mirror tile_ab_amd64_test.go's abShapes/abMs — the real cells this
// kernel's own S-01 acceptance test already sweeps.
func BenchmarkW4A8TileRows_production_qwen1_5B(b *testing.B) {
	benchW4A8TileRows(b, w4a8TileRows, 8, 1536, 8960)
}
func BenchmarkW4A8TileRows_indexHoisted_qwen1_5B(b *testing.B) {
	benchW4A8TileRows(b, w4a8TileRowsIndexHoisted, 8, 1536, 8960)
}

func BenchmarkW4A8TileRows_production_7bFFN(b *testing.B) {
	benchW4A8TileRows(b, w4a8TileRows, 8, 1536, 18944)
}
func BenchmarkW4A8TileRows_indexHoisted_7bFFN(b *testing.B) {
	benchW4A8TileRows(b, w4a8TileRowsIndexHoisted, 8, 1536, 18944)
}

func BenchmarkW4A8TileRows_production_annScan(b *testing.B) {
	benchW4A8TileRows(b, w4a8TileRows, 8, 768, 100000)
}
func BenchmarkW4A8TileRows_indexHoisted_annScan(b *testing.B) {
	benchW4A8TileRows(b, w4a8TileRowsIndexHoisted, 8, 768, 100000)
}
