package linalg

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

// BenchmarkActGroupSplitHalf: the M=1 split-half W4A8 decode matmul, per-row activation scale vs
// per-32 (SetActQuantGroup(32), the combined-scale path), at goinfer decode projection shapes:
// qwen2.5-coder-1.5b (hidden 1536, FFN 8960) and qwen2.5-7b (hidden 3584, FFN 18944), K×N.
// goinfer docs/tasks/task-actquant-pergroup-2026-09.md's pre-registered kernel gate compares
// the two sub-benchmarks' times per shape: <= 1.05 ship, (1.05, 1.10] optimize, > 1.10 fail.
// Run on an idle box with -count >= 10 and compare paired medians.
func BenchmarkActGroupSplitHalf(b *testing.B) {
	if !splitHalfUsable() {
		b.Skip("split-half AVX2 kernel not usable on this CPU")
	}
	for _, sh := range [][2]int{{1536, 1536}, {1536, 8960}, {8960, 1536}, {3584, 3584}, {3584, 18944}, {18944, 3584}} {
		K, N := sh[0], sh[1]
		r := rand.New(rand.NewPCG(uint64(K), uint64(N)))
		a := agRandMat(r, K)
		q4, s4 := QuantizeGroupsInt4(agRandMat(r, N*K), N, K, 32)
		wm, ok := RepackInt4SplitHalfInPlace(q4, s4, N, K, 32)
		if !ok {
			b.Skip("split-half repack declined")
		}
		dst := make([]float32, N)
		for _, g := range []int{0, 32} {
			b.Run(fmt.Sprintf("K%d_N%d/group%d", K, N, g), func(b *testing.B) {
				prev := ActQuantGroup()
				SetActQuantGroup(g)
				defer SetActQuantGroup(prev)
				var ws Workspace
				ws.SetThreshold(1 << 20) // goinfer's int4 decode threshold (int4ParThreshold)
				b.SetBytes(int64(N * K / 2))
				b.ResetTimer()
				for range b.N {
					wm.MatmulBTW4A8Into(&ws, a, dst, 1)
				}
			})
		}
	}
}
