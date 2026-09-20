package linalg

import "testing"

// BenchmarkMatmulQKAVGroupAB is R13's (docs/tasks/red-october.md, goinfer) "aikit's kernel A/B
// bench first (per-head as the do-nothing arm, A's two-head shape, B's full group) — it
// nominates." The do-nothing arm here is G separate MatmulQKAcc64/MatmulAVAcc64 calls (today's
// shipped shape); "full group" is one MatmulQKAcc64Group/MatmulAVAcc64Group call. Pure Go, no
// assembly yet — this decides whether the mechanism (fewer K/V loads) has ANY merit before an
// ISA-specific port is worth building, matching the brief's own sequencing.
//
// G=6 is the 1.5B's real group size (the brief's own registered decision shape); hd=128 and the
// depth set match BenchmarkMatmulQKAcc64/BenchmarkMatmulAVAcc64's own convention so all four can
// be read side by side.
func BenchmarkMatmulQKAVGroupAB(b *testing.B) {
	const hd = 128
	const G = 6
	const nKV = 2
	for _, depth := range []int{130, 2048, 8192} {
		kvDim := nKV * hd
		q := randF(G * hd)
		keys := randF(depth * kvDim)
		vals := randF(depth * kvDim)
		scores := randF(G * depth)
		qkDst := make([]float32, G*depth)
		avDst := make([]float32, G*hd)
		avAcc1 := make([]float64, hd)
		avAccG := make([]float64, G*hd)

		b.Run(depthName(depth)+"/ungrouped_G_separate_calls", func(b *testing.B) {
			b.SetBytes(int64(depth * kvDim * 4 * 2)) // K and V both re-read per head
			b.ResetTimer()
			for b.Loop() {
				for g := range G {
					MatmulQKAcc64(q[g*hd:(g+1)*hd], keys, qkDst[g*depth:(g+1)*depth], 1, hd, depth, 0, kvDim)
					MatmulAVAcc64(scores[g*depth:(g+1)*depth], vals, avDst[g*hd:(g+1)*hd], avAcc1, 1, depth, hd, 0, kvDim)
				}
			}
		})
		b.Run(depthName(depth)+"/grouped_one_call", func(b *testing.B) {
			b.SetBytes(int64(depth * kvDim * 4 * 2))
			b.ResetTimer()
			for b.Loop() {
				MatmulQKAcc64Group(q, keys, qkDst, G, hd, depth, 0, kvDim)
				MatmulAVAcc64Group(scores, vals, avDst, avAccG, G, depth, hd, 0, kvDim)
			}
		})
	}
}
