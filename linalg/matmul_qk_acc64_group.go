package linalg

// MatmulQKAcc64Group computes dst[G,N] = a[G,K] · bMat[N,K]ᵀ, f64-accumulated — G query heads
// that share ONE kv head's K data (bOff/rowStride select the shared kv head, same convention as
// MatmulQKAcc64), each with its own query row in a. This is decode's actual GQA shape: G query
// heads' worth of MatmulQKAcc64 calls against the identical bMat/bOff/rowStride, currently issued
// as G separate calls that each re-read every key row from scratch.
//
// BIT-IDENTICAL to calling MatmulQKAcc64(a[g*K:(g+1)*K], bMat, dst[g*N:(g+1)*N], 1, K, N, bOff,
// rowStride) once per head g, by the same argument matmul_av_acc64.go's own header already makes
// for its loop-nest interchange: floating-point addition depends on the order of operations
// applied to ONE accumulator, not on what unrelated accumulators do in between. Each (head, key)
// dot product here is still the exact d=0..K-1 ascending sequential fold MatmulQKAcc64 computes
// for that same (head, key) pair — only the loop NESTING changes (key-outer, head-inner, so each
// key's row is read once and reused across G heads instead of read once per head). No reassociation,
// no reduction-order change — the gain is memory traffic, not arithmetic. TestMatmulQKAcc64Group_
// matchesUngrouped is the check, not just the argument.
//
// MEASURED, NOT ASSUMED: a first cut that read one key's row then folded each head's dot
// SEQUENTIALLY (G separate d=0..K-1 chains per key, one after another) was 4.3-7.3x SLOWER than
// today's G separate MatmulQKAcc64 calls (matmul_group_acc64_bench_test.go), despite sharing the
// K load — because that version had no FMA-latency hiding at all, while MatmulQKAcc64 interleaves
// 8 keys' chains per head AND takes a NEON-accelerated 16-key path. Sharing a load buys nothing if
// the arithmetic that follows it is slower than what it replaced. This version instead interleaves
// the G heads' chains for a FIXED key — every real G here is <= 8, the same width goinfer's own
// key-interleaving already validated as the FMA-latency sweet spot on this hardware — so the gain
// composes with the shared load instead of being cancelled by giving up interleaving to get it.
// Still bit-identical: acc[g] receives exactly the same d-ascending sequence of adds regardless of
// what happens to acc[g'] in between (float64 addition is a per-variable operation) — the same
// argument matmul_av_acc64.go's header already makes for its own loop-nest interchange.
//
// Deliberately still the plain Go form (no NEON) — R13's own Build text calls for "Go definitions
// first (they become the oracle)" before the assembly ports; this is that oracle, now shaped so a
// later NEON/AVX2 port has a real per-head-interleaved chain to translate rather than a
// sequential one that would need restructuring anyway.
func MatmulQKAcc64Group(a, bMat, dst []float32, G, K, N, bOff, rowStride int) {
	checkMatmulQKAcc64Group(len(a), len(bMat), len(dst), G, K, N, bOff, rowStride)
	// arm64, G=6: whole 8-key blocks go through the NEON kernel
	// (attn_acc64_group_arm64.s) — 24 f64 lane-pair accumulators (G queries ×
	// 4 key-pairs), each K quad widened once per key-pair and shared across
	// all G queries' FMLA chains instead of reloaded per query. Same adds,
	// same d-ascending order per (query, key); TestQKAcc64NEON8G6_matchesGo
	// pins the kernel against this Go loop block for block. Elsewhere
	// qkAcc64GroupKeys returns 0 and the Go loop below does everything.
	j0 := qkAcc64GroupKeys(a, bMat, dst, G, K, N, bOff, rowStride)
	acc := make([]float64, G)
	for j := j0; j < N; j++ {
		row := bMat[bOff+j*rowStride : bOff+j*rowStride+K]
		for g := range G {
			acc[g] = 0
		}
		for d := range K {
			rd := float64(row[d])
			for g := range G {
				acc[g] += float64(a[g*K+d]) * rd
			}
		}
		for g := range G {
			dst[g*N+j] = float32(acc[g])
		}
	}
}

func checkMatmulQKAcc64Group(aLen, bLen, dstLen, G, K, N, bOff, rowStride int) {
	if G < 0 || K < 0 || N < 0 || bOff < 0 || rowStride < K {
		panic("linalg: MatmulQKAcc64Group invalid shape")
	}
	requireExactLen("MatmulQKAcc64Group", "a", aLen, mul(G, K))
	requireExactLen("MatmulQKAcc64Group", "dst", dstLen, mul(G, N))
	need := bOff + max(0, N-1)*rowStride + K
	if bLen < need {
		panic("linalg: MatmulQKAcc64Group bMat too short for the given shape/strides")
	}
}
