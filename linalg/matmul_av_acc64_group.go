package linalg

// MatmulAVAcc64Group computes dst[G,hd] = scores[G,nKeys] · V_head[nKeys,hd], f64-accumulated —
// G query heads sharing ONE kv head's V data (headOff/rowStride select the shared kv head), each
// with its own score row in scores. G separate MatmulAVAcc64 calls against the identical
// vals/headOff/rowStride currently re-read every key's V row once per head; this reads each key's
// V row ONCE and folds it into G independent per-head dim-accumulators.
//
// BIT-IDENTICAL to calling MatmulAVAcc64(scores[g*nKeys:(g+1)*nKeys], vals, dst[g*hd:(g+1)*hd],
// accScratch, 1, nKeys, hd, headOff, rowStride) once per head g — matmul_av_acc64.go's own header
// already makes this exact argument for its own loop-nest interchange (keys-outer/dims-inner vs
// dims-outer/keys-inner): each (head, dim) accumulator here receives exactly the same
// key-ascending sequence of adds MatmulAVAcc64 would compute for that (head, dim) pair alone; only
// the nesting changes so a key's V row, once loaded, is reused across all G heads instead of
// reloaded per head. TestMatmulAVAcc64Group_matchesUngrouped is the check.
//
// acc is caller-provided [G*hd]-float64 scratch (decode calls this hundreds of times per token;
// see MatmulAVAcc64's own note on why this isn't a fresh allocation per call). Zeroed on entry,
// not preserved on exit.
//
// MEASURED, NOT ASSUMED: see MatmulQKAcc64Group's own note — a first cut here processed one
// head's whole hd-wide accumulator row before moving to the next head (key-outer, head-middle,
// dim-inner), which shares the V load but gives up any cross-accumulator interleaving; restructured
// to interleave the G heads for a FIXED (key, dim) instead (key-outer, dim-middle, head-inner),
// same "acc[g] gets the same fold regardless of what happens to acc[g'] in between" bit-identity
// argument. acc is laid out [hd][G] (dim-major) rather than [G][hd] specifically so the innermost
// head loop walks a contiguous run, not a stride-hd scatter.
func MatmulAVAcc64Group(scores, vals, dst []float32, acc []float64, G, nKeys, hd, headOff, rowStride int) {
	checkMatmulAVAcc64Group(len(scores), len(vals), len(dst), len(acc), G, nKeys, hd, headOff, rowStride)
	// arm64, G=6: whole 8-dim blocks go through the NEON kernel
	// (attn_acc64_group_arm64.s) — 24 f64 lane-pair accumulators (G queries ×
	// 4 dim-pairs), the V row widened once per key and shared across all G
	// queries instead of reloaded per query. Same adds, same key-ascending
	// order per (query, dim); TestAVAcc64NEON8G6_matchesGo pins the kernel
	// against this Go loop block for block. Elsewhere avAcc64GroupBlocks
	// returns 0 and the Go loop below does everything.
	d0 := avAcc64GroupBlocks(scores, vals, dst, G, nKeys, hd, headOff, rowStride)
	if d0 >= hd {
		return
	}
	span := hd - d0
	for i := range G * span {
		acc[i] = 0
	}
	for s := range nKeys {
		vrow := vals[headOff+s*rowStride+d0 : headOff+s*rowStride+hd]
		for d := range span {
			vd := float64(vrow[d])
			arow := acc[d*G : d*G+G]
			for g := range G {
				arow[g] += float64(scores[g*nKeys+s]) * vd
			}
		}
	}
	for g := range G {
		drow := dst[g*hd+d0 : g*hd+hd]
		for d := range span {
			drow[d] = float32(acc[d*G+g])
		}
	}
}

func checkMatmulAVAcc64Group(scoresLen, valsLen, dstLen, accLen, G, nKeys, hd, headOff, rowStride int) {
	if G < 0 || nKeys < 0 || hd < 0 || headOff < 0 || rowStride < hd {
		panic("linalg: MatmulAVAcc64Group invalid shape")
	}
	requireExactLen("MatmulAVAcc64Group", "scores", scoresLen, mul(G, nKeys))
	requireExactLen("MatmulAVAcc64Group", "dst", dstLen, mul(G, hd))
	if accLen < G*hd {
		panic("linalg: MatmulAVAcc64Group acc scratch shorter than G*hd")
	}
	need := headOff + max(0, nKeys-1)*rowStride + hd
	if valsLen < need {
		panic("linalg: MatmulAVAcc64Group vals too short for the given shape/strides")
	}
}
