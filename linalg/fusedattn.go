package linalg

import "math"

// Fused (FlashAttention-style) attention over one query tile, keeping the score block resident.
//
// WHAT THIS IS. The materialized schedule writes a kt x nKeys score block, reads and rewrites it in
// the softmax, and reads it again for scores*V: three trips through memory for a block that is
// megabytes at a production tile budget. This blocks over KEYS instead, keeping the score block
// small enough to stay in cache and folding it into the output accumulator with a running max and a
// running sum, so the kt x nKeys matrix never exists.
//
// IT IS NOT BIT-IDENTICAL TO A MATERIALIZED QK -> SOFTMAX -> AV, and that is structural rather than
// incidental: the running-max rescale re-associates both the softmax denominator and the AV fold.
// Callers that need bit-identity must not use it. AttendTileFused is gated raw-bit against a frozen
// reference of ITSELF (fusedattn_test.go), which is the property that actually travels — "the fused
// schedule computes what it computed yesterday" — and separately against
// MatmulQKAcc64+softmax+MatmulAVAcc64 at a cosine floor, which is a sanity bound and NOT a bit gate.
//
// THE f64 math.Exp IS PART OF THE BITS, NOT AN OVERSIGHT. Both exponentials below widen to float64,
// call math.Exp, and narrow back. Replacing them with a float32 exp (linalg.ExpF32) changes results
// and is a numerics decision for the caller, not a cleanup — do not "fix" it here.
//
// SCRATCH IS EXPLICIT (FusedAttnScratch), never pooled internally: a caller running one of these per
// worker needs to own the lifetime, and a hidden pool would either allocate per call or serialise
// them. See NewFusedAttnScratch / Fits for the reuse contract.
//
// Provenance: moved from goinfer decoder/fusedattn.go (P19), body verbatim.
// See docs/task-goinfer-kernel-moves.md M1.

// FusedAttnKeyBlock is the key-block width. 256 and 512 measured within noise of each other and
// both beat 1024; 512 keeps the per-tile score block at kt*512 floats, which is the point — small
// enough to stay resident, which is the entire mechanism.
const FusedAttnKeyBlock = 512

// FusedAttnScratch holds the per-worker buffers AttendTileFused needs. Every buffer is used as a
// PREFIX slice, so a larger one is always acceptable — that is what lets a pool slot be reused
// across shapes rather than reallocated per call.
type FusedAttnScratch struct {
	SBlk, Tmp, Acc, MRun, LRun []float32
}

// NewFusedAttnScratch allocates scratch for a tile of kt query rows and head dim hd.
func NewFusedAttnScratch(kt, hd int) *FusedAttnScratch {
	return &FusedAttnScratch{
		SBlk: make([]float32, kt*FusedAttnKeyBlock),
		Tmp:  make([]float32, kt*hd),
		Acc:  make([]float32, kt*hd),
		MRun: make([]float32, kt),
		LRun: make([]float32, kt),
	}
}

// Fits reports whether this scratch is already large enough for the requested shape.
func (f *FusedAttnScratch) Fits(kt, hd int) bool {
	return f != nil &&
		len(f.SBlk) >= kt*FusedAttnKeyBlock &&
		len(f.Tmp) >= kt*hd && len(f.Acc) >= kt*hd &&
		len(f.MRun) >= kt && len(f.LRun) >= kt
}

// AttendTileFused computes one query tile's attention into ch, keeping the score block resident.
// It returns false when it declines, in which case the caller must run a materialized path instead.
//
//	mm    matmul callback, (a, b, dst, M, K, N) — linalg.MatmulBT satisfies it
//	qh    gathered Q for this tile, [kt, hd]
//	kh    gathered K, [nKeys, hd]
//	vBlk  gathered V in BLOCK-MAJOR layout — see GatherVBlockMajor
//	ch    output, [kt, hd]
//	lo/hi per-row INCLUSIVE key bounds, already mapped to physical columns
//
// lo/hi express a contiguous per-row key range. A predicate mask that is not contiguous (tree
// attention, for one) cannot be folded into a key-blocked running softmax this way; such callers
// must use a materialized path.
func AttendTileFused(
	mm func(a, b, dst []float32, M, K, N int),
	qh, kh, vBlk, ch []float32,
	sc *FusedAttnScratch,
	kt, hd, nKeys int, scale float64,
	lo, hi []int,
) bool {
	if mm == nil || !sc.Fits(kt, hd) {
		return false
	}
	sBlk, tmp, acc, mRun, lRun := sc.SBlk, sc.Tmp, sc.Acc, sc.MRun, sc.LRun
	for i := range kt {
		mRun[i], lRun[i] = float32(math.Inf(-1)), 0
	}
	for i := range kt * hd {
		acc[i] = 0
	}
	// The widest bound any row in this tile needs: past it every row is masked, so the whole
	// remaining key range is skippable. This is the block-skip that causality buys, and it is worth
	// nothing on the LAST tile of a prompt and a great deal on the first.
	hiMax := -1
	loMin := nKeys
	for i := range kt {
		if hi[i] > hiMax {
			hiMax = hi[i]
		}
		if lo[i] < loMin {
			loMin = lo[i]
		}
	}
	for k0 := 0; k0 < nKeys; k0 += FusedAttnKeyBlock {
		if k0 > hiMax {
			break
		}
		k1 := min(k0+FusedAttnKeyBlock, nKeys)
		n := k1 - k0
		// A block entirely below every row's window start contributes nothing — every row's
		// [lo,hi] fails to overlap it, so the per-row a0>a1 branch below would zero it and
		// continue without touching acc/mRun/lRun. Skipping the two matmuls changes nothing about
		// the result; it just stops paying for output every row already discards.
		if k1-1 < loMin {
			continue
		}
		mm(qh, kh[k0*hd:k1*hd], sBlk[:kt*n], kt, hd, n)
		for i := range kt {
			row := sBlk[i*n : i*n+n]
			a0, a1 := max(lo[i], k0), min(hi[i], k1-1) // this row's allowed sub-range in the block
			if a0 > a1 {
				for j := range row {
					row[j] = 0
				}
				continue
			}
			j0, j1 := a0-k0, a1-k0
			blkMax := float32(math.Inf(-1))
			for j := j0; j <= j1; j++ {
				row[j] = float32(float64(row[j]) * scale)
				if row[j] > blkMax {
					blkMax = row[j]
				}
			}
			mNew := mRun[i]
			if blkMax > mNew {
				mNew = blkMax
			}
			corr := float32(math.Exp(float64(mRun[i] - mNew)))
			var sum float64
			for j := range j0 {
				row[j] = 0
			}
			for j := j0; j <= j1; j++ {
				e := math.Exp(float64(row[j] - mNew))
				row[j] = float32(e)
				sum += e
			}
			for j := j1 + 1; j < n; j++ {
				row[j] = 0
			}
			lRun[i] = lRun[i]*corr + float32(sum)
			mRun[i] = mNew
			if corr != 1 {
				av := acc[i*hd : (i+1)*hd]
				for d := range av {
					av[d] *= corr
				}
			}
		}
		mm(sBlk[:kt*n], vBlk[k0*hd:k0*hd+hd*n], tmp[:kt*hd], kt, n, hd)
		for i := range kt * hd {
			acc[i] += tmp[i]
		}
	}
	for i := range kt {
		av := acc[i*hd : (i+1)*hd]
		o := ch[i*hd : i*hd+hd]
		if lRun[i] == 0 { // no key in range — leave the row zero, as a masked path does
			for d := range o {
				o[d] = 0
			}
			continue
		}
		inv := 1 / lRun[i]
		for d := range av {
			o[d] = av[d] * inv
		}
	}
	return true
}

// GatherVBlockMajor writes one kv head's V in the BLOCK-MAJOR layout AttendTileFused requires:
// block b occupies [b*FusedAttnKeyBlock*hd, ...) holding [hd, n] contiguously, which is what the mm
// callback's b operand needs for the per-block AV fold.
//
// A materialized path wants [hd, nKeys], and a key-range slice of THAT is not contiguous — which is
// why the layout is chosen at gather time rather than re-transposed per block. Same work, different
// order: the gather is not made more expensive by this, only differently indexed.
//
// It also gathers this head's K into kh as [nKeys, hd]. vals/keys are the caller's cache, strided by
// kvDim with this head at offset kvh*hd.
func GatherVBlockMajor(kh, vBlk, keys, vals []float32, kvh, hd, kvDim, nKeys int) {
	for s := range nKeys {
		kvBase := s*kvDim + kvh*hd
		copy(kh[s*hd:s*hd+hd], keys[kvBase:kvBase+hd])
		b0 := (s / FusedAttnKeyBlock) * FusedAttnKeyBlock
		n := min(FusedAttnKeyBlock, nKeys-b0)
		vrow := vals[kvBase : kvBase+hd]
		for d := range hd {
			vBlk[b0*hd+d*n+(s-b0)] = vrow[d]
		}
	}
}
