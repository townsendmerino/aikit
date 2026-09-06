package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// refAttendTileFused is goinfer's decoder/fusedattn.go body, FROZEN at the move (M1). It is the
// thing AttendTileFused must reproduce bit-for-bit; it is deliberately a copy rather than a call,
// because a reference that shares code with the thing it checks proves nothing.
func refAttendTileFused(
	mm func(a, b, dst []float32, M, K, N int),
	qh, kh, vBlk, ch []float32,
	sBlk, tmp, acc, mRun, lRun []float32,
	kt, hd, nKeys int, scale float64,
	lo, hi []int,
) bool {
	for i := range kt {
		mRun[i], lRun[i] = float32(math.Inf(-1)), 0
	}
	for i := range kt * hd {
		acc[i] = 0
	}
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
	for k0 := 0; k0 < nKeys; k0 += 512 {
		if k0 > hiMax {
			break
		}
		k1 := min(k0+512, nKeys)
		n := k1 - k0
		if k1-1 < loMin {
			continue
		}
		mm(qh, kh[k0*hd:k1*hd], sBlk[:kt*n], kt, hd, n)
		for i := range kt {
			row := sBlk[i*n : i*n+n]
			a0, a1 := max(lo[i], k0), min(hi[i], k1-1)
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
		if lRun[i] == 0 {
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

// refGatherKVFused is goinfer's gatherKVFused, frozen at the move.
func refGatherKVFused(kh, vBlk, keys, vals []float32, kvh, hd, kvDim, nKeys int) {
	for s := range nKeys {
		kvBase := s*kvDim + kvh*hd
		copy(kh[s*hd:s*hd+hd], keys[kvBase:kvBase+hd])
		b0 := (s / 512) * 512
		n := min(512, nKeys-b0)
		vrow := vals[kvBase : kvBase+hd]
		for d := range hd {
			vBlk[b0*hd+d*n+(s-b0)] = vrow[d]
		}
	}
}

// fusedCase is one (shape, data) configuration the gate drives both implementations with.
type fusedCase struct {
	name          string
	kt, hd, nKeys int
	kvh, kvDim    int
	scale         float64
	window        int // 0 = causal to the row; >0 = sliding window
	fill          func(r *rand.Rand, i int) float32
}

// fusedGateCases spans the shapes goinfer actually calls with AND the ones that break a key-blocked
// schedule: nKeys either side of the 512 block boundary, tails at K%4 / K%8 / K%32, and head dims
// 64/128/256. A kernel that is right at nKeys=1024 and wrong at 1025 is the failure this catches.
func fusedGateCases() []fusedCase {
	wide := func(r *rand.Rand, i int) float32 {
		// wide binades: values spanning ~2^-60..2^60 so the running max actually moves and the
		// rescale path (corr != 1) is exercised rather than skipped.
		return float32(math.Ldexp(r.Float64()*2-1, r.Intn(121)-60))
	}
	small := func(r *rand.Rand, i int) float32 { return float32(r.NormFloat64()) * 0.1 }
	denorm := func(r *rand.Rand, i int) float32 {
		if i%7 == 0 {
			return float32(math.Ldexp(1, -140)) // denormal
		}
		return float32(r.NormFloat64())
	}
	var cs []fusedCase
	for _, hd := range []int{64, 128, 256} {
		for _, nk := range []int{1, 7, 33, 64, 511, 512, 513, 1024, 1025, 1536} {
			cs = append(cs,
				fusedCase{"causal/wide", 8, hd, nk, 1, 2 * hd, 1 / math.Sqrt(float64(hd)), 0, wide},
				fusedCase{"causal/small", 8, hd, nk, 0, hd, 1 / math.Sqrt(float64(hd)), 0, small},
			)
			if nk > 64 {
				cs = append(cs, fusedCase{"window/denorm", 8, hd, nk, 1, 2 * hd, 0.125, 48, denorm})
			}
		}
	}
	return cs
}

// TestAttendTileFused_bitIdenticalToGoinferRef is M1's gate: RAW BITS against the frozen goinfer
// body, never a tolerance. The fused schedule re-associates a softmax denominator, so a tolerance
// test would pass on a genuinely different schedule — the whole point of the move is that goinfer's
// long-prompt goldens, already regenerated once for P19, must not move again.
func TestAttendTileFused_bitIdenticalToGoinferRef(t *testing.T) {
	for _, c := range fusedGateCases() {
		r := rand.New(rand.NewSource(int64(c.kt*1_000_003 + c.hd*10_007 + c.nKeys)))
		keys := make([]float32, c.nKeys*c.kvDim)
		vals := make([]float32, c.nKeys*c.kvDim)
		for i := range keys {
			keys[i] = c.fill(r, i)
			vals[i] = c.fill(r, i+1)
		}
		qh := make([]float32, c.kt*c.hd)
		for i := range qh {
			qh[i] = c.fill(r, i+2)
		}
		lo := make([]int, c.kt)
		hi := make([]int, c.kt)
		for i := range c.kt {
			pos := c.nKeys - c.kt + i // this tile sits at the end of the prefix
			if pos < 0 {
				pos = 0
			}
			hi[i] = pos
			if c.window > 0 && pos-c.window+1 > 0 {
				lo[i] = pos - c.window + 1
			}
		}

		khA := make([]float32, c.nKeys*c.hd)
		vBlkA := make([]float32, c.hd*c.nKeys)
		khB := make([]float32, c.nKeys*c.hd)
		vBlkB := make([]float32, c.hd*c.nKeys)
		refGatherKVFused(khA, vBlkA, keys, vals, c.kvh, c.hd, c.kvDim, c.nKeys)
		GatherVBlockMajor(khB, vBlkB, keys, vals, c.kvh, c.hd, c.kvDim, c.nKeys)
		for i := range khA {
			if math.Float32bits(khA[i]) != math.Float32bits(khB[i]) {
				t.Fatalf("%s hd=%d nKeys=%d: gather kh[%d] %08x != %08x",
					c.name, c.hd, c.nKeys, i, math.Float32bits(khA[i]), math.Float32bits(khB[i]))
			}
		}
		for i := range vBlkA {
			if math.Float32bits(vBlkA[i]) != math.Float32bits(vBlkB[i]) {
				t.Fatalf("%s hd=%d nKeys=%d: gather vBlk[%d] %08x != %08x",
					c.name, c.hd, c.nKeys, i, math.Float32bits(vBlkA[i]), math.Float32bits(vBlkB[i]))
			}
		}

		chA := make([]float32, c.kt*c.hd)
		chB := make([]float32, c.kt*c.hd)
		refAttendTileFused(MatmulBT, qh, khA, vBlkA, chA,
			make([]float32, c.kt*512), make([]float32, c.kt*c.hd), make([]float32, c.kt*c.hd),
			make([]float32, c.kt), make([]float32, c.kt),
			c.kt, c.hd, c.nKeys, c.scale, lo, hi)
		sc := NewFusedAttnScratch(c.kt, c.hd)
		if !AttendTileFused(MatmulBT, qh, khB, vBlkB, chB, sc, c.kt, c.hd, c.nKeys, c.scale, lo, hi) {
			t.Fatalf("%s hd=%d nKeys=%d: AttendTileFused declined a shape the reference accepted",
				c.name, c.hd, c.nKeys)
		}
		for i := range chA {
			if math.Float32bits(chA[i]) != math.Float32bits(chB[i]) {
				t.Fatalf("%s hd=%d nKeys=%d: ch[%d] ref %08x (%v) != aikit %08x (%v)",
					c.name, c.hd, c.nKeys, i, math.Float32bits(chA[i]), chA[i],
					math.Float32bits(chB[i]), chB[i])
			}
		}
	}
}

// TestAttendTileFused_matchesAcc64WithinCosine is the SANITY BOUND, not a bit gate. 0.9982 is P19's
// own measured model-level figure for acc64-vs-fused; the fused path is documented non-identical to
// acc64, so this can only ever say "still the same kernel, not a different one", and it is recorded
// as such.
func TestAttendTileFused_matchesAcc64WithinCosine(t *testing.T) {
	const kt, hd, nKeys, kvDim, kvh = 8, 128, 700, 128, 0
	r := rand.New(rand.NewSource(99))
	keys := make([]float32, nKeys*kvDim)
	vals := make([]float32, nKeys*kvDim)
	for i := range keys {
		keys[i] = float32(r.NormFloat64())
		vals[i] = float32(r.NormFloat64())
	}
	qh := make([]float32, kt*hd)
	for i := range qh {
		qh[i] = float32(r.NormFloat64())
	}
	lo := make([]int, kt)
	hi := make([]int, kt)
	for i := range kt {
		hi[i] = nKeys - kt + i
	}
	scale := 1 / math.Sqrt(float64(hd))

	kh := make([]float32, nKeys*hd)
	vBlk := make([]float32, hd*nKeys)
	GatherVBlockMajor(kh, vBlk, keys, vals, kvh, hd, kvDim, nKeys)
	got := make([]float32, kt*hd)
	sc := NewFusedAttnScratch(kt, hd)
	if !AttendTileFused(MatmulBT, qh, kh, vBlk, got, sc, kt, hd, nKeys, scale, lo, hi) {
		t.Fatal("declined")
	}

	// Materialized acc64 reference: QK -> scaled softmax -> AV, the schedule fusion replaces.
	// Called exactly as goinfer's materialized path calls them (forwardn.go), strides and all.
	scores := make([]float32, kt*nKeys)
	MatmulQKAcc64(qh, keys, scores, kt, hd, nKeys, kvh*hd, kvDim)
	for i := range kt {
		row := scores[i*nKeys : i*nKeys+nKeys]
		lim := hi[i] + 1
		mx := float32(math.Inf(-1))
		for j := range lim {
			row[j] = float32(float64(row[j]) * scale)
			if row[j] > mx {
				mx = row[j]
			}
		}
		var sum float64
		for j := range lim {
			e := math.Exp(float64(row[j] - mx))
			row[j] = float32(e)
			sum += e
		}
		inv := float32(1 / sum)
		for j := range lim {
			row[j] *= inv
		}
		for j := lim; j < nKeys; j++ {
			row[j] = 0
		}
	}
	want := make([]float32, kt*hd)
	avAcc := make([]float64, hd)
	MatmulAVAcc64(scores, vals, want, avAcc, kt, nKeys, hd, kvh*hd, kvDim)

	for i := range kt {
		var dot, na, nb float64
		for d := range hd {
			x, y := float64(got[i*hd+d]), float64(want[i*hd+d])
			dot += x * y
			na += x * x
			nb += y * y
		}
		cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
		if cos < 0.9982 {
			t.Errorf("row %d: cosine %.9f vs acc64 below the 0.9982 sanity bound", i, cos)
		}
	}
}
