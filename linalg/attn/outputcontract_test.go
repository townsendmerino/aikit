package attn

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/aikit/linalg/internal/outputcontract"
)

// The output-contract gate over attn's writers (the harness and its rules are linalg/internal/outputcontract's package
// doc). Run it per build: the host's default kernels, GOARCH=386 for the generic ones, -tags aikit_noavx2 for the
// narrower amd64 tier.
func TestOutputContract_overwriteOrZeroInternally(t *testing.T) {
	for _, c := range outputContractCases() {
		t.Run(c.Name, func(t *testing.T) { outputcontract.Run(t, c) })
	}
}

func outputContractCases() []outputcontract.Case {
	var cs []outputcontract.Case
	add := func(name string, setup func() (func(), []*outputcontract.OutBuf)) {
		cs = append(cs, outputcontract.Case{Name: name, Setup: setup})
	}

	// ---- MatmulQKAcc64 at the dense matmuls' shapes, keys contiguous
	for _, s := range []struct{ M, K, N int }{{1, 64, 10}, {3, 96, 19}, {8, 64, 33}} {
		add(fmt.Sprintf("MatmulQKAcc64/M=%d,K=%d,N=%d", s.M, s.K, s.N), func() (func(), []*outputcontract.OutBuf) {
			dst, o := outputcontract.MkOut[float32]("dst", s.M*s.N)
			aa, bb := randF32(20, s.M*s.K), randF32(21, s.N*s.K)
			return func() { MatmulQKAcc64(aa, bb, dst, s.M, s.K, s.N, 0, s.K) }, []*outputcontract.OutBuf{o}
		})
	}

	// ---- attention kernels (f64 accumulated): scratch is part of the contract too
	for _, g := range [][4]int{{1, 20, 16, 0}, {3, 33, 64, 16}, {6, 70, 64, 64}} {
		M, nKeys, hd, headOff := g[0], g[1], g[2], g[3]
		rowStride := headOff + hd + 8
		nm := fmt.Sprintf("M=%d,nKeys=%d,hd=%d,headOff=%d", M, nKeys, hd, headOff)
		add("MatmulAVAcc64/"+nm, func() (func(), []*outputcontract.OutBuf) {
			scores, vals := randF32(40, M*nKeys), randF32(41, nKeys*rowStride)
			dst, o1 := outputcontract.MkOut[float32]("dst", M*hd)
			acc, o2 := outputcontract.MkOut[float64]("acc", hd)
			return func() { MatmulAVAcc64(scores, vals, dst, acc, M, nKeys, hd, headOff, rowStride) }, []*outputcontract.OutBuf{o1, o2.AsScratch()}
		})
		add("MatmulAVAcc64Group/"+nm, func() (func(), []*outputcontract.OutBuf) {
			scores, vals := randF32(42, M*nKeys), randF32(43, nKeys*rowStride)
			dst, o1 := outputcontract.MkOut[float32]("dst", M*hd)
			acc, o2 := outputcontract.MkOut[float64]("acc", M*hd)
			return func() { MatmulAVAcc64Group(scores, vals, dst, acc, M, nKeys, hd, headOff, rowStride) }, []*outputcontract.OutBuf{o1, o2.AsScratch()}
		})
		K, N := hd, nKeys
		qkStride := headOff + K + 4
		add("MatmulQKAcc64/"+nm, func() (func(), []*outputcontract.OutBuf) {
			a, bm := randF32(44, M*K), randF32(45, (N-1)*qkStride+headOff+K+qkStride)
			dst, o := outputcontract.MkOut[float32]("dst", M*N)
			return func() { MatmulQKAcc64(a, bm, dst, M, K, N, headOff, qkStride) }, []*outputcontract.OutBuf{o}
		})
		add("MatmulQKAcc64Group/"+nm, func() (func(), []*outputcontract.OutBuf) {
			a, bm := randF32(46, M*K), randF32(47, (N-1)*qkStride+headOff+K+qkStride)
			dst, o := outputcontract.MkOut[float32]("dst", M*N)
			return func() { MatmulQKAcc64Group(a, bm, dst, M, K, N, headOff, qkStride) }, []*outputcontract.OutBuf{o}
		})
	}
	for _, fc := range []struct {
		name         string
		contractExp  bool
		maskFirstRow bool
	}{{"AttendTileFused", false, false}, {"AttendTileFusedContractExp", true, false}, {"AttendTileFused/maskedRow", false, true}} {
		for _, nKeys := range []int{70, 600} {
			add(fmt.Sprintf("%s/nKeys=%d", fc.name, nKeys), func() (func(), []*outputcontract.OutBuf) {
				const kt, hd = 4, 64
				qh, kh := randF32(50, kt*hd), randF32(51, nKeys*hd)
				vBlk := randF32(52, hd*nKeys)
				ch, o1 := outputcontract.MkOut[float32]("ch", kt*hd)
				sbl, o2 := outputcontract.MkOut[float32]("sc.SBlk", kt*FusedAttnKeyBlock)
				tmp, o3 := outputcontract.MkOut[float32]("sc.Tmp", kt*hd)
				acc, o4 := outputcontract.MkOut[float32]("sc.Acc", kt*hd)
				mr, o5 := outputcontract.MkOut[float32]("sc.MRun", kt)
				lr, o6 := outputcontract.MkOut[float32]("sc.LRun", kt)
				sc := &FusedAttnScratch{SBlk: sbl, Tmp: tmp, Acc: acc, MRun: mr, LRun: lr}
				lo, hi := make([]int, kt), make([]int, kt)
				for i := range kt {
					hi[i] = nKeys - kt + i
				}
				if fc.maskFirstRow {
					lo[0], hi[0] = 1, 0 // an empty range: its output row must still be written (zero), not left alone
				}
				run := AttendTileFused
				if fc.contractExp {
					run = AttendTileFusedContractExp
				}
				return func() {
					if !run(linalg.MatmulBT, qh, kh, vBlk, ch, sc, kt, hd, nKeys, 0.125, lo, hi) {
						panic("declined")
					}
				}, []*outputcontract.OutBuf{o1, o2.AsScratch(), o3.AsScratch(), o4.AsScratch(), o5.AsScratch(), o6.AsScratch()}
			})
		}
	}
	add("GatherVBlockMajor", func() (func(), []*outputcontract.OutBuf) {
		const hd, kvh, kvDim, nKeys = 64, 1, 128, 600
		keys, vals := randF32(53, nKeys*kvDim), randF32(54, nKeys*kvDim)
		kh, o1 := outputcontract.MkOut[float32]("kh", nKeys*hd)
		vb, o2 := outputcontract.MkOut[float32]("vBlk", hd*nKeys)
		return func() { GatherVBlockMajor(kh, vb, keys, vals, kvh, hd, kvDim, nKeys) }, []*outputcontract.OutBuf{o1, o2}
	})

	return cs
}

// nonDstWriters are attn's writers whose output parameter is not called dst, so the name scan cannot find them.
var nonDstWriters = map[string]bool{"AttendTileFused": true, "AttendTileFusedContractExp": true, "GatherVBlockMajor": true}

// TestOutputContract_everyWriterIsDocumented: every exported attn writer carries an "Output contract:" line in one of
// outputcontract.ContractForm's forms, and nonDstWriters names only writers that exist.
func TestOutputContract_everyWriterIsDocumented(t *testing.T) {
	seen := map[string]bool{}
	for _, d := range outputcontract.ParseDecls(t, ".") {
		if !d.Exported {
			continue
		}
		if strings.Contains(d.Doc, "Output contract:") && !outputcontract.ContractForm.MatchString(d.Doc) {
			t.Errorf("%s %s: 'Output contract:' line is not one of the recognised forms", d.File, d.Key)
		}
		seen[d.Key] = true
		if (d.OutParam || nonDstWriters[d.Key]) && !outputcontract.ContractForm.MatchString(d.Doc) {
			t.Errorf("%s: %s writes into a caller's buffer but its doc comment has no 'Output contract:' line", d.File, d.Key)
		}
	}
	for n := range nonDstWriters {
		if !seen[n] {
			t.Errorf("nonDstWriters lists %s, which no longer exists in attn", n)
		}
	}
}

func randF32(seed int64, n int) []float32 {
	r := rand.New(rand.NewSource(seed))
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(r.NormFloat64())
	}
	return v
}
