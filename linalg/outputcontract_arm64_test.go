package linalg

import "fmt"

// archContractCases: the arm64-only writers (the 4-row-interleaved W4A8 family). The matmul kernels need DotProd; the repack helpers are plain Go and always run.
// This file is the reason R-11's brief says the arm64 assembly paths need a run on arm64 hardware: on amd64 none of it is compiled.
func archContractCases() []contractCase {
	var cs []contractCase
	add := func(name string, setup func() (func(), []*outBuf)) { cs = append(cs, contractCase{name, setup}) }

	for _, K := range []int{32, 96} {
		K := K
		add(fmt.Sprintf("RepackInt4Row4Quad/K=%d", K), func() (func(), []*outBuf) {
			src := make([]byte, 4*K/2)
			for i, v := range ri8(int64(K), len(src)) {
				src[i] = byte(v)
			}
			dst, o := mkOut[byte]("dst", 4*K/2)
			return func() { RepackInt4Row4Quad(dst, src, K) }, []*outBuf{o}
		})
		ng := K / 32
		add(fmt.Sprintf("RepackInt4Row4ScalesQuad/nGroups=%d", ng), func() (func(), []*outBuf) {
			src := rf32(5, 4*ng)
			dst, o := mkOut[float32]("dst", 4*ng)
			return func() { RepackInt4Row4ScalesQuad(dst, src, ng) }, []*outBuf{o}
		})
		add(fmt.Sprintf("RepackInt4Row4ScalesQuadF16/nGroups=%d", ng), func() (func(), []*outBuf) {
			src := F32ToF16Scales(rf32(6, 4*ng))
			dst, o := mkOut[uint16]("dst", 4*ng)
			return func() { RepackInt4Row4ScalesQuadF16(dst, src, ng) }, []*outBuf{o}
		})
	}

	if !hasDotProd {
		return cs
	}
	// matmul kernels: M=1 for the decode kernels, M>1 for the tile.
	for _, s := range []struct{ M, K, N int }{{1, 64, 8}, {1, 96, 20}} {
		s := s
		q4 := func() ([]byte, []float32) {
			pk, sc := QuantizeGroupsInt4(rf32(26, s.N*s.K), s.N, s.K, 32)
			return RepackW4A8Row4(pk, s.N, s.K, 32), RepackW4A8Row4Scales(sc, s.N, s.K, 32)
		}
		nm := fmt.Sprintf("M=%d,K=%d,N=%d", s.M, s.K, s.N)
		add("MatmulBTW4A8Row4Into/"+nm, func() (func(), []*outBuf) {
			w, sc := q4()
			dst, o := mkOut[float32]("dst", s.M*s.N)
			aa := rf32(27, s.M*s.K)
			return func() { MatmulBTW4A8Row4Into(&Workspace{}, aa, w, sc, dst, s.M, s.K, s.N, 32) }, []*outBuf{o}
		})
		add("MatmulBTW4A8Row4PrefetchInto/"+nm, func() (func(), []*outBuf) {
			w, sc := q4()
			dst, o := mkOut[float32]("dst", s.M*s.N)
			aa := rf32(27, s.M*s.K)
			return func() { MatmulBTW4A8Row4PrefetchInto(&Workspace{}, aa, w, sc, dst, s.M, s.K, s.N, 32, 4) }, []*outBuf{o}
		})
		add("MatmulBTW4A8Row4DesharedInto/"+nm, func() (func(), []*outBuf) {
			pk, sc := QuantizeGroupsInt4(rf32(26, s.N*s.K), s.N, s.K, 32)
			b0, b1, b2, b3 := RepackW4A8Row4Deshared(pk, s.N, s.K, 32)
			s0, s1, s2, s3 := RepackW4A8Row4DesharedScales(sc, s.N, s.K, 32)
			dst, o := mkOut[float32]("dst", s.M*s.N)
			aa := rf32(27, s.M*s.K)
			return func() {
				MatmulBTW4A8Row4DesharedInto(&Workspace{}, aa, b0, b1, b2, b3, s0, s1, s2, s3, dst, s.M, s.K, s.N, 32)
			}, []*outBuf{o}
		})
	}
	for _, s := range []struct{ M, K, N int }{{2, 64, 8}, {5, 96, 20}, {8, 64, 12}} {
		s := s
		add(fmt.Sprintf("MatmulBTW4A8Row4TileInto/M=%d,K=%d,N=%d", s.M, s.K, s.N), func() (func(), []*outBuf) {
			pk, sc := QuantizeGroupsInt4(rf32(26, s.N*s.K), s.N, s.K, 32)
			w, ss := RepackW4A8Row4(pk, s.N, s.K, 32), RepackW4A8Row4Scales(sc, s.N, s.K, 32)
			dst, o := mkOut[float32]("dst", s.M*s.N)
			aa := rf32(27, s.M*s.K)
			return func() { MatmulBTW4A8Row4TileInto(&Workspace{}, aa, w, ss, dst, s.M, s.K, s.N, 32) }, []*outBuf{o}
		})
	}
	// the WeightMat dispatch over both row4 branches (M=1 decode kernel, M>1 tile)
	for _, s := range []struct{ M, K, N int }{{1, 64, 8}, {4, 96, 20}} {
		s := s
		add(fmt.Sprintf("WeightMat.MatmulBTW4A8Into[row4]/M=%d,K=%d,N=%d", s.M, s.K, s.N), func() (func(), []*outBuf) {
			wm := weightMatKind("int4", rf32(41, s.N*s.K), s.N, s.K)
			if !wm.RepackInt4Row4() {
				return func() {}, nil
			}
			dst, o := mkOut[float32]("dst", s.M*s.N)
			aa := rf32(42, s.M*s.K)
			return func() { wm.MatmulBTW4A8Into(&Workspace{}, aa, dst, s.M) }, []*outBuf{o}
		})
	}
	return cs
}
