package linalg

import "fmt"

// archContractCases: the amd64-only writers. The split-half W4A8 matmul is reached through WeightMat.RepackInt4SplitHalf, which declines (returns false) on an
// AVX-512-VNNI host, so those cases run only where the repack took.
func archContractCases() []contractCase {
	var cs []contractCase
	for _, s := range []struct{ M, K, N int }{{1, 64, 10}, {3, 96, 19}, {8, 64, 33}} {
		s := s
		cs = append(cs, contractCase{fmt.Sprintf("WeightMat.MatmulBTW4A8Into[splithalf]/M=%d,K=%d,N=%d", s.M, s.K, s.N), func() (func(), []*outBuf) {
			wm := weightMatKind("int4", rf32(41, s.N*s.K), s.N, s.K)
			if !wm.RepackInt4SplitHalf() {
				// declined (VNNI host): the canonical path is covered by the portable cases
				return func() {}, nil
			}
			dst, o := mkOut[float32]("dst", s.M*s.N)
			aa := rf32(42, s.M*s.K)
			return func() { wm.MatmulBTW4A8Into(&Workspace{}, aa, dst, s.M) }, []*outBuf{o}
		}})
	}
	return cs
}
