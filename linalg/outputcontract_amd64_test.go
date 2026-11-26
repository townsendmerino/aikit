package linalg

import (
	"fmt"

	"github.com/townsendmerino/aikit/linalg/internal/outputcontract"
)

// archContractCases: the amd64-only writers. The split-half W4A8 matmul is reached through WeightMat.RepackInt4SplitHalf, which declines (returns false) on an
// AVX-512-VNNI host, so those cases run only where the repack took.
func archContractCases() []outputcontract.Case {
	var cs []outputcontract.Case
	for _, s := range []struct{ M, K, N int }{{1, 64, 10}, {3, 96, 19}, {8, 64, 33}} {
		cs = append(cs, outputcontract.Case{Name: fmt.Sprintf("WeightMat.MatmulBTW4A8Into[splithalf]/M=%d,K=%d,N=%d", s.M, s.K, s.N), Setup: func() (func(), []*outputcontract.OutBuf) {
			wm := weightMatKind("int4", rf32(41, s.N*s.K), s.N, s.K)
			if !wm.RepackInt4SplitHalf() {
				// declined (VNNI host): the canonical path is covered by the portable cases
				return func() {}, nil
			}
			dst, o := outputcontract.MkOut[float32]("dst", s.M*s.N)
			aa := rf32(42, s.M*s.K)
			return func() { wm.MatmulBTW4A8Into(&Workspace{}, aa, dst, s.M) }, []*outputcontract.OutBuf{o}
		}})
	}
	return cs
}
