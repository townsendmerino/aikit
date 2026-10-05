package encoder

import (
	"math"
	"math/rand"
	"testing"
)

// R-11 (goinfer docs/tasks/task-recompute-audit.md §5): Backend.MatmulBT's documented output contract is "overwrites dst[:M*N]; do not pre-zero". The CPU backend is the
// one implementation reachable on every host; the device backends (gpu/enccuda, gpu/encmetal) read the result back over dst and fall back to this one on small shapes.
// Poison dst with NaN, call, and require every covered element finite and the guard past it untouched — at a serial shape and one large enough to take the intra-op
// parallel path.
func TestOutputContract_cpuBackendMatmulBTOverwrites(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	fill := func(n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = rng.Float32()*2 - 1
		}
		return x
	}
	be, err := NewBackend("cpu")
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range []struct{ M, K, N int }{{1, 8, 3}, {5, 64, 33}, {16, 384, 1536}} {
		a, b := fill(s.M*s.K), fill(s.N*s.K)
		const guard = 8
		var res [2][]float32
		for i, poison := range []float32{float32(math.NaN()), 7.5} {
			dst := make([]float32, s.M*s.N+guard)
			for j := range dst {
				dst[j] = poison
			}
			be.MatmulBT(a, b, dst[:s.M*s.N], s.M, s.K, s.N)
			for j, v := range dst[:s.M*s.N] {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatalf("M=%d K=%d N=%d: dst[%d] = %v after the call — the covered region was not overwritten", s.M, s.K, s.N, j, v)
				}
			}
			for j, v := range dst[s.M*s.N:] {
				if v != poison && !(math.IsNaN(float64(v)) && math.IsNaN(float64(poison))) {
					t.Fatalf("M=%d K=%d N=%d: wrote past dst[:M*N] (guard %d = %v)", s.M, s.K, s.N, j, v)
				}
			}
			res[i] = dst[:s.M*s.N]
		}
		for j := range res[0] {
			if math.Float32bits(res[0][j]) != math.Float32bits(res[1][j]) {
				t.Fatalf("M=%d K=%d N=%d: result depends on dst's prior contents at %d", s.M, s.K, s.N, j)
			}
		}
	}
}
