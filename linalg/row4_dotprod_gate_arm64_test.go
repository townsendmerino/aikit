package linalg

import (
	"fmt"
	"runtime"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/internal/cpufeat"
)

// The exported 4-row W4A8 matmuls run SDOT kernels throughout. On a CPU without
// DotProd each refuses by name, as WeightMat's row4Usable gate does, instead of dying
// with SIGILL. The test turns the flag off, as the aikit_nodotprod tag does.
func TestRow4Matmuls_refuseWithoutDotProd(t *testing.T) {
	defer func(v bool) { cpufeat.HasDotProd = v }(cpufeat.HasDotProd)
	cpufeat.HasDotProd = false
	const K, N, group = 64, 8, 32
	bpr, ng := K/2, K/group
	a, dst := make([]float32, K), make([]float32, N)
	w4, wS := make([]byte, N*bpr), make([]float32, N*ng)
	q4, qS := make([]byte, N/4*bpr), make([]float32, N/4*ng)
	ws := new(Workspace)
	calls := map[string]func(){
		"MatmulBTW4A8Row4Into":     func() { MatmulBTW4A8Row4Into(ws, a, w4, wS, dst, 1, K, N, group) },
		"MatmulBTW4A8Row4TileInto": func() { MatmulBTW4A8Row4TileInto(ws, a, w4, wS, dst, 1, K, N, group) },
		"MatmulBTW4A8Row4PrefetchInto": func() {
			MatmulBTW4A8Row4PrefetchInto(ws, a, w4, wS, dst, 1, K, N, group, 256)
		},
		"MatmulBTW4A8Row4DesharedInto": func() {
			MatmulBTW4A8Row4DesharedInto(ws, a, q4, q4, q4, q4, qS, qS, qS, qS, dst, 1, K, N, group)
		},
	}
	for name, call := range calls {
		var r any
		func() {
			defer func() { r = recover() }()
			call()
		}()
		switch _, crashed := r.(runtime.Error); {
		case r == nil:
			t.Errorf("%s ran without DotProd; want it to refuse", name)
		case crashed:
			t.Errorf("%s crashed without DotProd: %v", name, r)
		case !strings.Contains(fmt.Sprint(r), name+" requires DotProd"):
			t.Errorf("%s refused with %q; want it to name DotProd", name, r)
		}
	}
}
