package linalg

import (
	"fmt"
	"runtime"
	"strings"
	"testing"
)

// matmulBTW4A8Row4Into refuses a K that is not a positive multiple of 32 by name on
// every path, including with a workspace activation group set, where it used to reach
// the per-group path and index past the 4-row weights.
func TestMatmulBTW4A8Row4Into_raggedKRefusedWithActGroup(t *testing.T) {
	const N, group = 8, 32
	for _, actGroup := range []int{0, 32, 64} {
		for _, K := range []int{0, 1, 31, 33, 127} {
			bpr, ng := (K+1)/2, (K+group-1)/group
			ws := new(Workspace)
			ws.SetActQuantGroup(actGroup)
			a, w4 := make([]float32, K), make([]byte, N*bpr)
			wS, dst := make([]float32, N*ng), make([]float32, N)
			var r any
			func() {
				defer func() { r = recover() }()
				matmulBTW4A8Row4Into(ws, a, w4, wS, dst, 1, K, N, group)
			}()
			name := fmt.Sprintf("actGroup=%d K=%d", actGroup, K)
			if r == nil {
				t.Errorf("%s: returned; want a refusal", name)
				continue
			}
			if _, ok := r.(runtime.Error); ok {
				t.Errorf("%s: crashed with a runtime error: %v", name, r)
				continue
			}
			if msg := fmt.Sprint(r); !strings.Contains(msg, "MatmulBTW4A8Row4") {
				t.Errorf("%s: refused with %q, want a refusal that names the entry point", name, msg)
			}
		}
	}
}
