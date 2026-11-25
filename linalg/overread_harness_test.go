package linalg

import (
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/internal/guardmem"
)

// TestOverReadHarnessCatchesAssembly proves that a read past a guard page inside one of
// this package's assembly kernels comes back as a named failure rather than killing the
// test binary: dotNEON reaches dotFMA on amd64 with AVX2 and dotNEON4 on arm64. With the
// length the inputs hold it must not fault and must equal the heap call; asked for one
// more 4-float block it must fault on an input's end guard.
func TestOverReadHarnessCatchesAssembly(t *testing.T) {
	a := guardmem.New(t)
	const n4 = 3
	ha, hb := make([]float32, 4*n4), make([]float32, 4*n4)
	for i := range ha {
		ha[i], hb[i] = float32(i+1), float32(2*i-5)
	}
	want := dotNEON(&ha[0], &hb[0], n4)

	ga, gb := guardmem.EndCopy(a, "a", ha), guardmem.EndCopy(a, "b", hb)
	var got float32
	if f := a.Run("dotNEON", func() { got = dotNEON(&ga[0], &gb[0], n4) }); f != nil {
		t.Fatal("in-bounds call faulted")
	}
	if got != want {
		t.Fatalf("guarded dotNEON = %v, heap = %v", got, want)
	}

	f := a.Catch("dotNEON", func() { dotNEON(&ga[0], &gb[0], n4+1) })
	if f == nil {
		t.Fatal("dotNEON read a block past both inputs without faulting: the harness cannot see an over-read here")
	}
	if !strings.Contains(f.Error(), "dotNEON: read outside input ") || !strings.Contains(f.Error(), "len 12 float32") {
		t.Fatalf("fault not named: %v", f)
	}
	t.Log(f)
}
