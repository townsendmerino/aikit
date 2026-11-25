package encoder

import (
	"testing"

	"github.com/townsendmerino/aikit/internal/guardmem"
)

// log1pPoolInto reads and writes pooled[:n] in place and touches nothing past either
// end; pooled sits against a guard page, so a full-vector load or store of a short tail
// in the SIMD build faults instead of reading the next bytes.
func TestLog1pPoolInto_noReadPastShortTail(t *testing.T) {
	a := guardmem.New(t)
	for n := 1; n <= 40; n++ {
		in := make([]float32, n)
		for i := range in {
			in[i] = float32(i%5) * 0.75
		}
		want := append([]float32(nil), in...)
		log1pPoolInto(want)
		for _, atEnd := range []bool{true, false} {
			var x []float32
			if atEnd {
				x = guardmem.EndCopy(a, "pooled", in)
			} else {
				x = guardmem.StartCopy(a, "pooled", in)
			}
			if f := a.Run("log1pPoolInto", func() { log1pPoolInto(x) }); f != nil {
				t.Fatalf("n=%d: %v", n, f)
			}
			for i := range want {
				if x[i] != want[i] {
					t.Fatalf("n=%d: pooled[%d] = %v against the guards, %v on the heap", n, i, x[i], want[i])
				}
			}
		}
	}
}
