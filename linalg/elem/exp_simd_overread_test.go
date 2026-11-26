package elem

import (
	"testing"

	"github.com/townsendmerino/aikit/internal/guardmem"
)

// The element-wise and softmax entry points read src[:n] and write dst[:n] and touch
// nothing past either end, at every length up to a few vector widths. Their SIMD build's
// tails take a partial vector: the inputs and outputs sit against guard pages so a
// full-width load or store of a short tail faults instead of reading the next bytes.
func TestElementwise_noReadPastShortTail(t *testing.T) {
	fns := map[string]func(dst, src []float32){
		"ExpF32Into":             expF32Into,
		"ExpContractInto":        ExpContractInto,
		"SiLUInto":               siluInto,
		"SiLUContractInto":       SiLUContractInto,
		"GELUInto":               geluInto,
		"GELUContractInto":       GELUContractInto,
		"GELUTanhInto":           GELUTanhInto,
		"GELUTanhContractInto":   GELUTanhContractInto,
		"TanhInto":               TanhInto,
		"SoftmaxRowInto":         softmaxRowInto,
		"SoftmaxRowContractInto": SoftmaxRowContractInto,
		"SoftmaxRowScaledInto":   func(dst, src []float32) { softmaxRowScaledInto(dst, src, 0.125) },
		"SoftmaxRowScaledContractInto": func(dst, src []float32) {
			SoftmaxRowScaledContractInto(dst, src, 0.125)
		},
	}
	a := guardmem.New(t)
	for name, fn := range fns {
		for n := 1; n <= 40; n++ {
			in := make([]float32, n)
			for i := range in {
				in[i] = float32(i%7) - 3
			}
			want := make([]float32, n)
			fn(want, in)
			for _, atEnd := range []bool{true, false} {
				var src, dst []float32
				if atEnd {
					src, dst = guardmem.EndCopy(a, "src", in), guardmem.End[float32](a, "dst", n)
				} else {
					src, dst = guardmem.StartCopy(a, "src", in), guardmem.Start[float32](a, "dst", n)
				}
				if f := a.Run(name, func() { fn(dst, src) }); f != nil {
					t.Fatalf("%s n=%d: %v", name, n, f)
				}
				for i := range want {
					if dst[i] != want[i] {
						t.Fatalf("%s n=%d: dst[%d] = %v against the guards, %v on the heap", name, n, i, dst[i], want[i])
					}
				}
			}
		}
	}
}
