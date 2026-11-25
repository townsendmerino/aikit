package elem

import (
	"fmt"
	"go/token"
	"maps"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/internal/kernreg"
)

// The element-wise entry points over tiny lengths (OG3): every exported one, and the
// unexported forms elem keeps for linalg's callers, with dst and src against the end guard
// and the start guard, under Sweep's deadline. TestTinyShapes_complete holds the list to
// elem's exported names.

// tinyLens is each length swept: 0, 1, one under, at and one over each block size the
// kernels use (4, 8, 16, 32, 64), and the K-quant block size's neighbours.
var tinyLens = []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257}

var tinyUnary = map[string]func(dst, src []float32){
	"ExpContractInto":              ExpContractInto,
	"expF32Into":                   expF32Into,
	"GELUContractInto":             GELUContractInto,
	"geluInto":                     geluInto,
	"GELUTanhContractInto":         GELUTanhContractInto,
	"GELUTanhInto":                 GELUTanhInto,
	"SiLUContractInto":             SiLUContractInto,
	"siluInto":                     siluInto,
	"SoftmaxRowContractInto":       SoftmaxRowContractInto,
	"softmaxRowInto":               softmaxRowInto,
	"TanhInto":                     TanhInto,
	"softmaxRowScaledInto":         func(dst, src []float32) { softmaxRowScaledInto(dst, src, 0.125) },
	"SoftmaxRowScaledContractInto": func(dst, src []float32) { SoftmaxRowScaledContractInto(dst, src, 0.125) },
}

var tinyExempt = map[string]string{
	"GELUTanhF32": "a scalar function of one value; no shape",
	"SiLUF32":     "a scalar function of one value; no shape",
}

type tinyResult struct {
	Refused string
	Out     []float32
}

func tinyAdapters() []kernreg.Adapter {
	var ads []kernreg.Adapter
	idx := make([]int, len(tinyLens))
	for i := range idx {
		idx[i] = i
	}
	for _, name := range slices.Sorted(maps.Keys(tinyUnary)) {
		f := tinyUnary[name]
		ads = append(ads, kernreg.Adapter{
			Symbol:  name,
			Lengths: idx,
			Label:   func(i int) string { return fmt.Sprintf("n=%d", tinyLens[i]) },
			Call: func(p *kernreg.Placer, i int) any {
				n := tinyLens[i]
				src := kernreg.Put(p, "src", seqF32(n, 1))
				dst := kernreg.Put(p, "dst", make([]float32, n))
				return tinyResult{kernreg.Refused(func() { f(dst, src) }), dst}
			},
		})
	}
	return ads
}

func TestTinyShapes_everyExportedEntryPoint(t *testing.T) {
	kernreg.Sweep(t, tinyAdapters())
}

func TestTinyShapes_complete(t *testing.T) {
	probs, err := kernreg.ExportedProblems(".", exportedSymbols(tinyAdapters()), tinyExempt)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range probs {
		t.Error(p)
	}
}

// exportedSymbols is the adapters' symbols that are exported names (a method's own name
// decides): the unexported forms are swept too but are not part of the completeness check.
func exportedSymbols(ads []kernreg.Adapter) []string {
	var out []string
	for _, ad := range ads {
		if i := strings.LastIndexByte(ad.Symbol, '.'); token.IsExported(ad.Symbol[i+1:]) {
			out = append(out, ad.Symbol)
		}
	}
	return out
}
