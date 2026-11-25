package linalg

import (
	"go/token"
	"maps"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/internal/kernreg"
)

// tinyAdapters drive every exported entry point that takes a shape with each of M, K, N
// and the group count at 0 and 1 and at one under each block size. An adapter's Symbol is
// the exported name ("WeightMat.MatmulBT" for a method), its n indexes a list of shapes,
// and its Call wraps the entry point in kernreg.Refused, so an explicit panic that refuses
// a shape is a result (the same on the heap and against the guards) while a runtime error
// fails the sweep. Sweep's deadline turns a loop that does not end into a failure that
// names the call. Each group of entry points declares its adapters, and the names it
// exempts with the reason, beside its own tests; these lists gather them.
func tinyAdapters() []kernreg.Adapter {
	return slices.Concat(
		tinyMatmulAdapters(),
		tinyWeightMatAdapters(),
		tinyOtherAdapters(),
	)
}

func tinyExempt() map[string]string {
	out := map[string]string{}
	for _, m := range []map[string]string{tinyMatmulExempt, tinyWeightMatExempt, tinyOtherExempt} {
		maps.Copy(out, m)
	}
	return out
}

// TestTinyShapes_everyExportedEntryPoint runs every tiny-shape adapter, with its inputs
// on the heap, against the end guard and against the start guard.
func TestTinyShapes_everyExportedEntryPoint(t *testing.T) {
	kernreg.Sweep(t, tinyAdapters())
}

// TestTinyShapes_complete fails on an exported name of linalg with neither a tiny-shape
// adapter nor a listed reason, and on a listed name that no longer exists.
func TestTinyShapes_complete(t *testing.T) {
	// An adapter of an unexported function (one LS2 or LS6 unexported, still swept for the
	// kernel it reaches) is not part of the completeness check.
	var swept []string
	for _, ad := range tinyAdapters() {
		if i := strings.LastIndexByte(ad.Symbol, '.'); token.IsExported(ad.Symbol[i+1:]) {
			swept = append(swept, ad.Symbol)
		}
	}
	probs, err := kernreg.ExportedProblems(".", swept, tinyExempt())
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range probs {
		t.Error(p)
	}
}

// tinyDims are the sizes each dimension takes: 0, 1, and one under, at and one over each
// block size the kernels use (4, 8, 16, 32, 64).
var tinyDims = []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65}

// tinyGroups are the group sizes swept where an entry point takes one: 0, 1, and one
// under, at and one over the group sizes the int4 and W4A8 formats use.
var tinyGroups = []int{0, 1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129}

// shapeLabels returns a Label for an adapter whose n indexes shapes.
func shapeLabels[S interface{ String() string }](shapes []S) func(int) string {
	return func(i int) string { return shapes[i].String() }
}

// shapeIndices is 0..n-1, the Lengths of an adapter whose n indexes a list of shapes.
func shapeIndices(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = i
	}
	return out
}
