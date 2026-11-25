package bitvec

import (
	"fmt"
	"go/token"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/internal/kernreg"
)

// The binary-vector entry points over tiny shapes (OG3): every exported one, and
// packSignBits, with each buffer against the end guard and the start guard, under Sweep's
// deadline. TestTinyShapes_complete holds the list to bitvec's exported names.

var tinyLens = []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257}

// tinyShape is rows by cols (HammingRows: n rows by words).
type tinyShape struct{ rows, cols int }

func (s tinyShape) String() string { return fmt.Sprintf("rows=%d cols=%d", s.rows, s.cols) }

type tinyResult struct {
	Refused string
	Out     []any
}

func tinyAd(name string, shapes []tinyShape, call func(p *kernreg.Placer, s tinyShape) []any) kernreg.Adapter {
	return kernreg.Adapter{
		Symbol:  name,
		Lengths: shapeIndices(len(shapes)),
		Label:   func(i int) string { return shapes[i].String() },
		Call: func(p *kernreg.Placer, i int) any {
			var out []any
			msg := kernreg.Refused(func() { out = call(p, shapes[i]) })
			return tinyResult{msg, out}
		},
	}
}

func tinyN() []tinyShape {
	var out []tinyShape
	for _, n := range tinyLens {
		out = append(out, tinyShape{1, n})
	}
	return out
}

func tinyRC() []tinyShape {
	out := tinyN()
	for _, r := range tinyLens[:18] {
		out = append(out, tinyShape{r, 4})
	}
	return append(out, tinyShape{0, 0}, tinyShape{2, 3}, tinyShape{3, 1})
}

// tinyHamming crosses words over the kernels' 4-word blocks with row counts.
func tinyHamming() []tinyShape {
	var out []tinyShape
	for _, w := range []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 16, 17} {
		for _, n := range []int{0, 1, 2, 3, 4, 5, 8, 9, 17} {
			out = append(out, tinyShape{n, w})
		}
	}
	return out
}

func tinySeqF32(n, seed int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32((i*7+seed*13)%17 - 8)
	}
	return x
}

func tinyU64(n, seed int) []uint64 {
	x := make([]uint64, n)
	for i := range x {
		x[i] = uint64(i+1)*0x9E3779B97F4A7C15 ^ uint64(seed)*0xBF58476D1CE4E5B9
	}
	return x
}

func tinyAdapters() []kernreg.Adapter {
	return []kernreg.Adapter{
		tinyAd("HammingRows", tinyHamming(), func(p *kernreg.Placer, s tinyShape) []any {
			words, n := s.cols, s.rows
			dst := kernreg.Put(p, "dst", make([]uint16, n))
			q := kernreg.Put(p, "q", tinyU64(words, 1))
			HammingRows(q, kernreg.Put(p, "codes", tinyU64(n*words, 2)), words, n, dst)
			return []any{dst}
		}),
		tinyAd("packSignBits", tinyRC(), func(p *kernreg.Placer, s tinyShape) []any {
			dst := kernreg.Put(p, "dst", make([]uint64, s.rows*PackedWords(s.cols)))
			packSignBits(dst, kernreg.Put(p, "src", tinySeqF32(s.rows*s.cols, 1)), s.rows, s.cols)
			return []any{dst}
		}),
		tinyAd("PackSignBitsRow", tinyN(), func(p *kernreg.Placer, s tinyShape) []any {
			dst := kernreg.Put(p, "dst", make([]uint64, PackedWords(s.cols)))
			PackSignBitsRow(dst, kernreg.Put(p, "v", tinySeqF32(s.cols, 1)))
			return []any{dst}
		}),
		tinyAd("PackedWords", tinyN(), func(_ *kernreg.Placer, s tinyShape) []any { return []any{PackedWords(s.cols)} }),
	}
}

func TestTinyShapes_everyExportedEntryPoint(t *testing.T) {
	kernreg.Sweep(t, tinyAdapters())
}

func TestTinyShapes_complete(t *testing.T) {
	var swept []string
	for _, ad := range tinyAdapters() {
		if i := strings.LastIndexByte(ad.Symbol, '.'); token.IsExported(ad.Symbol[i+1:]) {
			swept = append(swept, ad.Symbol)
		}
	}
	probs, err := kernreg.ExportedProblems(".", swept, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range probs {
		t.Error(p)
	}
}
