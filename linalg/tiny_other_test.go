package linalg

import (
	"cmp"
	"fmt"
	"maps"
	"slices"

	"github.com/townsendmerino/aikit/internal/kernreg"
)

// toShape is one call's dimensions: R rows (M, N or n), C columns (K, cols or n) and G
// the group size (or a third count); a field an entry point does not take stays 0.
type toShape struct{ R, C, G int }

func (s toShape) String() string { return fmt.Sprintf("r=%d c=%d g=%d", s.R, s.C, s.G) }

// toResult is what a tiny-shape call produced: the refusal text, or the outputs.
type toResult struct {
	Refused string
	Out     []any
}

// toAd builds the adapter for one exported name over shapes; call places its inputs and
// outputs with Put, calls the entry point, and returns the outputs.
func toAd(name string, shapes []toShape, call func(p *kernreg.Placer, s toShape) []any) kernreg.Adapter {
	return kernreg.Adapter{
		Symbol:  name,
		Lengths: shapeIndices(len(shapes)),
		Label:   shapeLabels(shapes),
		Call: func(p *kernreg.Placer, i int) any {
			var out []any
			msg := kernreg.Refused(func() { out = call(p, shapes[i]) })
			return toResult{msg, out}
		},
	}
}

// toN is one length over tinyDims and the K-quant block size's neighbours.
func toN() []toShape {
	var out []toShape
	for _, n := range slices.Concat(tinyDims, []int{127, 128, 129, 255, 256, 257}) {
		out = append(out, toShape{C: n})
	}
	return out
}

// toRC varies cols over tinyDims at one row, rows over tinyDims at 4 columns, and the
// corners.
func toRC() []toShape {
	var out []toShape
	for _, c := range tinyDims {
		out = append(out, toShape{R: 1, C: c})
	}
	for _, r := range tinyDims {
		out = append(out, toShape{R: r, C: 4})
	}
	return append(out, toShape{0, 0, 0}, toShape{2, 3, 0}, toShape{3, 1, 0})
}

// toRCG crosses cols over tinyDims with every group in tinyGroups at one row, then rows
// over tinyDims at cols = group = 32, and the corners.
func toRCG() []toShape {
	var out []toShape
	for _, c := range tinyDims {
		for _, g := range tinyGroups {
			out = append(out, toShape{1, c, g})
		}
	}
	for _, r := range tinyDims {
		out = append(out, toShape{r, 32, 32})
	}
	return append(out, toShape{0, 0, 0}, toShape{0, 0, 32}, toShape{2, 64, 32}, toShape{4, 64, 32}, toShape{5, 96, 32})
}

// toQK is the K-quant shapes: K over 0, 1 and the 256 block's neighbours, at rows 0..2.
func toQK() []toShape {
	var out []toShape
	for _, r := range []int{0, 1, 2} {
		for _, k := range []int{0, 1, 255, 256, 257, 512} {
			out = append(out, toShape{R: r, C: k})
		}
	}
	return out
}

// toCeil is ⌈a/b⌉, and 0 for b ≤ 0 (no group size, so no groups to size a buffer for).
func toCeil(a, b int) int {
	if b <= 0 {
		return 0
	}
	return (a + b - 1) / b
}

// toWM reads a WeightMat back through its public accessors: its size, kind, and every row
// dequantized.
func toWM(w WeightMat) []any {
	out := []any{w.Rows(), w.Cols(), w.Kind()}
	for i := range w.Rows() {
		dst := make([]float32, w.Cols())
		w.Row(i, dst)
		out = append(out, dst)
	}
	return out
}

// toInt4 is a canonical int4 matrix for shape s, made on the heap by QuantizeGroupsInt4;
// at group 0, which QuantizeGroupsInt4 refuses, it is made at group 32, so the entry point
// under test is the one that meets the group of 0.
func toInt4(s toShape) ([]byte, []float32) {
	return QuantizeGroupsInt4(seqF32(s.R*s.C, 3), s.R, s.C, cmp.Or(s.G, 32))
}

func tinyOtherAdapters() []kernreg.Adapter {
	// The element-wise kernels, the binary-vector kernels and the attention scratch are
	// swept by their own packages' tiny_test.go (linalg/elem, linalg/bitvec, linalg/attn).
	var ads []kernreg.Adapter
	n4 := []toShape{}
	for _, n := range []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17} {
		n4 = append(n4, toShape{C: n})
	}
	ads = append(ads,
		toAd("Dot", toN(), func(p *kernreg.Placer, s toShape) []any {
			return []any{Dot(kernreg.Put(p, "a", seqF32(s.C, 1)), kernreg.Put(p, "b", seqF32(s.C, 2)))}
		}),
		toAd("DotI8", toN(), func(p *kernreg.Placer, s toShape) []any {
			return []any{DotI8(kernreg.Put(p, "a", seqI8(s.C, 1)), kernreg.Put(p, "b", seqI8(s.C, 2)))}
		}),
		toAd("DotI8x8", toN(), func(p *kernreg.Placer, s toShape) []any {
			var b [8][]int8
			for r := range b {
				b[r] = kernreg.Put(p, fmt.Sprint("b", r), seqI8(s.C, r+2))
			}
			return []any{DotI8x8(kernreg.Put(p, "a", seqI8(s.C, 1)), b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7])}
		}),
		toAd("dot4x4", n4, func(p *kernreg.Placer, s toShape) []any {
			n := 4 * s.C
			var b [4][]float32
			for r := range b {
				b[r] = kernreg.Put(p, fmt.Sprint("b", r), seqF32(n, r+2))
			}
			sums := kernreg.Put(p, "sums", make([]float32, 16))
			dot4x4(ptr(kernreg.Put(p, "a", seqF32(n, 1))), ptr(b[0]), ptr(b[1]), ptr(b[2]), ptr(b[3]), s.C, (*[16]float32)(sums))
			return []any{toLaneSums(sums)}
		}),
		toAd("Dot8x4", n4, func(p *kernreg.Placer, s toShape) []any {
			n := 4 * s.C
			var b [8][]float32
			for r := range b {
				b[r] = kernreg.Put(p, fmt.Sprint("b", r), seqF32(n, r+2))
			}
			sums := kernreg.Put(p, "sums", make([]float32, 32))
			Dot8x4(ptr(kernreg.Put(p, "a", seqF32(n, 1))), ptr(b[0]), ptr(b[1]), ptr(b[2]), ptr(b[3]), ptr(b[4]), ptr(b[5]), ptr(b[6]), ptr(b[7]), s.C, (*[32]float32)(sums))
			return []any{toLaneSums(sums)}
		}),
		toAd("Dot2x8", n4, func(p *kernreg.Placer, s toShape) []any {
			n := 4 * s.C
			var b [8][]float32
			for r := range b {
				b[r] = kernreg.Put(p, fmt.Sprint("b", r), seqF32(n, r+3))
			}
			sums := kernreg.Put(p, "sums", make([]float32, 64))
			Dot2x8(ptr(kernreg.Put(p, "a0", seqF32(n, 1))), ptr(kernreg.Put(p, "a1", seqF32(n, 2))), ptr(b[0]), ptr(b[1]), ptr(b[2]), ptr(b[3]), ptr(b[4]), ptr(b[5]), ptr(b[6]), ptr(b[7]), s.C, (*[64]float32)(sums))
			return []any{toLaneSums(sums)}
		}),
		toAd("F16ToF32Slice", toN(), func(p *kernreg.Placer, s toShape) []any {
			dst := kernreg.Put(p, "dst", make([]float32, s.C))
			F16ToF32Slice(dst, kernreg.Put(p, "src", F32ToF16Scales(seqF32(s.C, 1))))
			return []any{dst}
		}),
		toAd("F32ToF16Slice", toN(), func(p *kernreg.Placer, s toShape) []any {
			dst := kernreg.Put(p, "dst", make([]uint16, s.C))
			F32ToF16Slice(dst, kernreg.Put(p, "src", seqF32(s.C, 1)))
			return []any{dst}
		}),
		toAd("F32ToF16Scales", toN(), func(p *kernreg.Placer, s toShape) []any {
			return []any{F32ToF16Scales(kernreg.Put(p, "src", seqF32(s.C, 1)))}
		}),
		toAd("DequantizeRowInt8", toN(), func(p *kernreg.Placer, s toShape) []any {
			dst := kernreg.Put(p, "dst", make([]float32, s.C))
			DequantizeRowInt8(kernreg.Put(p, "q", seqI8(s.C, 1)), 0.5, dst)
			return []any{dst}
		}),
		toAd("DequantizeRowsInt8Into", toRC(), func(p *kernreg.Placer, s toShape) []any {
			dst := kernreg.Put(p, "dst", make([]float32, s.R*s.C))
			DequantizeRowsInt8Into(dst, kernreg.Put(p, "q", seqI8(s.R*s.C, 1)), kernreg.Put(p, "scales", seqF32(s.R, 2)), s.R, s.C)
			return []any{dst}
		}),
		toAd("DequantizeRowInt4", toRCG(), func(p *kernreg.Placer, s toShape) []any {
			dst := kernreg.Put(p, "dst", make([]float32, s.C))
			DequantizeRowInt4(kernreg.Put(p, "packed", seqU8((s.C+1)/2, 1)), kernreg.Put(p, "scales", seqF32(toCeil(s.C, s.G), 2)), s.G, s.C, dst)
			return []any{dst}
		}),
		toAd("QuantizeRowInt8", toN(), func(p *kernreg.Placer, s toShape) []any {
			q := kernreg.Put(p, "q", make([]int8, s.C))
			sc := QuantizeRowInt8(kernreg.Put(p, "row", seqF32(s.C, 1)), q)
			return []any{sc, q}
		}),
		toAd("QuantizeRowsInt8", toRC(), func(p *kernreg.Placer, s toShape) []any {
			q, sc := QuantizeRowsInt8(kernreg.Put(p, "w", seqF32(s.R*s.C, 1)), s.R, s.C)
			return []any{q, sc}
		}),
		toAd("quantizeActivationsInto", toRC(), func(p *kernreg.Placer, s toShape) []any {
			aq := kernreg.Put(p, "aq", make([]int8, s.R*s.C))
			sc := kernreg.Put(p, "scales", make([]float32, s.R))
			quantizeActivationsInto(aq, sc, kernreg.Put(p, "a", seqF32(s.R*s.C, 1)), s.R, s.C)
			return []any{aq, sc}
		}),
		toAd("QuantizeActivationsGroupedInto", toRCG(), func(p *kernreg.Placer, s toShape) []any {
			aq := kernreg.Put(p, "aq", make([]int8, s.R*s.C))
			sc := kernreg.Put(p, "scales", make([]float32, s.R*toCeil(s.C, s.G)))
			QuantizeActivationsGroupedInto(aq, sc, kernreg.Put(p, "a", seqF32(s.R*s.C, 1)), s.R, s.C, s.G)
			return []any{aq, sc}
		}),
		toAd("sumActGroupsInto", toRCG(), func(p *kernreg.Placer, s toShape) []any {
			sum := kernreg.Put(p, "sumAct", make([]int32, s.R*toCeil(s.C, s.G)))
			sumActGroupsInto(sum, kernreg.Put(p, "aq", seqI8(s.R*s.C, 1)), s.R, s.C, s.G)
			return []any{sum}
		}),
		toAd("QuantizeActQ", toRCG(), func(p *kernreg.Placer, s toShape) []any {
			var q ActQ
			QuantizeActQ(kernreg.Put(p, "a", seqF32(s.R*s.C, 1)), s.R, s.C, s.G, &q)
			return []any{q.Q, q.S, q.M, q.K, q.Group}
		}),
		toAd("quantizeActQ8K", toQK(), func(p *kernreg.Placer, s toShape) []any {
			q := quantizeActQ8K(kernreg.Put(p, "a", seqF32(s.R*s.C, 1)), s.R, s.C)
			return []any{q}
		}),
		toAd("QuantizeGroupInt4Row", toRCG(), func(p *kernreg.Placer, s toShape) []any {
			packed := kernreg.Put(p, "packed", make([]byte, (s.C+1)/2))
			sc := kernreg.Put(p, "scales", make([]float32, toCeil(s.C, s.G)))
			QuantizeGroupInt4Row(kernreg.Put(p, "row", seqF32(s.C, 1)), s.C, s.G, packed, sc)
			return []any{packed, sc}
		}),
		toAd("QuantizeGroupsInt4", toRCG(), func(p *kernreg.Placer, s toShape) []any {
			packed, sc := QuantizeGroupsInt4(kernreg.Put(p, "w", seqF32(s.R*s.C, 1)), s.R, s.C, s.G)
			return []any{packed, sc}
		}),
		toAd("QuantizeInt4", toRCG(), func(p *kernreg.Placer, s toShape) []any {
			return toWM(QuantizeInt4(kernreg.Put(p, "w", seqF32(s.R*s.C, 1)), s.R, s.C, s.G))
		}),
		toAd("QuantizeInt8", toRC(), func(p *kernreg.Placer, s toShape) []any {
			a := toWM(QuantizeInt8(kernreg.Put(p, "w", seqF32(s.R*s.C, 1)), s.R, s.C, false))
			b := toWM(QuantizeInt8(kernreg.Put(p, "w8a8", seqF32(s.R*s.C, 1)), s.R, s.C, true))
			return append(a, b...)
		}),
		toAd("QuantizeBF16", toRC(), func(p *kernreg.Placer, s toShape) []any {
			return toWM(QuantizeBF16(kernreg.Put(p, "w", seqF32(s.R*s.C, 1)), s.R, s.C))
		}),
		toAd("QuantizeF16", toRC(), func(p *kernreg.Placer, s toShape) []any {
			return toWM(QuantizeF16(kernreg.Put(p, "w", seqF32(s.R*s.C, 1)), s.R, s.C))
		}),
		toAd("RepackInt4SplitHalfRow", toN(), func(p *kernreg.Placer, s toShape) []any {
			dst := kernreg.Put(p, "dst", make([]byte, s.C/2))
			RepackInt4SplitHalfRow(dst, kernreg.Put(p, "src", seqU8(s.C/2, 1)), s.C)
			return []any{dst}
		}),
		toAd("RepackW4A8SplitHalf", toRCG(), func(p *kernreg.Placer, s toShape) []any {
			packed, _ := toInt4(s)
			return []any{RepackW4A8SplitHalf(kernreg.Put(p, "packed", packed), s.R, s.C, s.G)}
		}),
		// Both repacked layouts through the one in-place repack; a layout this CPU does not
		// run is declined with its input untouched, which toRepacked records.
		toAd("RepackInt4InPlace", toRCG(), func(p *kernreg.Placer, s toShape) []any {
			var out []any
			for _, l := range []Layout{LayoutCanonical, LayoutRow4, LayoutSplitHalf} {
				q4, q4s := toInt4(s)
				q4, q4s16 := kernreg.Put(p, "q4", q4), kernreg.Put(p, "q4s", F32ToF16Scales(q4s))
				w, ok := RepackInt4InPlace(q4, q4s16, s.R, s.C, s.G, l)
				out = append(out, toRepacked(w, ok, q4, q4s16)...)
			}
			return out
		}),
		toAd("Q4KRowBytes", toQK(), func(_ *kernreg.Placer, s toShape) []any { return []any{Q4KRowBytes(s.C)} }),
		toAd("Q5KRowBytes", toQK(), func(_ *kernreg.Placer, s toShape) []any { return []any{Q5KRowBytes(s.C)} }),
		toAd("Q6KRowBytes", toQK(), func(_ *kernreg.Placer, s toShape) []any { return []any{Q6KRowBytes(s.C)} }),
		toAd("PreferredInt4Layout", toRCG(), func(_ *kernreg.Placer, s toShape) []any {
			return []any{PreferredInt4Layout(s.R, s.C, s.G)}
		}),
	)
	return append(ads, toArchAdapters()...)
}

// toLaneSums folds each 4-lane block of a Dot*x* result: the per-lane layout within a
// block is arch-specific by contract, the block's sum is not.
func toLaneSums(sums []float32) []float32 {
	out := make([]float32, len(sums)/4)
	for i := range out {
		out[i] = sums[4*i] + sums[4*i+1] + sums[4*i+2] + sums[4*i+3]
	}
	return out
}

// toRepacked reads back an in-place repack: ok, the result (when ok), and the input
// buffers, which a declined repack must leave untouched.
func toRepacked(w WeightMat, ok bool, in ...any) []any {
	out := append([]any{ok}, in...)
	if ok {
		out = append(out, toWM(w)...)
	}
	return out
}

var tinyOtherExempt = func() map[string]string {
	scalar := "a scalar function of one value; no shape"
	config := "process-wide configuration; no shape (and setting it in a test races every other test)"
	report := "a report of this CPU's kernels or self-check; no shape"
	m := map[string]string{
		"BF16ToF32": scalar, "F16ToF32": scalar, "F32ToBF16": scalar, "F32ToF16": scalar,
		"ParallelThreshold": config, "ParallelWidth": config, "W4A8RowFoldForTest": config,
		"SetActQuantGroup": config, "SetParallelThreshold": config,
		"SetParallelWidth": config, "SetW4A8RowFoldForTest": config,
		"Kind.String": scalar, "Layout.String": scalar,
		"ActiveKernels": report, "ForcedFallbacks": report, "KQuantNative": report,
		"SelfCheck": report, "SelfCheckReport.OK": report, "Mismatch.String": report,
	}
	maps.Copy(m, toArchExempt)
	return m
}()
