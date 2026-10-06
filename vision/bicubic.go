package vision

import "math"

// The reference resize of Gemma 4's image processor (and of every HF processor on the torchvision backend that resizes
// with BICUBIC and antialias=True): torchvision's antialiased bicubic on the uint8 image, as its uint8 CPU path computes
// it. Separable, horizontal pass first, each pass skipped when its size is unchanged; Keys weights with a = -0.5 over a
// support widened by the scale on downscale, normalised per output pixel, quantised to int16 at the precision the
// pass's largest weight allows, accumulated in integers with a rounding offset, clamped to uint8 after each pass.
// Bit-identical to torchvision 0.29.1 on goinfer's pinned cases and real images (goinfer's
// docs/tasks/task-embeddinggemma2.md, the Phase V resize; moved here from goinfer's embeddinggemma2 on 2026-10-06).

// bicubicFloatForTest (tests only) runs the resize in float64 with one rounding at the end instead of torchvision's
// fixed-point passes: the control that shows a comparison sees a rounding-level change.
var bicubicFloatForTest = false

// ResizeBicubicAA resizes packed 8-bit RGB (HWC) from h x w to th x tw with torchvision's antialiased bicubic on uint8.
func ResizeBicubicAA(src []uint8, h, w, th, tw int) []uint8 {
	return resizeBicubicRGB(src, h, w, th, tw)
}

// aaWeights are one axis's antialiased bicubic taps: for output i, inputs [start[i], start[i]+len(w[i])).
type aaWeights struct {
	start []int
	w     [][]float64
}

// bicubicAA is the Keys cubic with a = -0.5, torchvision's antialiased bicubic filter.
func bicubicAA(x float64) float64 {
	const a = -0.5
	x = math.Abs(x)
	if x < 1 {
		return ((a+2)*x-(a+3))*x*x + 1
	}
	if x < 2 {
		return (((x-5)*x+8)*x - 4) * a
	}
	return 0
}

// aaTaps computes torchvision's antialiased taps from in samples to out (align_corners false).
func aaTaps(in, out int) aaWeights {
	scale := float64(in) / float64(out)
	support, invscale := 2.0, 1.0
	if scale >= 1 {
		support, invscale = 2*scale, 1/scale
	}
	t := aaWeights{start: make([]int, out), w: make([][]float64, out)}
	for i := range out {
		center := scale * (float64(i) + 0.5)
		xmin := max(int(center-support+0.5), 0)
		xmax := min(int(center+support+0.5), in)
		w := make([]float64, xmax-xmin)
		var total float64
		for j := range w {
			w[j] = bicubicAA((float64(j+xmin) - center + 0.5) * invscale)
			total += w[j]
		}
		if total != 0 {
			for j := range w {
				w[j] /= total
			}
		}
		t.start[i], t.w[i] = xmin, w
	}
	return t
}

// resizeBicubicRGB resizes HWC RGB from (h, w) to (th, tw), the horizontal pass first, each pass skipped when its
// size does not change.
func resizeBicubicRGB(src []uint8, h, w, th, tw int) []uint8 {
	cur, cw := src, w
	if tw != w {
		cur = resizePass(cur, h, w, tw, true)
		cw = tw
	}
	if th != h {
		cur = resizePass(cur, h, cw, th, false)
	}
	return cur
}

// resizePass resizes along one axis: horizontal (w -> n) or vertical (h -> n), in torchvision's uint8 fixed point
// (or in float64 under bicubicFloatForTest).
func resizePass(src []uint8, h, w, n int, horizontal bool) []uint8 {
	in := w
	if !horizontal {
		in = h
	}
	taps := aaTaps(in, n)
	oh, ow := h, n
	if !horizontal {
		oh, ow = n, w
	}
	dst := make([]uint8, oh*ow*3)
	// torchvision's precision: the largest that keeps the largest weight under 2^15 at one more bit.
	var maxW float64
	for _, ws := range taps.w {
		for _, x := range ws {
			maxW = max(maxW, x)
		}
	}
	prec := uint(0)
	for prec = 0; prec < 22; prec++ {
		if int(0.5+maxW*float64(int(1)<<(prec+1))) >= 1<<15 {
			break
		}
	}
	iw := make([][]int32, len(taps.w))
	for i, ws := range taps.w {
		iw[i] = make([]int32, len(ws))
		for j, x := range ws {
			r := 0.5
			if x < 0 {
				r = -0.5
			}
			iw[i][j] = int32(int16(r + x*float64(int(1)<<prec)))
		}
	}
	at := func(y, x, c int) int { return (y*w+x)*3 + c }
	for oy := range oh {
		for ox := range ow {
			i := ox
			if !horizontal {
				i = oy
			}
			s := taps.start[i]
			for c := range 3 {
				if bicubicFloatForTest {
					var acc float64
					for j, wt := range taps.w[i] {
						if horizontal {
							acc += wt * float64(src[at(oy, s+j, c)])
						} else {
							acc += wt * float64(src[at(s+j, ox, c)])
						}
					}
					dst[(oy*ow+ox)*3+c] = uint8(math.Max(0, math.Min(255, math.Round(acc))))
					continue
				}
				acc := int32(1) << (prec - 1)
				for j, wt := range iw[i] {
					if horizontal {
						acc += wt * int32(src[at(oy, s+j, c)])
					} else {
						acc += wt * int32(src[at(s+j, ox, c)])
					}
				}
				v := acc >> prec
				dst[(oy*ow+ox)*3+c] = uint8(max(0, min(255, v)))
			}
		}
	}
	return dst
}
