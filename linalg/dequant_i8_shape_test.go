package linalg

import (
	"strings"
	"testing"
)

// The dst-length guard of DequantizeRowInt8 and WeightMat.Row holds in the production build. The tag-gated TestChecks_fire
// covers the same call only under -tags aikit_checks, which is the build that was never shipped.
func TestDequantizeRowInt8_shortDstPanicsInEveryBuild(t *testing.T) {
	for _, n := range []int{1, 7, 8, 16, 64, 100} {
		q := make([]int8, n)
		for _, short := range []int{0, n - 1} {
			if short < 0 {
				continue
			}
			func() {
				defer func() {
					r := recover()
					msg, _ := r.(string)
					if r == nil || !strings.Contains(msg, "DequantizeRowInt8") {
						t.Errorf("q len %d, dst len %d: want a DequantizeRowInt8 panic, got %v", n, short, r)
					}
				}()
				DequantizeRowInt8(q, 1, make([]float32, short))
			}()
		}
		DequantizeRowInt8(q, 1, make([]float32, n)) // exact and oversized dst are fine
		DequantizeRowInt8(q, 1, make([]float32, n+9))
	}
}

func TestWeightMatRow_shortDstPanicsForEveryKind(t *testing.T) {
	const rows, cols = 4, 64
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(i%13) - 6
	}
	for name, wm := range map[string]WeightMat{
		"int8": QuantizeInt8(w, rows, cols, false),
		"int4": QuantizeInt4(w, rows, cols, 32),
		"f32":  WrapF32(w, rows, cols),
	} {
		func() {
			defer func() {
				r := recover()
				msg, _ := r.(string)
				if r == nil || !strings.Contains(msg, "WeightMat.Row") {
					t.Errorf("%s: want a WeightMat.Row panic for dst shorter than cols, got %v", name, r)
				}
			}()
			wm.Row(1, make([]float32, cols-1))
		}()
		wm.Row(1, make([]float32, cols)) // exact is fine
	}
}
