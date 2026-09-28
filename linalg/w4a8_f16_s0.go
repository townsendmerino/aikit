package linalg

import (
	"fmt"
	"math"
)

// S0 PROTOTYPE — not an API. goinfer docs/tasks/task-cpu-decode-peer-gap-2026-09.md, L1 S0: does
// storing CPU int4 group scales as f16 (2 bytes per 32 weights instead of 4) speed up M=1 decode once
// the kernel pays a VCVTPH2PS per group? A WeightMat may carry an f16 copy of its scales
// (AttachInt4ScalesF16S0) and a W4A8Op a ScalesF16 slice; SetInt4F16ScalesS0 selects them at run time,
// so one loaded model runs both arms of a paired A/B. amd64 M<4 decode only; everything else keeps the
// f32 path. Given f16-representable scales both arms are bit-identical (f16→f32 is exact).

var int4F16Scales bool

// f16S0BufGroups bounds the stack buffer a row's widened scales use (K up to 32768).
const f16S0BufGroups = 1024

// SetInt4F16ScalesS0 makes the M=1 W4A8 paths read f16 scales where a weight carries them.
func SetInt4F16ScalesS0(on bool) { int4F16Scales = on }

// Int4F16ScalesS0 reports SetInt4F16ScalesS0's setting.
func Int4F16ScalesS0() bool { return int4F16Scales }

// AttachInt4ScalesF16S0 stores an f16 copy of an int4 WeightMat's canonical scales. Every scale must
// already be f16-representable (goinfer loads with GOINFER_INT4_F16_SCALES=1); it panics otherwise,
// since a rounding here would make the A/B's arms differ numerically. Returns false for a non-int4 or
// repacked-only weight.
func (w *WeightMat) AttachInt4ScalesF16S0() bool {
	if w.q4 == nil || w.q4s == nil {
		return false
	}
	h := make([]uint16, len(w.q4s))
	for i, s := range w.q4s {
		h[i] = f32ToF16Exact(s)
	}
	w.q4sF16 = h
	return true
}

// Int4ScalesF16S0 is the f16 copy AttachInt4ScalesF16S0 stored, or nil.
func (w *WeightMat) Int4ScalesF16S0() []uint16 { return w.q4sF16 }

// f32ToF16Exact encodes s as IEEE half, panicking unless the encoding is exact.
func f32ToF16Exact(s float32) uint16 {
	b := math.Float32bits(s)
	sign := uint16(b>>16) & 0x8000
	exp := int32(b>>23) & 0xff
	mant := b & 0x7fffff
	var h uint16
	switch {
	case exp == 0 && mant == 0:
		h = sign
	case exp >= 113 && exp <= 142: // normal f16
		h = sign | uint16(exp-112)<<10 | uint16(mant>>13)
	case exp >= 103 && exp < 113: // subnormal f16: value = m10 · 2^-24
		m24 := mant | 0x800000
		h = sign | uint16(m24>>uint(126-exp))
	default:
		panic(fmt.Sprintf("linalg: f32ToF16Exact(%g): outside half precision's range", s))
	}
	if f16ToF32(h) != s {
		panic(fmt.Sprintf("linalg: f32ToF16Exact(%g): not exactly representable as f16", s))
	}
	return h
}

// dotW4A8F16Go widens the scales and calls dotW4A8, so it is bit-identical to the f32 path.
func dotW4A8F16Go(act []int8, packed []byte, scales []uint16, group, K int) float32 {
	f := make([]float32, len(scales))
	for i, h := range scales {
		f[i] = f16ToF32(h)
	}
	return dotW4A8(act, packed, f, group, K)
}

// MatmulBTW4A8F16Into is MatmulBTW4A8Into over f16 scales, for M < 4 (decode) and per-row activations
// only — the S0's scope. It mirrors MatmulBTW4A8Into's quantization and serial/parallel split exactly.
func MatmulBTW4A8F16Into(ws *Workspace, a []float32, w4 []byte, wScales16 []uint16, dst []float32, M, K, N, group int) {
	if M >= 4 {
		panic("linalg: MatmulBTW4A8F16Into: S0 prototype serves M < 4 only")
	}
	if g := actGroupFor(ws); g > 0 {
		panic("linalg: MatmulBTW4A8F16Into: S0 prototype serves per-row activations only")
	}
	nGroups, bpr := groupsFor(K, group)
	if len(wScales16) != N*nGroups || len(w4) != N*bpr || len(a) < M*K || len(dst) < M*N {
		panic(fmt.Sprintf("linalg: MatmulBTW4A8F16Into: shape mismatch (w4 %d, scales %d, a %d, dst %d; M=%d K=%d N=%d)",
			len(w4), len(wScales16), len(a), len(dst), M, K, N))
	}
	aq := ws.int8Buf(M * K)
	aScales := ws.f32Buf(M)
	quantizeRowsInto(aq, aScales, a, M, K, ws.width)
	if M*N*K < ws.thr() || N < 2 {
		w4a8SpanRowsF16(aq, aScales, w4, wScales16, dst, K, N, group, nGroups, bpr, 0, M, 0, N)
		return
	}
	ws.parallel(N, func(j0, j1 int) {
		w4a8SpanRowsF16(aq, aScales, w4, wScales16, dst, K, N, group, nGroups, bpr, 0, M, j0, j1)
	})
}

// w4a8SpanRowsF16 is w4a8SpanRows over f16 scales. By default each weight row's scales are widened once
// into buf (eight per instruction, L1-resident) and the unchanged f32 kernel runs on them.
func w4a8SpanRowsF16(aq []int8, aScales []float32, w4 []byte, wScales16 []uint16, dst []float32, K, N, group, nGroups, bpr, i0, i1, j0, j1 int) {
	if !f16S0PerGroup && nGroups <= f16S0BufGroups {
		var buf [f16S0BufGroups]float32
		for j := j0; j < j1; j++ {
			prow := w4[j*bpr : j*bpr+bpr]
			srow := wScales16[j*nGroups : j*nGroups+nGroups]
			for i := i0; i < i1; i++ {
				if aScales[i] == 0 {
					dst[i*N+j] = 0
					continue
				}
				dst[i*N+j] = dotW4A8F16Row(aq[i*K:i*K+K], prow, srow, group, K, buf[:]) * aScales[i]
			}
		}
		return
	}
	for j := j0; j < j1; j++ {
		prow := w4[j*bpr : j*bpr+bpr]
		srow := wScales16[j*nGroups : j*nGroups+nGroups]
		for i := i0; i < i1; i++ {
			if aScales[i] == 0 {
				dst[i*N+j] = 0
				continue
			}
			dst[i*N+j] = dotW4A8F16(aq[i*K:i*K+K], prow, srow, group, K) * aScales[i]
		}
	}
}
