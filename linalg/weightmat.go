package linalg

import (
	"fmt"
	"unsafe"

	"github.com/townsendmerino/aikit/mmap"
)

// WeightMat is a [rows, cols] = [out, in] weight matrix that hides its storage
// precision behind a uniform matmul + dequant surface. It consolidates three
// open-coded wrappers that each held the same thing — an f32 / int8 / int4 weight
// plus its scales plus a precision-dispatched MatmulBT:
//
//   - aikit encoder.LayerWeightsQ8 — per-row int8 projection weights (storage only;
//     the encoder keeps its own baked-scale blocked matmul for large-M prefill, fed
//     from Int8()/Scales(), since that path is numerically distinct from MatmulBTQ8).
//   - goinfer decoder.weightMat — the richest: f32 / per-row int8 / group int4 / W8A8,
//     with the matmul dispatch and tied-embedding Row lookup.
//   - goinfer vision.qmat — f32 / W8A8 for the SigLIP tower.
//
// Experimental tier. It hides STORAGE only — model policy stays with the consumer:
// which precision a table gets (e.g. goinfer keeping logit-critical embeddings at
// int8 in an int4 model), the int4 group size, and any GPU-backend dispatch (a
// consumer routes to its accelerator via the raw accessors, falling back to
// MatmulBT for CPU). Dispatch reuses the existing kernels — no new asm; outputs are
// bit-identical to each consumer's prior kernel call.
type WeightMat struct {
	f32    []float32 // non-nil ⇒ f32 path
	q8     []int8    // non-nil ⇒ per-row int8 (weight-only Q8, or W8A8 if w8a8)
	scales []float32 // [rows] per-row int8 scales
	q4     []byte    // non-nil ⇒ group-wise int4 packed nibbles
	q4s    []float32 // [rows*nGroups] per-group int4 scales
	group  int       // int4 group size (0 unless int4)
	w8a8   bool      // int8 weights run full int8×int8 (W8A8) instead of weight-only Q8
	rows   int       // out features (N)
	cols   int       // in features (K)
}

// WrapF32 wraps an existing [rows, cols] f32 weight WITHOUT copying — the WeightMat
// aliases w (the caller keeps it alive, e.g. an mmap'd tensor). A consumer that must
// release the source after construction should pass a copy.
//
// Panics if rows or cols is negative or len(w) != rows*cols (checked
// overflow-safe, so a wrapped-int shape can't slip a short buffer past).
func WrapF32(w []float32, rows, cols int) WeightMat {
	if rows < 0 || cols < 0 {
		panic(fmt.Sprintf("linalg: WrapF32 negative dim (rows=%d cols=%d)", rows, cols))
	}
	requireExactLen("WrapF32", "w", len(w), mul(rows, cols))
	return WeightMat{f32: w, rows: rows, cols: cols}
}

// QuantizeInt8 quantizes a [rows, cols] f32 weight to per-row symmetric int8 (¼ f32),
// not retaining the source (it can be released). w8a8 selects the matmul: false ⇒
// weight-only Q8 (dequant-then-f32, lossless activations), true ⇒ full int8×int8 W8A8.
func QuantizeInt8(w []float32, rows, cols int, w8a8 bool) WeightMat {
	q, s := QuantizeRowsInt8(w, rows, cols)
	return WeightMat{q8: q, scales: s, w8a8: w8a8, rows: rows, cols: cols}
}

// QuantizeInt4 quantizes a [rows, cols] f32 weight to group-wise symmetric int4
// (~⅛ f32; group consecutive input features share one scale), not retaining the source.
func QuantizeInt4(w []float32, rows, cols, group int) WeightMat {
	q4, q4s := QuantizeGroupsInt4(w, rows, cols, group)
	return WeightMat{q4: q4, q4s: q4s, group: group, rows: rows, cols: cols}
}

// WrapInt8 wraps ALREADY-quantized per-row int8 weights (q8 [rows*cols] + per-row
// scales [rows]) WITHOUT copying or re-quantizing — the inverse of Int8(). Like
// WrapF32 it aliases the caller's slices (which may point into an mmap'd blob), so
// the caller keeps them alive. w8a8 selects the matmul (false ⇒ weight-only Q8,
// true ⇒ full int8×int8). For a loader that reads pre-quantized weights straight
// off disk and must not pay a dequant→requantize round-trip.
// Panics if rows or cols is negative, len(q8) != rows*cols, or
// len(scales) != rows (all checked overflow-safe).
func WrapInt8(q8 []int8, scales []float32, rows, cols int, w8a8 bool) WeightMat {
	if rows < 0 || cols < 0 {
		panic(fmt.Sprintf("linalg: WrapInt8 negative dim (rows=%d cols=%d)", rows, cols))
	}
	requireExactLen("WrapInt8", "q8", len(q8), mul(rows, cols))
	requireExactLen("WrapInt8", "scales", len(scales), rows)
	return WeightMat{q8: q8, scales: scales, w8a8: w8a8, rows: rows, cols: cols}
}

// WrapInt4 wraps ALREADY-quantized group-wise int4 weights WITHOUT copying or
// re-quantizing — the inverse of Int4(). q4 is [rows*((cols+1)/2)] packed nibbles
// (two per byte, row-major) and q4s is [rows*nGroups] per-group scales, where
// nGroups = ⌈cols/group⌉. Aliases the caller's slices (e.g. a zero-copy mmap of a
// quantized checkpoint), so the caller keeps them alive.
// Panics if rows or cols is negative, group <= 0, len(q4) != rows*⌈cols/2⌉, or
// len(q4s) != rows*⌈cols/group⌉ (all checked overflow-safe).
func WrapInt4(q4 []byte, q4s []float32, rows, cols, group int) WeightMat {
	if rows < 0 || cols < 0 {
		panic(fmt.Sprintf("linalg: WrapInt4 negative dim (rows=%d cols=%d)", rows, cols))
	}
	if group <= 0 {
		panic(fmt.Sprintf("linalg: WrapInt4 needs group > 0, got %d", group))
	}
	nGroups, bpr := groupsFor(cols, group)
	requireExactLen("WrapInt4", "q4", len(q4), mul(rows, bpr))
	requireExactLen("WrapInt4", "q4s", len(q4s), mul(rows, nGroups))
	return WeightMat{q4: q4, q4s: q4s, group: group, rows: rows, cols: cols}
}

// MatmulBT computes dst[M, rows] = a[M, cols] · weight[rows, cols]ᵀ, dispatching by
// stored precision to the matching linalg kernel. CPU only — a consumer with a GPU
// backend dispatches via the raw accessors and uses this as the fallback.
func (w *WeightMat) MatmulBT(a, dst []float32, M int) {
	switch {
	case w.q4 != nil:
		MatmulBTW4A8(a, w.q4, w.q4s, dst, M, w.cols, w.rows, w.group)
	case w.q8 != nil && w.w8a8:
		MatmulBTW8A8(a, w.q8, w.scales, dst, M, w.cols, w.rows)
	case w.q8 != nil:
		MatmulBTQ8(a, w.q8, w.scales, dst, M, w.cols, w.rows)
	default:
		MatmulBT(a, w.f32, dst, M, w.cols, w.rows)
	}
}

// MatmulBTInto is MatmulBT through a Workspace: the W8A8 path quantizes the activation
// once into the Workspace's reusable scratch (zero per-call alloc — the steady-state
// decode win), and the f32 path runs the Workspace-scoped parallel matmul honoring its
// SetThreshold/SetWorkers. The weight-only Q8 and int4 paths have no Workspace variant
// and ignore ws.
func (w *WeightMat) MatmulBTInto(ws *Workspace, a, dst []float32, M int) {
	// Case order kept identical to MatmulBT above: with constructor-built values
	// the storage kinds are mutually exclusive so order is inert, but a
	// hand-built WeightMat with more than one set would otherwise route the two
	// entry points to different kernels.
	switch {
	case w.q4 != nil:
		MatmulBTW4A8(a, w.q4, w.q4s, dst, M, w.cols, w.rows, w.group)
	case w.q8 != nil && w.w8a8:
		MatmulBTW8A8Into(ws, a, w.q8, w.scales, dst, M, w.cols, w.rows)
	case w.q8 != nil:
		MatmulBTQ8(a, w.q8, w.scales, dst, M, w.cols, w.rows)
	default:
		ws.MatmulBT(a, w.f32, dst, M, w.cols, w.rows)
	}
}

// Row dequantizes row i (one out-feature's weights, or a token's embedding when this
// matrix is an embedding table) into dst[:cols].
func (w *WeightMat) Row(i int, dst []float32) {
	switch {
	case w.q4 != nil:
		bpr := (w.cols + 1) / 2
		nGroups := (w.cols + w.group - 1) / w.group
		DequantizeRowInt4(w.q4[i*bpr:(i+1)*bpr], w.q4s[i*nGroups:(i+1)*nGroups], w.group, w.cols, dst)
	case w.q8 != nil:
		lo := i * w.cols
		DequantizeRowInt8(w.q8[lo:lo+w.cols], w.scales[i], dst)
	default:
		copy(dst, w.f32[i*w.cols:i*w.cols+w.cols])
	}
}

func (w *WeightMat) Rows() int { return w.rows }
func (w *WeightMat) Cols() int { return w.cols }

// Kind reports the stored precision: "int4", "int8", "f32", or "" (empty/zero value).
func (w *WeightMat) Kind() string {
	switch {
	case w.q4 != nil:
		return "int4"
	case w.q8 != nil:
		return "int8"
	case w.f32 != nil:
		return "f32"
	default:
		return ""
	}
}

// Int8 returns the int8 weights, per-row scales, and whether the matmul is W8A8
// (ok=false unless int8-resident). For GPU export, serialization, and a consumer's
// own int8 matmul (e.g. the encoder's baked-scale blocked kernel).
func (w *WeightMat) Int8() (q8 []int8, scales []float32, w8a8, ok bool) {
	return w.q8, w.scales, w.w8a8, w.q8 != nil
}

// Int4 returns the packed nibbles, per-group scales, and group size (ok=false unless
// int4-resident).
func (w *WeightMat) Int4() (q4 []byte, q4s []float32, group int, ok bool) {
	return w.q4, w.q4s, w.group, w.q4 != nil
}

// F32 returns the dense weights (ok=false unless f32-resident) — e.g. for a GPU
// backend's f32 matmul.
func (w *WeightMat) F32() (f32 []float32, ok bool) { return w.f32, w.f32 != nil }

// MappedSpan returns the page-aligned interior of this weight's quantized backing
// bytes — but ONLY if those bytes lie inside the [base, end) mapping (a region from
// mmap.MapReadOnly). It returns nil for an f32 weight, an empty weight, or any
// weight whose bytes are heap-backed rather than aliased from the mapping.
//
// This is the bridge between a WeightMat and mmap.SpanCache: the returned span is
// exactly what Advise (MADV_DONTNEED) can release without disturbing a neighbor's
// page, so a pager can register it with SpanCache.Add and page the tensor in and out
// under a RAM budget. Page-rounding the start up and the end down (via
// mmap.PageAlignedInterior) keeps the span strictly within this weight's bytes; the
// few boundary bytes it omits are negligible against a multi-MB tensor.
//
// Only quantized (int8/int4) storage is pageable here — that is goinfer's expert /
// layer weight case, and the f32 path stays out because the scales and dense weights
// it would mix in are small and commonly heap-backed. The caller obtains base/end
// from the mapping it passed to MapReadOnly (e.g. &mapping[0] and one past its end).
func (w *WeightMat) MappedSpan(base, end uintptr) []byte {
	var raw []byte
	switch {
	case len(w.q8) > 0:
		// int8 and byte are both 1 byte wide, so the length is unchanged.
		raw = unsafe.Slice((*byte)(unsafe.Pointer(&w.q8[0])), len(w.q8))
	case len(w.q4) > 0:
		raw = w.q4
	default:
		return nil // f32 or empty — not a pageable quantized tensor
	}
	start := uintptr(unsafe.Pointer(&raw[0]))
	if start < base || start+uintptr(len(raw)) > end {
		return nil // heap-backed, not part of the mapping
	}
	return mmap.PageAlignedInterior(raw)
}
