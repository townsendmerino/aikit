package linalg

import (
	"fmt"
	"unsafe"

	"github.com/townsendmerino/aikit/mmap"
)

// WeightMat is a [rows, cols] = [out, in] weight matrix that hides its storage
// precision behind a uniform matmul + dequant surface. It consolidates three
// open-coded wrappers that each held the same thing — an f32 / int8 / int4 weight
// plus its scales plus a precision-dispatched MatmulBT. All three are now migrated
// onto it (roadmap §2.8), each bit-identically:
//
//   - aikit encoder.LayerWeightsQ8 — per-row int8 projection weights (storage only;
//     the encoder keeps its own baked-scale blocked matmul for large-M prefill, fed
//     from Int8()/Scales(), since that path is numerically distinct from MatmulBTQ8).
//   - goinfer decoder.weightMat — the richest: f32 / per-row int8 / group int4 / W8A8,
//     with the matmul dispatch and tied-embedding Row lookup.
//   - aikit vision.qmat (was goinfer's, before §2.9 moved the tower) — f32 / W8A8
//     for the SigLIP / Qwen-ViT towers. Now only vision.newQMat remains, holding the
//     storage POLICY (which weights quantize; f32 copies because the source mmap is
//     released after load) rather than a second abstraction.
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

	// q4Row4/q4Row4Scales: an OPTIONAL, additional arm64-only in-RAM layout —
	// split-half nibbles + 4-row interleave (docs/task-w4a8-neon-bandwidth.md's
	// item-3+4 harness, GO 2026-08-23/24) — set only by an explicit
	// RepackInt4Row4 call (never probed for or built implicitly inside a
	// matmul). Bit-identical to the canonical q4/q4s for the same logical
	// weights (TestDotW4A8SplitHalf4Row_bitIdenticalToCanonical), so this is a
	// pure performance cache, not a second source of truth: q4/q4s stay
	// authoritative and are never dropped by RepackInt4Row4 itself (the
	// drop-canonical-after-repack question is the plumbing phase's own
	// load-time/resident-memory measurement to make, not decided here).
	q4Row4       []byte    // non-nil ⇒ split-half + 4-row-interleaved packed nibbles (RepackW4A8Row4 layout)
	q4Row4Scales []float32 // interleaved per-group scales (RepackW4A8Row4Scales layout)

	// q4SplitHalf: the amd64 counterpart of q4Row4 — an OPTIONAL, additional in-RAM layout,
	// split-half only (no row interleave), set exclusively by an explicit RepackInt4SplitHalf
	// call and never probed for or built implicitly. q4/q4s stay authoritative and are never
	// dropped, so M>1 and any non-AVX2 path keep working unchanged. NO separate scale array:
	// split-half permutes nibbles within a group and never reorders groups, so q4s serves both
	// layouts. See dot_w4a8_amd64.s for why the shuffle port, not the accumulator chain, is the
	// AVX2 bottleneck this addresses.
	q4SplitHalf []byte
}

// SplitHalfBytes reports the size of the optional split-half nibble layout, or 0
// when this WeightMat has none (every arch but amd64, any CPU without AVX2, any
// shape RepackInt4SplitHalf rejected, or simply no repack requested).
//
// The layout is a SECOND copy — q4 is never dropped — so this is exactly the
// extra resident memory the repack cost, and a caller deciding whether to opt in
// needs it. It exists so that caller does not have to re-derive rows x ceil(cols/2)
// from the outside and thereby duplicate this package's knowledge of the layout's
// size; goinfer's load-time accounting for the split-half A/B does exactly that
// and reads this instead.
//
// It also keeps q4SplitHalf referenced on architectures with no split-half kernel.
// Without it the field is written and read only from amd64-tagged files, so
// `staticcheck` on any other target reports it as an unused field (U1000) — a red
// that has nothing to do with the code under test and that this repo's CI, which
// analyses linux/amd64, would never show. That is the same build-tag/U1000 trap
// that has held CI red here before, arrived at from the opposite direction.
func (w *WeightMat) SplitHalfBytes() int { return len(w.q4SplitHalf) }

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

// WrapInt4Row4 is WrapInt4 plus an ALREADY-repacked split-half + 4-row-interleaved
// layout (q4Row4/q4Row4Scales, in RepackW4A8Row4/RepackW4A8Row4Scales's own output
// shape) — for a caller that computed or read those bytes itself (a prequant tool
// writing them to disk, or a loader mmap-aliasing them back from a serialized
// bundle) rather than one that wants RepackInt4Row4 to derive them from q4/q4s at
// call time. Portable (no build tag): on a non-arm64 build the fields are simply
// unused, since MatmulBTW4A8Into's non-arm64 form never reads them.
//
// Aliases q4Row4/q4Row4Scales WITHOUT copying, same contract as WrapInt4 for
// q4/q4s. Pass both nil to get exactly WrapInt4's result (no row4 layout attached).
//
// SAFETY GATE: silently keeps q4Row4/q4Row4Scales unset (canonical-only, same as
// passing nil) when row4Usable() is false — a non-arm64 build, or an arm64 core
// without the DotProd extension. RepackInt4Row4 already refuses to populate
// q4Row4 on such a core; this is the same gate applied to bytes that arrived
// from somewhere else (a .giw kind-4 file, computed on a DIFFERENT machine that
// may have DotProd when this one doesn't) rather than derived locally. Without
// this, MatmulBTW4A8Into's dispatch (which only checks q4Row4 != nil, not the
// CPU feature — that was always RepackInt4Row4's job) could route to a kernel
// this core cannot safely execute.
//
// Panics on the same q4/q4s length mismatches WrapInt4 checks, plus (when
// q4Row4/q4Row4Scales are non-nil AND row4Usable()) if they don't match
// RepackW4A8Row4's own output-length contract for this rows/cols/group — the
// same shape the repack functions would have produced, applied to bytes that
// arrived from elsewhere.
func WrapInt4Row4(q4 []byte, q4s []float32, rows, cols, group int, q4Row4 []byte, q4Row4Scales []float32) WeightMat {
	w := WrapInt4(q4, q4s, rows, cols, group)
	if q4Row4 == nil && q4Row4Scales == nil {
		return w
	}
	if !row4Usable() {
		return w
	}
	// SHAPE, NOT JUST LENGTH (task-simd-audit.md S-10). The row4 kernel interleaves
	// FOUR rows and reads whole groups; it cannot express rows%4 != 0 or
	// cols%group != 0. Until this check existed, a blob with the right byte COUNT
	// but the wrong shape passed both requireExactLen calls below, got stored, and
	// panicked later inside MatmulBTW4A8Row4Into on the first M=1 matmul — far from
	// the blob that caused it and with nothing naming it.
	//
	// RepackInt4Row4 (weightmat_row4_arm64.go) already declines exactly this shape,
	// so the constraint was known; this is the OTHER door into the same field, the
	// one a prebuilt kind-4 blob comes through. Panicking here is deliberate and
	// matches the two length checks immediately below: a caller cannot recover from
	// a malformed weight blob, and failing at load names it while failing at the
	// first matmul does not.
	if group <= 0 || rows%4 != 0 || cols%group != 0 {
		panic(fmt.Sprintf("linalg: WrapInt4Row4: the row4 layout needs rows%%4==0 and cols%%group==0, "+
			"got rows=%d cols=%d group=%d", rows, cols, group))
	}
	nGroups, bpr := groupsFor(cols, group)
	requireExactLen("WrapInt4Row4", "q4Row4", len(q4Row4), mul(rows, bpr))
	requireExactLen("WrapInt4Row4", "q4Row4Scales", len(q4Row4Scales), mul(rows, nGroups))
	w.q4Row4 = q4Row4
	w.q4Row4Scales = q4Row4Scales
	return w
}

// Int4Row4Usable reports whether a row4-repacked (canonical or repacked-
// only) WeightMat can be built for this shape on THIS core — the same gate
// RepackInt4Row4 already applies before setting q4Row4, exported (audit
// M-22) so a caller can check it BEFORE committing to repacked-only at
// construction time: RepackInt4Row4InPlace/WrapInt4Row4Only have no canonical
// to fall back to if the shape or core does not qualify, unlike
// RepackInt4Row4/WrapInt4Row4, which decline quietly and keep the tensor
// canonical-only.
func Int4Row4Usable(rows, cols, group int) bool {
	return row4Usable() && group == 32 && rows%4 == 0 && cols%group == 0
}

// Int4SplitHalfUsable is Int4Row4Usable's amd64 split-half twin — the same
// gate RepackInt4SplitHalf applies (AVX2 present, no VNNI — see that
// method's comment for why a VNNI host declines split-half rather than
// downgrading to it), exported for the same before-construction reason.
func Int4SplitHalfUsable(cols, group int) bool {
	return splitHalfUsable() && group == 32 && cols%32 == 0
}

// RepackInt4Row4InPlace permutes the CALLER'S OWN q4/q4s into row4 layout —
// audit M-22's repacked-only policy, in-place form. Declared here rather
// than alongside the other repacked-only constructors below because its
// implementation is arm64-specific (RepackInt4Row4Quad/RepackInt4Row4Scales
// Quad, weightmat_row4_arm64.go); see that file for the real body and
// weightmat_row4_other.go for the portable (always ok=false) stub.

// WrapInt4Row4Only wraps ALREADY-repacked row4 bytes (RepackW4A8Row4/
// RepackW4A8Row4Scales layout — a .giw kind-4 file storing pre-repacked
// bytes, or a loader that streamed canonical straight into row4 order via
// RepackInt4Row4Quad per quad as it copied off an mmap'd checkpoint) into a
// repacked-only WeightMat — audit M-22, WrapInt4's no-canonical-at-all twin.
// This is the form that actually avoids EVERY extra byte
// RepackInt4Row4InPlace's quad scratch still costs: canonical never exists
// in this process's heap at all, not even transiently.
//
// ok=false (zero WeightMat, q4Row4/q4Row4Scales NOT retained) when
// Int4Row4Usable(rows, cols, group) is false — this core cannot safely run
// the kernel these bytes are shaped for, and unlike WrapInt4Row4 there is no
// canonical to fall back to, so returning a half-built WeightMat would be
// worse than refusing.
//
// Panics on a length mismatch against RepackW4A8Row4/RepackW4A8Row4Scales's
// own output-length contract for this rows/cols/group — the same shape
// check WrapInt4Row4 applies to externally-supplied row4 bytes.
func WrapInt4Row4Only(q4Row4 []byte, q4Row4Scales []float32, rows, cols, group int) (WeightMat, bool) {
	if !Int4Row4Usable(rows, cols, group) {
		return WeightMat{}, false
	}
	nGroups, bpr := groupsFor(cols, group)
	requireExactLen("WrapInt4Row4Only", "q4Row4", len(q4Row4), mul(rows, bpr))
	requireExactLen("WrapInt4Row4Only", "q4Row4Scales", len(q4Row4Scales), mul(rows, nGroups))
	return WeightMat{q4Row4: q4Row4, q4Row4Scales: q4Row4Scales, group: group, rows: rows, cols: cols}, true
}

// RepackInt4SplitHalfInPlace is RepackInt4Row4InPlace's amd64 split-half
// twin: q4 is permuted in place, ROW BY ROW (split-half does not interleave
// rows, so the scratch is one row — bpr bytes — not one quad), using
// RepackInt4SplitHalfRow the same scratch-then-overwrite way. q4s is
// retained UNCHANGED in the result: split-half shares the canonical scale
// array rather than repacking it (see the q4SplitHalf field comment), so it
// is the one piece of "canonical" state a split-half-only WeightMat
// legitimately keeps — Int4()'s ok is still false, since ok means "packed
// NIBBLES present", not scales. Same input-is-consumed contract as
// RepackInt4Row4InPlace.
func RepackInt4SplitHalfInPlace(q4 []byte, q4s []float32, rows, cols, group int) (WeightMat, bool) {
	if !Int4SplitHalfUsable(cols, group) {
		return WeightMat{}, false
	}
	nGroups, bpr := groupsFor(cols, group)
	requireExactLen("RepackInt4SplitHalfInPlace", "q4", len(q4), mul(rows, bpr))
	requireExactLen("RepackInt4SplitHalfInPlace", "q4s", len(q4s), mul(rows, nGroups))
	rowScratch := make([]byte, bpr) // the one allocation, reused every row
	for r := range rows {
		row := q4[r*bpr : (r+1)*bpr]
		copy(rowScratch, row)
		RepackInt4SplitHalfRow(row, rowScratch, cols)
	}
	return WeightMat{q4SplitHalf: q4, q4s: q4s, group: group, rows: rows, cols: cols}, true
}

// WrapInt4SplitHalfOnly is WrapInt4Row4Only's amd64 split-half twin: q4s is
// still required (it is the canonical PER-GROUP SCALES, shared by split-half
// — see the q4SplitHalf field comment — not the canonical packed NIBBLES,
// which this constructor never takes). ok=false when
// Int4SplitHalfUsable(cols, group) is false.
func WrapInt4SplitHalfOnly(q4SplitHalf []byte, q4s []float32, rows, cols, group int) (WeightMat, bool) {
	if !Int4SplitHalfUsable(cols, group) {
		return WeightMat{}, false
	}
	nGroups, bpr := groupsFor(cols, group)
	requireExactLen("WrapInt4SplitHalfOnly", "q4SplitHalf", len(q4SplitHalf), mul(rows, bpr))
	requireExactLen("WrapInt4SplitHalfOnly", "q4s", len(q4s), mul(rows, nGroups))
	return WeightMat{q4SplitHalf: q4SplitHalf, q4s: q4s, group: group, rows: rows, cols: cols}, true
}

// MatmulBT computes dst[M, rows] = a[M, cols] · weight[rows, cols]ᵀ, dispatching by
// stored precision to the matching linalg kernel. CPU only — a consumer with a GPU
// backend dispatches via the raw accessors and uses this as the fallback.
func (w *WeightMat) MatmulBT(a, dst []float32, M int) {
	switch {
	case w.q4 != nil:
		MatmulBTW4A8(a, w.q4, w.q4s, dst, M, w.cols, w.rows, w.group)
	case w.q4Row4 != nil, w.q4SplitHalf != nil:
		// Repacked-only (audit M-22): the case above is UNCHANGED (still the
		// free-function call, still the exact code path a canonical-only or
		// "both" WeightMat took before this case existed) — this is reached
		// only by a WeightMat that never had canonical bytes at all, a state
		// no constructor could produce before M-22's WrapInt4Row4Only /
		// RepackInt4Row4InPlace / the split-half twins. Routes through the
		// per-arch method, which already prefers the repacked layout.
		var ws Workspace
		w.MatmulBTW4A8Into(&ws, a, dst, M)
	case w.q8 != nil && w.w8a8:
		MatmulBTW8A8(a, w.q8, w.scales, dst, M, w.cols, w.rows)
	case w.q8 != nil:
		MatmulBTQ8(a, w.q8, w.scales, dst, M, w.cols, w.rows)
	default:
		MatmulBT(a, w.f32, dst, M, w.cols, w.rows)
	}
}

// MatmulBTInto is MatmulBT through a Workspace: the W8A8 and W4A8 paths quantize
// the activation once into the Workspace's reusable scratch (zero per-call alloc
// — the steady-state decode win), the weight-only Q8 path takes its widened-row
// scratch from the Workspace too (MatmulBTQ8Into), and the f32 path runs the
// Workspace-scoped parallel matmul honoring its SetThreshold/SetWorkers. So every
// storage kind is now zero-alloc on the serial decode path.
func (w *WeightMat) MatmulBTInto(ws *Workspace, a, dst []float32, M int) {
	// Case order kept identical to MatmulBT above: with constructor-built values
	// the storage kinds are mutually exclusive so order is inert, but a
	// hand-built WeightMat with more than one set would otherwise route the two
	// entry points to different kernels.
	switch {
	case w.q4 != nil:
		MatmulBTW4A8Into(ws, a, w.q4, w.q4s, dst, M, w.cols, w.rows, w.group)
	case w.q4Row4 != nil, w.q4SplitHalf != nil:
		// Repacked-only (audit M-22) — see MatmulBT's identical case for why
		// the q4 != nil branch above is untouched.
		w.MatmulBTW4A8Into(ws, a, dst, M)
	case w.q8 != nil && w.w8a8:
		MatmulBTW8A8Into(ws, a, w.q8, w.scales, dst, M, w.cols, w.rows)
	case w.q8 != nil:
		MatmulBTQ8Into(ws, a, w.q8, w.scales, dst, M, w.cols, w.rows)
	default:
		ws.MatmulBT(a, w.f32, dst, M, w.cols, w.rows)
	}
}

// Row dequantizes row i (one out-feature's weights, or a token's embedding when this
// matrix is an embedding table) into dst[:cols].
//
// Layout-independent since audit M-22: the row4 and split-half int4 layouts
// are fixed PERMUTATIONS of the same nibbles canonical packing holds (see
// weightmat_int4_repacked_row.go), so a repacked-only WeightMat — one that
// never retained w.q4 — still answers Row() correctly. That is what lets a
// tied embedding table (read per-token via Row() AND driven through the
// matmul as the LM head) be repacked-only: the largest tensor in the model,
// and the one where the memory saving matters most. Checked in canonical's
// favor first when both are present, so a "both"-policy WeightMat's Row()
// output and code path are UNCHANGED from before this case existed.
func (w *WeightMat) Row(i int, dst []float32) {
	switch {
	case w.q4 != nil:
		bpr := (w.cols + 1) / 2
		nGroups := (w.cols + w.group - 1) / w.group
		DequantizeRowInt4(w.q4[i*bpr:(i+1)*bpr], w.q4s[i*nGroups:(i+1)*nGroups], w.group, w.cols, dst)
	case w.q4Row4 != nil:
		dequantizeRowFromRow4(w.q4Row4, w.q4Row4Scales, w.cols, i, dst)
	case w.q4SplitHalf != nil:
		dequantizeRowFromSplitHalf(w.q4SplitHalf, w.q4s, w.cols, i, dst)
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
// Reports "int4" for a repacked-only WeightMat too (audit M-22) — Kind is a
// precision label, not a "canonical bytes present" check; use Int4() for that.
func (w *WeightMat) Kind() string {
	switch {
	case w.q4 != nil, w.q4Row4 != nil, w.q4SplitHalf != nil:
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

// Int4 returns the packed CANONICAL nibbles, per-group scales, and group
// size — ok iff canonical bytes are present, NOT iff this WeightMat is
// int4-resident (audit M-22 made those two questions different: a
// repacked-only WeightMat is int4 — Kind()/IsInt4() say so — but ok is
// false here, deliberately, because there is nothing to return). This is
// the accessor a staged GPU consult uploads from (the row4/split-half
// layouts are CPU-SIMD-specific interleaves, not valid input for a GPU
// kernel expecting canonical bytes), so ok=true with nil q4 would be the
// worst outcome — a caller trusting ok and dereferencing nil. Use IsInt4()
// to ask "is this tensor int4 at all" and Int4Layout() to ask "in which
// form".
func (w *WeightMat) Int4() (q4 []byte, q4s []float32, group int, ok bool) {
	return w.q4, w.q4s, w.group, w.q4 != nil
}

// IsInt4 reports whether this WeightMat is int4-resident in ANY layout —
// canonical, row4, or split-half (audit M-22). Unlike Int4()'s ok, this is
// true for a repacked-only WeightMat. The question a caller choosing which
// matmul branch to take should ask; Int4()'s ok is the narrower "do I have
// canonical bytes to hand somewhere that needs them" question.
func (w *WeightMat) IsInt4() bool {
	return w.q4 != nil || w.q4Row4 != nil || w.q4SplitHalf != nil
}

// Int4Layout reports which int4 representation(s) this WeightMat holds:
// "canonical", "row4", "splithalf", "canonical+row4", "canonical+splithalf",
// or "" if not int4-resident. row4 and splithalf are mutually exclusive
// (arm64 vs amd64 repacks), so "row4+splithalf" never occurs. Informational
// — for logging, a manifest, or load-time resident-memory accounting; no
// dispatch reads this.
func (w *WeightMat) Int4Layout() string {
	canonical := w.q4 != nil
	switch {
	case canonical && w.q4Row4 != nil:
		return "canonical+row4"
	case canonical && w.q4SplitHalf != nil:
		return "canonical+splithalf"
	case canonical:
		return "canonical"
	case w.q4Row4 != nil:
		return "row4"
	case w.q4SplitHalf != nil:
		return "splithalf"
	default:
		return ""
	}
}

// Int4Row4 returns the split-half + 4-row-interleaved layout (RepackW4A8Row4/
// RepackW4A8Row4Scales) if RepackInt4Row4 has populated it (ok=false
// otherwise — the zero value on non-arm64 builds, on int4-less WeightMats,
// and on any int4 WeightMat that hasn't been repacked, e.g. a paged-MoE
// tensor built transient over an mmap span). Pure performance cache: use
// (w *WeightMat) MatmulBTW4A8Into for the M=1 dispatch decision rather than
// branching on this directly, unless you specifically need the raw bytes
// (e.g. measuring the repack's resident-memory delta).
func (w *WeightMat) Int4Row4() (packed4 []byte, scales4 []float32, ok bool) {
	return w.q4Row4, w.q4Row4Scales, w.q4Row4 != nil
}

// F32 returns the dense weights (ok=false unless f32-resident) — e.g. for a GPU
// backend's f32 matmul.
func (w *WeightMat) F32() (f32 []float32, ok bool) { return w.f32, w.f32 != nil }

// MappedSpan returns the page-aligned interior of this weight's quantized backing
// bytes — but ONLY if those bytes lie inside the [base, end) mapping (a region from
// mmap.MapReadOnly). It returns nil for an f32 weight, an empty weight, or any
// weight whose bytes are heap-backed rather than aliased from the mapping.
//
// Canonical-only by nature (audit M-22): it reads w.q4, which a repacked-only
// WeightMat never has, so it returns nil there too — same as f32/empty. This
// is not expected to matter in practice: MappedSpan exists for the PAGED
// (mmap, never fully loaded) case, and a paged tensor has no load-time
// repack step to begin with (RepackInt4Row4InPlace/RepackInt4SplitHalfInPlace need
// canonical bytes IN HAND to repack from), so a pageable tensor and a
// repacked-only one are not expected to be the same tensor. A caller
// combining paging with a repacked-only policy would need to read w.q4Row4
// directly — there is no MappedSpanRow4-equivalent gap here to close, since
// MappedSpanRow4 already exists and does not depend on canonical presence.
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

// MappedSpanRow4 is MappedSpan for the split-half + 4-row-interleaved layout
// (q4Row4) instead of the canonical q4 — same page-alignment/containment contract,
// returned separately because a WeightMat carrying BOTH layouts (WrapInt4Row4) has
// two independently-mmap'd byte arrays a pager must account for separately: they
// are different spans at different offsets in the same file, not one span twice.
// nil if q4Row4 was never populated, or if it's heap-backed rather than part of
// [base, end).
func (w *WeightMat) MappedSpanRow4(base, end uintptr) []byte {
	if len(w.q4Row4) == 0 {
		return nil
	}
	start := uintptr(unsafe.Pointer(&w.q4Row4[0]))
	if start < base || start+uintptr(len(w.q4Row4)) > end {
		return nil
	}
	return mmap.PageAlignedInterior(w.q4Row4)
}
