// Package embed implements Model2Vec inference: hand-rolled WordPiece
// tokenization, safetensors weight loading, weighted-mean pooling, and L2
// normalization. Pure Go, no cgo.
//
// The model artifact for potion-code-16M contains three tensors:
//
//	embeddings  F32  [vocab_size, embed_dim]  — the embedding rows
//	mapping     I64  [vocab_size]             — token-id → embedding-row index
//	weights     F64  [vocab_size]             — per-vocab-token weight (runtime-applied)
//
// Inference:
//
//	v = Σ embeddings[mapping[id]] · weights[id]   (sum over tokens)
//	v = v / Σ weights[id]
//	output = v / ‖v‖₂                              (when config.normalize)
//
// See ken's DESIGN.md §4 for the design rationale and the pin_inference.py
// golden-test script that validated this algorithm. This package moved here
// from ken (ADR-034); "DESIGN.md" refers to
// https://github.com/townsendmerino/ken/blob/main/docs/DESIGN.md.
package embed

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"unsafe"

	"github.com/townsendmerino/aikit/mmap"
)

// SafetensorsFile is a parsed safetensors file. Tensor payloads are slices
// into the underlying file bytes — no copy. Goroutine-safe for reads.
//
// Lifetime: callers must keep the SafetensorsFile alive for as long as
// any tensor returned by Tensor() is in use. With the heap-loaded path
// (OpenSafetensors / OpenSafetensorsFromFS) the underlying []byte stays
// in Go's heap until SafetensorsFile is GC'd. With the mmap-loaded path
// (OpenSafetensorsMmap, M8) the underlying region is munmap'd via a
// runtime.SetFinalizer attached at Open time — Close() forces it earlier.
type SafetensorsFile struct {
	data    []byte
	tensors map[string]Tensor

	// mmapped holds the mmap region(s) backing this file iff it was loaded
	// via a *Mmap open (and Close hasn't run). One entry for a single file;
	// one per shard for OpenSafetensorsShardedMmap. Close/the finalizer
	// syscall.Munmap each; the heap opens leave it nil.
	mmapped [][]byte

	// closed is set by Close so Tensor() returns a clean error instead of
	// handing out an alias into a region Close may have unmapped.
	closed bool
}

// Tensor is a single named tensor within a SafetensorsFile.
//
// Lifetime: the bytes a Tensor exposes (Float32s, BFloat16sToF32, …) alias the
// owning SafetensorsFile's mapped region — they are NOT copies. They stay valid
// only while that file is alive and un-Closed. Calling Tensor() after Close
// returns an error; but a tensor (or slice) obtained BEFORE Close and read AFTER
// it dereferences unmapped memory — a crash or silent garbage the flag can't
// catch. Keep the file alive for the whole lifetime of any data drawn from it:
//
//	// WRONG — the returned slice outlives the file's mapping.
//	func load() []float32 {
//	    f, _ := embed.OpenSafetensorsMmap("model.safetensors")
//	    defer f.Close()                 // unmaps on return…
//	    t, _ := f.Tensor("weight")
//	    v, _ := t.Float32s()
//	    return v                        // …v now aliases freed memory
//	}
//
//	// RIGHT — copy out, or keep the file alive as long as the data is used.
//	func load() []float32 {
//	    f, _ := embed.OpenSafetensorsMmap("model.safetensors")
//	    defer f.Close()
//	    t, _ := f.Tensor("weight")
//	    v, _ := t.Float32s()
//	    return append([]float32(nil), v...) // independent copy; safe to outlive f
//	}
type Tensor struct {
	Name  string
	DType string // "F32", "F64", "I64", ...
	Shape []int
	raw   []byte // little-endian bytes; slice into the owning file's []byte
	// owner is the SafetensorsFile whose data/mmap backs raw. The byte-reading
	// accessors runtime.KeepAlive(owner) across a read so an mmap-backed file's
	// finalizer can't munmap the region mid-decode (§2.5 — the Tensor value alone
	// doesn't otherwise keep the file reachable).
	owner *SafetensorsFile
}

// safetensors header JSON shape:
//
//	{
//	  "tensor_name": {"dtype": "F32", "shape": [...], "data_offsets": [start, end]},
//	  ...
//	  "__metadata__": {...}    // optional, skipped
//	}
type rawHeader map[string]json.RawMessage

type rawTensor struct {
	DType       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets [2]int `json:"data_offsets"`
}

// OpenSafetensors reads a safetensors file from disk and parses its header.
// The file body is loaded into memory once; tensor data is referenced by
// zero-copy slice. For the model sizes we care about (~64 MB for
// potion-code-16M) this is the right trade-off. Swap to mmap if/when memory
// becomes an issue.
func OpenSafetensors(path string) (*SafetensorsFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read safetensors: %w", err)
	}
	return parseSafetensors(data)
}

// OpenSafetensorsFromFS reads a safetensors file from fsys at name and
// parses its header. Same semantics as OpenSafetensors but takes an fs.FS
// so callers can serve the model out of an //go:embed embed.FS, a
// fstest.MapFS, or any other fs.FS implementation.
func OpenSafetensorsFromFS(fsys fs.FS, name string) (*SafetensorsFile, error) {
	data, err := fs.ReadFile(fsys, name)
	if err != nil {
		return nil, fmt.Errorf("read safetensors: %w", err)
	}
	return parseSafetensors(data)
}

// shardIndex is the model.safetensors.index.json shape: weight_map names every
// tensor and the shard file it lives in.
type shardIndex struct {
	WeightMap map[string]string `json:"weight_map"`
}

// parseShardIndex parses a model.safetensors.index.json and returns the unique
// shard filenames (sorted, deterministic) plus the full weight_map (for a
// post-load completeness check).
func parseShardIndex(indexBytes []byte) (files []string, weightMap map[string]string, err error) {
	var idx shardIndex
	if err := json.Unmarshal(indexBytes, &idx); err != nil {
		return nil, nil, fmt.Errorf("safetensors: parse shard index: %w: %w", err, ErrFormat)
	}
	if len(idx.WeightMap) == 0 {
		return nil, nil, errFormatf("safetensors: shard index has empty weight_map")
	}
	seen := make(map[string]bool)
	for _, fn := range idx.WeightMap {
		// Shard names come from the (untrusted) index JSON and are joined to
		// the index's directory. Require a plain filename so a crafted bundle
		// (`"w": "../../etc/passwd"`, an absolute path, or a Windows drive/UNC
		// path) cannot escape the model directory — a zip-slip-style arbitrary
		// read. The fs.FS variant is already safe via fs.ValidPath; this guards
		// the Mmap path that resolves against the real filesystem.
		if fn == "" || fn == "." || fn == ".." || filepath.IsAbs(fn) || strings.ContainsAny(fn, `/\`) {
			return nil, nil, fmt.Errorf("%w: shard index names unsafe shard path %q", ErrFormat, fn)
		}
		if !seen[fn] {
			seen[fn] = true
			files = append(files, fn)
		}
	}
	sort.Strings(files)
	return files, idx.WeightMap, nil
}

// mergeShards folds each shard's tensors into agg and verifies every tensor the
// weight_map promises actually resolved. Caller supplies the per-shard parsed
// files; this is the format-level logic shared by the mmap and fs paths.
func mergeShards(agg *SafetensorsFile, shards []*SafetensorsFile, weightMap map[string]string) error {
	for _, shard := range shards {
		for name, t := range shard.tensors {
			// Repoint owner to agg: after the merge it's agg that holds every
			// shard's mmap region (agg.mmapped) and carries the finalizer, so a
			// tensor's KeepAlive must keep AGG reachable, not its parse-time shard.
			t.owner = agg
			agg.tensors[name] = t
		}
	}
	for name := range weightMap {
		if _, ok := agg.tensors[name]; !ok {
			return fmt.Errorf("safetensors: shard index names tensor %q but no shard contains it", name)
		}
	}
	return nil
}

// OpenSafetensorsShardedMmap loads a multi-shard checkpoint named by a
// model.safetensors.index.json at indexPath (shard files resolved relative to
// it). Each shard is mmap'd once; the returned SafetensorsFile resolves
// Tensor() across all shards and Close munmaps them all — so callers use it
// exactly like a single-file SafetensorsFile.
func OpenSafetensorsShardedMmap(indexPath string) (*SafetensorsFile, error) {
	indexBytes, err := os.ReadFile(indexPath)
	if err != nil {
		return nil, fmt.Errorf("read shard index: %w", err)
	}
	files, weightMap, err := parseShardIndex(indexBytes)
	if err != nil {
		return nil, err
	}
	dir := filepath.Dir(indexPath)
	agg := &SafetensorsFile{tensors: make(map[string]Tensor)}
	shards := make([]*SafetensorsFile, 0, len(files))
	for _, fn := range files {
		data, err := mmap.MapReadOnly(filepath.Join(dir, fn))
		if err != nil {
			finalizeMmaps(agg)
			return nil, err
		}
		agg.mmapped = append(agg.mmapped, data)
		shard, err := parseSafetensors(data)
		if err != nil {
			finalizeMmaps(agg)
			return nil, fmt.Errorf("safetensors: shard %s: %w", fn, err)
		}
		shards = append(shards, shard)
	}
	if err := mergeShards(agg, shards, weightMap); err != nil {
		finalizeMmaps(agg)
		return nil, err
	}
	runtime.SetFinalizer(agg, finalizeMmaps)
	return agg, nil
}

// OpenSafetensorsShardedFromFS is the fs.FS (heap) counterpart of
// OpenSafetensorsShardedMmap — for embed.FS / fstest.MapFS. Each shard is read
// into the heap; the tensor slices keep their shard bytes alive via the
// returned file, so there's nothing to Close.
func OpenSafetensorsShardedFromFS(fsys fs.FS, indexPath string) (*SafetensorsFile, error) {
	indexBytes, err := fs.ReadFile(fsys, indexPath)
	if err != nil {
		return nil, fmt.Errorf("read shard index: %w", err)
	}
	files, weightMap, err := parseShardIndex(indexBytes)
	if err != nil {
		return nil, err
	}
	dir := path.Dir(indexPath)
	agg := &SafetensorsFile{tensors: make(map[string]Tensor)}
	shards := make([]*SafetensorsFile, 0, len(files))
	for _, fn := range files {
		data, err := fs.ReadFile(fsys, path.Join(dir, fn))
		if err != nil {
			return nil, fmt.Errorf("read shard %s: %w", fn, err)
		}
		shard, err := parseSafetensors(data)
		if err != nil {
			return nil, fmt.Errorf("safetensors: shard %s: %w", fn, err)
		}
		shards = append(shards, shard)
	}
	if err := mergeShards(agg, shards, weightMap); err != nil {
		return nil, err
	}
	return agg, nil
}

// OpenSafetensorsMmap mmaps path into memory (read-only, MAP_PRIVATE)
// and parses the safetensors header from the mapped region. Tensor
// slices alias the mapping; resident memory cost is shared via the OS
// page cache instead of the Go heap.
//
// This is the M8 path for large models — CodeRankEmbed's 547 MB
// checkpoint would otherwise dominate ken-mcp's RSS. The mapping is
// released via syscall.Munmap when the SafetensorsFile is GC'd
// (runtime.SetFinalizer) or when Close() is called explicitly.
//
// IMPORTANT: tensor data returned by Tensor() aliases the mapped
// region. Callers must keep the SafetensorsFile alive for as long as
// any such tensor is in use. After Close(), tensor accesses dereference
// unmapped memory — undefined behavior.
//
// Platform: true memory-mapping on unix (darwin/linux/bsd) via
// syscall.Mmap; on non-unix targets (Windows) it transparently falls back
// to a heap read (mmap.MapReadOnly's !unix path) — same API and semantics,
// just without the OS-page-cache sharing.
func OpenSafetensorsMmap(path string) (*SafetensorsFile, error) {
	data, err := mmap.MapReadOnly(path)
	if err != nil {
		return nil, err
	}
	sf, err := parseSafetensors(data)
	if err != nil {
		_ = mmap.Unmap(data)
		return nil, err
	}
	sf.mmapped = [][]byte{data}
	// Finalizer guards the common case where callers forget Close().
	runtime.SetFinalizer(sf, finalizeMmaps)
	return sf, nil
}

// finalizeMmaps is the SetFinalizer callback for mmap-backed files: munmap
// every region. Close does the same eagerly.
func finalizeMmaps(s *SafetensorsFile) {
	for _, m := range s.mmapped {
		_ = mmap.Unmap(m)
	}
	s.mmapped = nil
}

// Close releases the underlying mmap, if any. No-op on heap-loaded
// SafetensorsFile (OpenSafetensors / OpenSafetensorsFromFS). Idempotent.
//
// After Close, any Tensor() returns alias unmapped memory — accessing
// them is undefined behavior. Callers MUST stop using the model and any
// downstream objects that hold tensor data before calling Close.
func (f *SafetensorsFile) Close() error {
	f.closed = true // make Tensor() error afterward, mmap-backed or not
	if len(f.mmapped) == 0 {
		return nil
	}
	regions := f.mmapped
	f.mmapped = nil
	runtime.SetFinalizer(f, nil)
	var firstErr error
	for _, m := range regions {
		if err := mmap.Unmap(m); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// parseSafetensors is the shared safetensors-bytes parser used by both
// OpenSafetensors and OpenSafetensorsFromFS. Takes ownership of data — the
// returned SafetensorsFile retains a reference (the unsafe-slice tensor
// data aliases into it).
func parseSafetensors(data []byte) (*SafetensorsFile, error) {
	if len(data) < 8 {
		return nil, errFormatf("safetensors: file too short for header length prefix")
	}

	// First 8 bytes: little-endian uint64 header length.
	headerLen := uint64(data[0]) |
		uint64(data[1])<<8 |
		uint64(data[2])<<16 |
		uint64(data[3])<<24 |
		uint64(data[4])<<32 |
		uint64(data[5])<<40 |
		uint64(data[6])<<48 |
		uint64(data[7])<<56

	// Compare without adding 8: 8+headerLen overflows uint64 for a hostile
	// headerLen near 2^64, which would wrap the bound small and let the slice
	// below panic. len(data) ≥ 8 is guaranteed above, so the subtraction is safe.
	if headerLen > uint64(len(data))-8 {
		return nil, errFormatf("safetensors: file truncated (header claims %d bytes, only %d available)",
			headerLen, len(data)-8)
	}

	headerBytes := data[8 : 8+headerLen]
	payload := data[8+headerLen:]

	var raw rawHeader
	if err := json.Unmarshal(headerBytes, &raw); err != nil {
		return nil, fmt.Errorf("safetensors: parse header JSON: %w: %w", err, ErrFormat)
	}

	sf := &SafetensorsFile{data: data, tensors: make(map[string]Tensor, len(raw))}
	tensors := sf.tensors
	for name, rawJSON := range raw {
		if name == "__metadata__" {
			continue
		}
		var t rawTensor
		if err := json.Unmarshal(rawJSON, &t); err != nil {
			return nil, fmt.Errorf("safetensors: parse tensor %q: %w: %w", name, err, ErrFormat)
		}
		if t.DataOffsets[0] < 0 || t.DataOffsets[1] > len(payload) || t.DataOffsets[0] > t.DataOffsets[1] {
			return nil, errFormatf("safetensors: tensor %q has invalid offsets %v (payload size %d)",
				name, t.DataOffsets, len(payload))
		}
		// Cross-validate shape × dtype against the declared byte range (H2): the
		// offset range alone doesn't stop a hostile header from pairing a giant
		// shape with a 1-element byte range, which would parse here and then
		// panic at inference when a caller indexes by shape. Reject non-positive
		// dims and require ∏shape · dtypeSize == byte range, overflow-safe.
		// Unknown dtypes skip this (the typed accessors reject them at read).
		rawLen := t.DataOffsets[1] - t.DataOffsets[0]
		if sz, known := dtypeSize(t.DType); known {
			n := int64(1)
			for _, d := range t.Shape {
				if d < 0 {
					return nil, errFormatf("safetensors: tensor %q has negative dim in shape %v", name, t.Shape)
				}
				// Bound the running product by the payload size before multiplying
				// (n·d can't fit if n already exceeds payload/d) so it never wraps.
				if d != 0 && n > int64(len(payload))/int64(d) {
					return nil, errFormatf("safetensors: tensor %q shape %v exceeds the payload", name, t.Shape)
				}
				n *= int64(d)
			}
			if n*int64(sz) != int64(rawLen) {
				return nil, errFormatf("safetensors: tensor %q shape %v × %s (%dB/elem) = %d bytes, but byte range is %d",
					name, t.Shape, t.DType, sz, n*int64(sz), rawLen)
			}
		}
		tensors[name] = Tensor{
			Name:  name,
			DType: t.DType,
			Shape: t.Shape,
			raw:   payload[t.DataOffsets[0]:t.DataOffsets[1]],
			owner: sf,
		}
	}

	return sf, nil
}

// Tensor returns the named tensor or an error if not present. After Close it
// errors rather than hand out an alias into a possibly-unmapped region — a clean
// guard for the common use-after-close mistake (the held-slice case the Tensor
// doc warns about can't be caught here).
func (f *SafetensorsFile) Tensor(name string) (Tensor, error) {
	if f.closed {
		return Tensor{}, fmt.Errorf("safetensors: Tensor(%q) called after Close", name)
	}
	t, ok := f.tensors[name]
	if !ok {
		return Tensor{}, fmt.Errorf("safetensors: tensor %q not found", name)
	}
	return t, nil
}

// Names lists the tensor names in the file.
func (f *SafetensorsFile) Names() []string {
	out := make([]string, 0, len(f.tensors))
	for n := range f.tensors {
		out = append(out, n)
	}
	return out
}

// TensorF32 reads the named tensor as []float32, widening BF16 or F16 to f32 if
// needed (allocating), so callers loading weights don't dispatch on dtype themselves.
// If want is non-empty the tensor's shape must equal it, else an error is returned —
// the common shape-checked weight read, e.g.
//
//	w, err := sf.TensorF32("model.layers.0.mlp.down_proj.weight", outDim, inDim)
//
// Omit want to read without a shape check.
func (f *SafetensorsFile) TensorF32(name string, want ...int) ([]float32, error) {
	// H6: the BF16/F16 widening below reads t.raw, which aliases f's mmap
	// region; f's finalizer munmaps it. The Tensor value doesn't reference f, so
	// keep f reachable across the read/widen. (An F32 result is a zero-copy view
	// whose later use by the caller is governed by the Tensor lifetime contract.)
	defer runtime.KeepAlive(f)
	t, err := f.Tensor(name)
	if err != nil {
		return nil, err
	}
	if len(want) > 0 && !shapeEqual(t.Shape, want) {
		return nil, fmt.Errorf("safetensors: tensor %q shape %v != want %v", name, t.Shape, want)
	}
	switch t.DType {
	case "F32":
		return t.Float32s()
	case "BF16":
		return t.BFloat16sToF32()
	case "F16":
		return t.Float16sToF32()
	default:
		return nil, fmt.Errorf("safetensors: tensor %q dtype %q unsupported for an F32 read (want F32/BF16/F16)", name, t.DType)
	}
}

// TensorI32 reads the named tensor as []int32 (optionally shape-checked against want).
// Requires DType I32 — e.g. GPTQ packed qweight/qzeros/g_idx.
func (f *SafetensorsFile) TensorI32(name string, want ...int) ([]int32, error) {
	t, err := f.Tensor(name)
	if err != nil {
		return nil, err
	}
	if len(want) > 0 && !shapeEqual(t.Shape, want) {
		return nil, fmt.Errorf("safetensors: tensor %q shape %v != want %v", name, t.Shape, want)
	}
	return t.Int32s()
}

func shapeEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// reinterpretLE views t.raw as []T. safetensors is little-endian and this
// assumes a little-endian host (x86, arm64), so the view is exact.
//
// H3: the payload offset (8 + headerLen + DataOffsets[0]) is NOT guaranteed to
// be a multiple of the element size — the format doesn't require it, and honest
// files pack tensors back-to-back, so e.g. an I64 after a 4-mod-8 F32 lands
// 4-aligned. A misaligned `(*T)(unsafe.Pointer(...))` conversion is undefined:
// an unrecoverable checkptr/-race "misaligned pointer conversion" throw, and a
// SIGBUS on strict-alignment ports. So take the zero-copy view only when the
// data is T-aligned (the common case — writers pad the header); otherwise copy
// the bytes into a fresh, properly-aligned []T (the result then does not alias
// the file, exactly like the BF16/F16 widening paths).
func reinterpretLE[T any](name string, raw []byte) ([]T, error) {
	var zero T
	size := int(unsafe.Sizeof(zero))
	if len(raw)%size != 0 {
		return nil, fmt.Errorf("tensor %q: raw size %d not a multiple of %d", name, len(raw), size)
	}
	if len(raw) == 0 {
		return nil, nil
	}
	p := unsafe.Pointer(&raw[0])
	if uintptr(p)%unsafe.Alignof(zero) == 0 {
		return unsafe.Slice((*T)(p), len(raw)/size), nil // aligned: zero-copy view
	}
	// Misaligned: copy the raw bytes into an aligned []T. make() aligns out for
	// T, and the destination is viewed as []byte (byte alignment always holds),
	// so no misaligned typed access occurs; the little-endian payload is
	// preserved verbatim.
	out := make([]T, len(raw)/size)
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&out[0])), len(raw)), raw)
	return out, nil
}

// Float32s returns the tensor data as []float32. Requires DType "F32". The
// result aliases the file's bytes when they are element-aligned (the common
// case); on a misaligned payload it is a decoded copy (see reinterpretLE). Do
// not mutate. Assumes a little-endian host (x86, arm64).
func (t Tensor) Float32s() ([]float32, error) {
	defer runtime.KeepAlive(t.owner) // §2.5: guard the read against a mid-decode munmap
	if t.DType != "F32" {
		return nil, fmt.Errorf("tensor %q: expected F32, got %s", t.Name, t.DType)
	}
	return reinterpretLE[float32](t.Name, t.raw)
}

// Float64s returns the tensor data as []float64. Requires DType "F64". Aliases
// when aligned, else a copy (see reinterpretLE).
func (t Tensor) Float64s() ([]float64, error) {
	defer runtime.KeepAlive(t.owner) // §2.5
	if t.DType != "F64" {
		return nil, fmt.Errorf("tensor %q: expected F64, got %s", t.Name, t.DType)
	}
	return reinterpretLE[float64](t.Name, t.raw)
}

// Int64s returns the tensor data as []int64. Requires DType "I64". Aliases when
// aligned, else a copy (see reinterpretLE).
func (t Tensor) Int64s() ([]int64, error) {
	defer runtime.KeepAlive(t.owner) // §2.5
	if t.DType != "I64" {
		return nil, fmt.Errorf("tensor %q: expected I64, got %s", t.Name, t.DType)
	}
	return reinterpretLE[int64](t.Name, t.raw)
}

// Int32s returns the tensor data as []int32. Requires DType "I32" — used for
// GPTQ's packed qweight/qzeros/g_idx. Aliases when aligned, else a copy (see
// reinterpretLE).
func (t Tensor) Int32s() ([]int32, error) {
	defer runtime.KeepAlive(t.owner) // §2.5
	if t.DType != "I32" {
		return nil, fmt.Errorf("tensor %q: expected I32, got %s", t.Name, t.DType)
	}
	return reinterpretLE[int32](t.Name, t.raw)
}

// BFloat16sToF32 decodes a BF16 tensor to a freshly-allocated []float32.
// bfloat16 IS the top 16 bits of an IEEE-754 float32 (same sign, same
// 8-bit exponent, mantissa truncated to 7 bits), so widening is exact and
// branch-free: shift the 16-bit pattern up by 16 and reinterpret. NaN,
// Inf, and subnormals all carry over for free. Requires DType "BF16".
//
// Unlike Float32s/Float64s this ALLOCATES (the f32 form is twice the
// bytes and not a view into the file), so the result does not alias the
// SafetensorsFile and is safe to keep past Close().
func (t Tensor) BFloat16sToF32() ([]float32, error) {
	defer runtime.KeepAlive(t.owner) // §2.5
	if t.DType != "BF16" {
		return nil, fmt.Errorf("tensor %q: expected BF16, got %s", t.Name, t.DType)
	}
	if len(t.raw)%2 != 0 {
		return nil, fmt.Errorf("tensor %q: BF16 raw size %d not a multiple of 2", t.Name, len(t.raw))
	}
	n := len(t.raw) / 2
	out := make([]float32, n)
	for i := range n {
		// Little-endian uint16 → high 16 bits of the float32 pattern.
		bits := uint16(t.raw[2*i]) | uint16(t.raw[2*i+1])<<8
		out[i] = math.Float32frombits(uint32(bits) << 16)
	}
	return out, nil
}

// Float16sToF32 decodes an IEEE-754 half-precision (F16) tensor to a
// freshly-allocated []float32. Unlike bf16, f16 has a 5-bit exponent and
// 10-bit mantissa, so widening is a real rebias (exponent 15→127) with
// explicit handling of zeros/subnormals (exp==0) and Inf/NaN (exp==0x1f).
// Requires DType "F16". Allocates (see BFloat16sToF32 on aliasing).
func (t Tensor) Float16sToF32() ([]float32, error) {
	defer runtime.KeepAlive(t.owner) // §2.5
	if t.DType != "F16" {
		return nil, fmt.Errorf("tensor %q: expected F16, got %s", t.Name, t.DType)
	}
	if len(t.raw)%2 != 0 {
		return nil, fmt.Errorf("tensor %q: F16 raw size %d not a multiple of 2", t.Name, len(t.raw))
	}
	n := len(t.raw) / 2
	out := make([]float32, n)
	for i := range n {
		h := uint16(t.raw[2*i]) | uint16(t.raw[2*i+1])<<8
		out[i] = halfBitsToF32(h)
	}
	return out, nil
}

// halfBitsToF32 converts one IEEE-754 binary16 bit pattern to float32.
// Standard three-case decode: subnormal/zero (exp==0), Inf/NaN
// (exp==0x1f), and normal (rebias the 5-bit exponent to f32's 8-bit one).
// Subnormals are renormalized into a float32 normal — exp is kept signed
// because the normalization loop can drive it below zero before rebias.
func halfBitsToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := int32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)

	var bits uint32
	switch {
	case exp == 0:
		if mant == 0 {
			bits = sign // ±0
			break
		}
		// Subnormal: shift the mantissa left until the implicit leading
		// 1 reaches bit 10, decrementing the exponent per shift, then drop
		// that leading bit and rebias (15 → 127).
		exp = 1
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		mant &= 0x3ff
		bits = sign | uint32(exp+(127-15))<<23 | mant<<13
	case exp == 0x1f:
		// Inf (mant==0) or NaN (mant!=0); shift mantissa into f32 position.
		bits = sign | 0x7f800000 | mant<<13
	default:
		// Normal: rebias exponent, shift 10-bit mantissa to 23-bit.
		bits = sign | uint32(exp+(127-15))<<23 | mant<<13
	}
	return math.Float32frombits(bits)
}

// dtypeSize returns the byte width of a safetensors dtype and whether aikit
// knows it. Only the dtypes the typed accessors decode are sized; anything else
// (I8/U8/BOOL/F8_*, …) is reported unknown, so parseSafetensors skips the
// shape×dtype byte-range check for it — the typed accessors reject it at read
// time exactly as before.
func dtypeSize(dtype string) (int, bool) {
	switch dtype {
	case "F64", "I64":
		return 8, true
	case "F32", "I32":
		return 4, true
	case "BF16", "F16":
		return 2, true
	default:
		return 0, false
	}
}

// Elements returns the total number of elements (product of shape).
// Elements returns the total number of elements (product of Shape), or -1 if
// the product overflows int or any dim is negative. For a known dtype the H2
// parse check already guarantees a sane shape, but parseSafetensors stores an
// unknown-dtype tensor without that check, so a hostile shape like [1<<40,1<<40]
// could reach here — hence the overflow guard rather than a silently wrapped
// value.
func (t Tensor) Elements() int {
	n := 1
	for _, d := range t.Shape {
		if d < 0 || (d != 0 && n > math.MaxInt/d) {
			return -1
		}
		n *= d
	}
	return n
}
