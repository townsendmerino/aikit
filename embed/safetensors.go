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
// See ../../../docs/DESIGN.md §4 for the design rationale and the
// pin_inference.py golden-test script that validated this algorithm.
package embed

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"sort"
	"syscall"
	"unsafe"
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
}

// Tensor is a single named tensor within a SafetensorsFile.
type Tensor struct {
	Name  string
	DType string // "F32", "F64", "I64", ...
	Shape []int
	raw   []byte // little-endian bytes; slice into the owning file's []byte
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
		return nil, nil, fmt.Errorf("safetensors: parse shard index: %w", err)
	}
	if len(idx.WeightMap) == 0 {
		return nil, nil, errors.New("safetensors: shard index has empty weight_map")
	}
	seen := make(map[string]bool)
	for _, f := range idx.WeightMap {
		if !seen[f] {
			seen[f] = true
			files = append(files, f)
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
		data, err := mmapReadOnly(filepath.Join(dir, fn))
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
// Platform: works on darwin/linux/bsd via syscall.Mmap. Not supported
// on Windows; the embed package's primary deployments are macOS/Linux.
func OpenSafetensorsMmap(path string) (*SafetensorsFile, error) {
	data, err := mmapReadOnly(path)
	if err != nil {
		return nil, err
	}
	sf, err := parseSafetensors(data)
	if err != nil {
		_ = syscall.Munmap(data)
		return nil, err
	}
	sf.mmapped = [][]byte{data}
	// Finalizer guards the common case where callers forget Close().
	runtime.SetFinalizer(sf, finalizeMmaps)
	return sf, nil
}

// mmapReadOnly opens path and returns a read-only MAP_PRIVATE mapping of its
// whole contents. The fd is closed before returning (the mapping survives it).
func mmapReadOnly(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close() // fd no longer needed after mmap

	st, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat %s: %w", path, err)
	}
	sz := st.Size()
	if sz < 8 {
		return nil, fmt.Errorf("safetensors %s: file too small (%d bytes)", path, sz)
	}
	if sz > int64(int(^uint(0)>>1)) {
		return nil, fmt.Errorf("safetensors %s: file too large for this platform (%d bytes)", path, sz)
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(sz), syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err != nil {
		return nil, fmt.Errorf("mmap %s: %w", path, err)
	}
	return data, nil
}

// finalizeMmaps is the SetFinalizer callback for mmap-backed files: munmap
// every region. Close does the same eagerly.
func finalizeMmaps(s *SafetensorsFile) {
	for _, m := range s.mmapped {
		_ = syscall.Munmap(m)
	}
	s.mmapped = nil
}

// Close releases the underlying mmap, if any. No-op on heap-loaded
// SafetensorsFile (OpenSafetensors / OpenSafetensorsFromFS). Idempotent.
//
// After Close, any Tensor() returns alias unmapped memory — accessing
// them is undefined behavior. Callers MUST stop using the model and any
// downstream objects that hold tensor data before calling Close.
func (sf *SafetensorsFile) Close() error {
	if len(sf.mmapped) == 0 {
		return nil
	}
	regions := sf.mmapped
	sf.mmapped = nil
	runtime.SetFinalizer(sf, nil)
	var firstErr error
	for _, m := range regions {
		if err := syscall.Munmap(m); err != nil && firstErr == nil {
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
		return nil, errors.New("safetensors: file too short for header length prefix")
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

	if uint64(len(data)) < 8+headerLen {
		return nil, fmt.Errorf("safetensors: file truncated (header claims %d bytes, only %d available)",
			headerLen, len(data)-8)
	}

	headerBytes := data[8 : 8+headerLen]
	payload := data[8+headerLen:]

	var raw rawHeader
	if err := json.Unmarshal(headerBytes, &raw); err != nil {
		return nil, fmt.Errorf("safetensors: parse header JSON: %w", err)
	}

	tensors := make(map[string]Tensor, len(raw))
	for name, rawJSON := range raw {
		if name == "__metadata__" {
			continue
		}
		var t rawTensor
		if err := json.Unmarshal(rawJSON, &t); err != nil {
			return nil, fmt.Errorf("safetensors: parse tensor %q: %w", name, err)
		}
		if t.DataOffsets[0] < 0 || t.DataOffsets[1] > len(payload) || t.DataOffsets[0] > t.DataOffsets[1] {
			return nil, fmt.Errorf("safetensors: tensor %q has invalid offsets %v (payload size %d)",
				name, t.DataOffsets, len(payload))
		}
		tensors[name] = Tensor{
			Name:  name,
			DType: t.DType,
			Shape: t.Shape,
			raw:   payload[t.DataOffsets[0]:t.DataOffsets[1]],
		}
	}

	return &SafetensorsFile{data: data, tensors: tensors}, nil
}

// Tensor returns the named tensor or an error if not present.
func (f *SafetensorsFile) Tensor(name string) (Tensor, error) {
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

// Float32s returns the tensor data as []float32. Requires DType "F32".
// The returned slice aliases the file's []byte; do not mutate.
// This assumes little-endian host byte order (x86, arm64).
func (t Tensor) Float32s() ([]float32, error) {
	if t.DType != "F32" {
		return nil, fmt.Errorf("tensor %q: expected F32, got %s", t.Name, t.DType)
	}
	if len(t.raw)%4 != 0 {
		return nil, fmt.Errorf("tensor %q: F32 raw size %d not a multiple of 4", t.Name, len(t.raw))
	}
	if len(t.raw) == 0 {
		return nil, nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&t.raw[0])), len(t.raw)/4), nil
}

// Float64s returns the tensor data as []float64. Requires DType "F64".
func (t Tensor) Float64s() ([]float64, error) {
	if t.DType != "F64" {
		return nil, fmt.Errorf("tensor %q: expected F64, got %s", t.Name, t.DType)
	}
	if len(t.raw)%8 != 0 {
		return nil, fmt.Errorf("tensor %q: F64 raw size %d not a multiple of 8", t.Name, len(t.raw))
	}
	if len(t.raw) == 0 {
		return nil, nil
	}
	return unsafe.Slice((*float64)(unsafe.Pointer(&t.raw[0])), len(t.raw)/8), nil
}

// Int64s returns the tensor data as []int64. Requires DType "I64".
func (t Tensor) Int64s() ([]int64, error) {
	if t.DType != "I64" {
		return nil, fmt.Errorf("tensor %q: expected I64, got %s", t.Name, t.DType)
	}
	if len(t.raw)%8 != 0 {
		return nil, fmt.Errorf("tensor %q: I64 raw size %d not a multiple of 8", t.Name, len(t.raw))
	}
	if len(t.raw) == 0 {
		return nil, nil
	}
	return unsafe.Slice((*int64)(unsafe.Pointer(&t.raw[0])), len(t.raw)/8), nil
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
	if t.DType != "BF16" {
		return nil, fmt.Errorf("tensor %q: expected BF16, got %s", t.Name, t.DType)
	}
	if len(t.raw)%2 != 0 {
		return nil, fmt.Errorf("tensor %q: BF16 raw size %d not a multiple of 2", t.Name, len(t.raw))
	}
	n := len(t.raw) / 2
	out := make([]float32, n)
	for i := 0; i < n; i++ {
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
	if t.DType != "F16" {
		return nil, fmt.Errorf("tensor %q: expected F16, got %s", t.Name, t.DType)
	}
	if len(t.raw)%2 != 0 {
		return nil, fmt.Errorf("tensor %q: F16 raw size %d not a multiple of 2", t.Name, len(t.raw))
	}
	n := len(t.raw) / 2
	out := make([]float32, n)
	for i := 0; i < n; i++ {
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

// Elements returns the total number of elements (product of shape).
func (t Tensor) Elements() int {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	return n
}
