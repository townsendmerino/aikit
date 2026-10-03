package embed

import (
	"fmt"
	"math"
	"os"
	"runtime"

	"github.com/townsendmerino/aikit/internal/cursor"
	"github.com/townsendmerino/aikit/mmap"
)

// GGUF reader — the llama.cpp container format that makes
// quantized models laptop-runnable. This parses the header, metadata key-values
// (which carry the architecture config), and the tensor directory, and
// dequantizes the common block types to float32: F32, F16, Q8_0, Q4_0, Q5_0,
// the K-quants Q2_K/Q3_K/Q4_K/Q5_K/Q6_K, and the codebook quants IQ4_NL/IQ4_XS
// plus the grid-codebook IQ2_S/IQ3_S (so Q2_K / Q3_K_M / Q4_K_M / Q5_K_M / IQ4_XS
// / IQ3_S / IQ2_S-style mixes load). Tensor returns a clear error for an
// unimplemented type (the remaining IQ1*/IQ2_XXS/IQ2_XS/IQ3_XXS grid quants).
//
// Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

const ggufMagic = 0x46554747 // "GGUF" little-endian

// ggml tensor (quantization) types.
const (
	ggmlTypeF32   uint32 = 0
	ggmlTypeF16   uint32 = 1
	ggmlTypeQ4_0  uint32 = 2
	ggmlTypeQ5_0  uint32 = 6
	ggmlTypeQ8_0  uint32 = 8
	ggmlTypeQ2_K  uint32 = 10
	ggmlTypeQ3_K  uint32 = 11
	ggmlTypeQ4_K  uint32 = 12
	ggmlTypeQ5_K  uint32 = 13
	ggmlTypeQ6_K  uint32 = 14
	ggmlTypeIQ4NL uint32 = 20
	ggmlTypeIQ3S  uint32 = 21
	ggmlTypeIQ2S  uint32 = 22
	ggmlTypeIQ4XS uint32 = 23
	ggmlTypeMXFP4 uint32 = 39 // OCP Microscaling FP4: 32-elem block, e8m0 scale + 16 e2m1 nibble-pairs (gpt-oss)
)

// qkK is the K-quant super-block size (elements per super-block).
const qkK = 256

// gguf metadata value types.
const (
	ggufUint8 uint32 = iota
	ggufInt8
	ggufUint16
	ggufInt16
	ggufUint32
	ggufInt32
	ggufFloat32
	ggufBool
	ggufString
	ggufArray
	ggufUint64
	ggufInt64
	ggufFloat64
)

type ggufTensorInfo struct {
	dims   []uint64
	typ    uint32
	offset uint64 // relative to the data section start
	sec    []byte // the data section offset is relative to: nil for the file's own (GGUFFile.data), its shard's in a split set
}

// GGUFFile is a parsed GGUF checkpoint: its metadata (architecture config,
// tokenizer, …) and a directory of dequantizable tensors over the mapped data.
type GGUFFile struct {
	Metadata map[string]any
	tensors  map[string]ggufTensorInfo
	data     []byte   // the tensor-data section (file bytes after the aligned header); nil for a split set
	mmap     []byte   // full mmap region iff opened via OpenGGUFMmap; nil for OpenGGUF
	mmaps    [][]byte // every shard's mmap region iff opened via OpenGGUFSplitMmap
}

// section is the data section a tensor's offset is relative to.
func (g *GGUFFile) section(info ggufTensorInfo) []byte {
	if info.sec != nil {
		return info.sec
	}
	return g.data
}

// gcur is a little-endian cursor over a byte slice with bounds checks — the
// shared primitive (internal/cursor.Cursor) plus GGUF-specific readers.
type gcur struct {
	cursor.Cursor
	depth int // nested-array recursion depth (see ggufMaxArrayDepth)
}

func newGcur(raw []byte) *gcur {
	return &gcur{B: raw, Context: "gguf", Errorf: errFormatf}
}

// ggufHintCap bounds a map-preallocation hint. No real GGUF has this many KV or
// tensor entries (a large model has hundreds of KV, thousands of tensors), and
// each hinted map slot costs ~48–64 bytes of eager bucket allocation, so an
// unbounded hint from a hostile count is its own amplification vector.
const ggufHintCap = 1 << 16

// hintLen clamps an untrusted entry count to a safe make() preallocation size.
// minEntryBytes is the smallest possible encoded size of one entry (a KV pair is
// ≥13 bytes, a tensor descriptor ≥24), so remaining/minEntryBytes bounds how many
// could even exist — clamping by raw remaining bytes assumed ≥1 byte/entry and
// still let a 100 MB file claiming billions of entries drive a multi-GB map
// prealloc before the loop hit EOF. Also capped by ggufHintCap. The loop stays
// bounded by the true count and stops at EOF regardless.
func (c *gcur) hintLen(n uint64, minEntryBytes int) int {
	if n > ggufHintCap {
		n = ggufHintCap
	}
	if fit := uint64(c.Remaining() / minEntryBytes); n > fit {
		n = fit
	}
	return int(n)
}

// str reads a gguf string: uint64 length + raw bytes.
func (c *gcur) str() string {
	n := int(c.U64())
	raw := c.Bytes(n)
	if raw == nil {
		return ""
	}
	return string(raw)
}

// ggufArrayPrealloc caps the eager capacity of a metadata array's []any. Small so
// nested arrays can't compound into an O(input²) preallocation (see value); append
// grows past it for the rare genuinely large flat array (e.g. a tokenizer vocab).
const ggufArrayPrealloc = 64

// ggufMaxArrayDepth caps metadata array-of-array nesting. The allocation blowup
// is bounded by ggufArrayPrealloc, but recursion DEPTH is still ~input/12 (each
// nesting level is a 4-byte element type + 8-byte count), so a ~50–150 MB file
// of repeated (et=array, n=…) headers would drive millions of value() frames
// past Go's ~1 GB goroutine-stack limit and abort the process with "goroutine
// stack exceeds" — not a recoverable panic, so recover() couldn't uphold the
// "error or succeed, never crash" parse contract. Real metadata nests 1–2 deep;
// 128 mirrors encoding/json's nesting cap.
const ggufMaxArrayDepth = 128

// value reads one metadata value of the given type (arrays recurse).
func (c *gcur) value(vtype uint32) any {
	switch vtype {
	case ggufUint8:
		return c.U8()
	case ggufInt8:
		return int8(c.U8())
	case ggufUint16:
		return c.U16()
	case ggufInt16:
		return int16(c.U16())
	case ggufUint32:
		return c.U32()
	case ggufInt32:
		return int32(c.U32())
	case ggufFloat32:
		return c.F32()
	case ggufBool:
		return c.U8() != 0
	case ggufString:
		return c.str()
	case ggufUint64:
		return c.U64()
	case ggufInt64:
		return int64(c.U64())
	case ggufFloat64:
		return c.F64()
	case ggufArray:
		// Bound recursion depth before descending: an array-of-arrays chain
		// nests one value() frame per level and would otherwise blow the
		// goroutine stack (see ggufMaxArrayDepth).
		if c.depth >= ggufMaxArrayDepth {
			c.Err = errFormatf("gguf: metadata array nesting exceeds %d levels", ggufMaxArrayDepth)
			return nil
		}
		c.depth++
		defer func() { c.depth-- }()
		et := c.U32()
		n := c.U64()
		// Each array element is ≥1 byte, so a count beyond the remaining input is
		// impossible — reject it rather than pre-allocate (or wrap int and panic).
		if n > uint64(c.Remaining()) {
			c.Err = errFormatf("gguf: array length %d exceeds %d remaining bytes", n, c.Remaining())
			return nil
		}
		// Cap the EAGER preallocation: n is bounded by the remaining bytes, but for
		// an array of arrays that bound recurses — every nesting level can claim a
		// count near the remaining input, and the nesting depth is itself ~input/12,
		// so make([]any, 0, n) at each level drives O(input²) allocation (a hostile
		// nested-array file that parses in seconds — a fuzz "deadline exceeded" slow
		// path). append grows to the true element count, so a small fixed prealloc
		// keeps total allocation linear in the bytes actually consumed.
		arr := make([]any, 0, min(n, ggufArrayPrealloc))
		for i := uint64(0); i < n && c.Err == nil; i++ {
			arr = append(arr, c.value(et))
		}
		return arr
	default:
		c.Err = errFormatf("gguf: unknown metadata value type %d", vtype)
		return nil
	}
}

// OpenGGUF reads and parses a .gguf file. The whole file is read into memory
// (heap); tensor data is dequantized into fresh slices by Tensor, so callers
// needn't retain the file. For large checkpoints prefer OpenGGUFMmap, which
// maps the file instead of heap-copying it — the raw quantized bytes then live
// in reclaimable page cache rather than the Go heap, and metadata-only readers
// (e.g. a tokenizer) never page in the weights at all.
func OpenGGUF(path string) (*GGUFFile, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("gguf: %w", err)
	}
	return parseGGUF(raw)
}

// OpenGGUFBytes parses a GGUF model from an in-memory byte slice — the same
// path OpenGGUF uses after reading the file, minus the filesystem. The slice
// is RETAINED by the returned *GGUFFile (tensor data aliases it, not copied),
// so the caller must keep raw alive until done; Tensor still dequantizes into
// fresh slices, so values read out survive independently. Nothing is mapped,
// so Close is a no-op here (no munmap).
//
// Use this for //go:embed-ed or downloaded-in-memory models, and in read-only
// environments with no writable temp dir — it avoids spilling the bytes to a
// temp .gguf just to call OpenGGUFMmap(path).
func OpenGGUFBytes(raw []byte) (*GGUFFile, error) {
	return parseGGUF(raw)
}

// OpenGGUFMmap memory-maps a .gguf file (read-only, MAP_PRIVATE) and parses it,
// so the raw quantized bytes are file-backed page cache, not heap. Metadata
// strings are copied during parse and Tensor dequantizes into fresh slices, so
// nothing aliases the mapping — call Close (or let the finalizer run) once the
// tensors have been read to munmap. Platform: true mmap on unix; on non-unix
// targets (Windows) it falls back to a heap read, same as OpenSafetensorsMmap.
func OpenGGUFMmap(path string) (*GGUFFile, error) {
	data, err := mmap.MapReadOnly(path)
	if err != nil {
		return nil, fmt.Errorf("gguf: %w", err)
	}
	g, err := parseGGUF(data)
	if err != nil {
		_ = mmap.Unmap(data)
		return nil, err
	}
	g.mmap = data
	runtime.SetFinalizer(g, finalizeGGUFMmap)
	return g, nil
}

// Close releases the mmap backing a GGUFFile opened via OpenGGUFMmap; its tensor
// data must not be read afterward. No-op for OpenGGUF (heap-backed). Safe to
// call more than once.
func (g *GGUFFile) Close() error {
	var err error
	if g.mmaps != nil {
		// A split set: forget each tensor's shard section first, so a read after Close fails on an empty section
		// the way a single file's does (g.data = nil below) instead of touching an unmapped page.
		for name, info := range g.tensors {
			info.sec = nil
			g.tensors[name] = info
		}
		for _, m := range g.mmaps {
			if e := mmap.Unmap(m); e != nil && err == nil {
				err = e
			}
		}
		g.mmaps = nil
	}
	if g.mmap != nil {
		if e := mmap.Unmap(g.mmap); e != nil && err == nil {
			err = e
		}
		g.mmap = nil
	}
	g.data = nil
	return err
}

func finalizeGGUFMmap(g *GGUFFile) { _ = g.Close() }

// parseGGUF parses the header, metadata key-values and tensor directory from an
// already-loaded (heap or mmap) byte slice. The data section is referenced in
// place via GGUFFile.data, so raw must outlive the GGUFFile's tensor reads.
func parseGGUF(raw []byte) (*GGUFFile, error) {
	c := newGcur(raw)
	if c.U32() != ggufMagic {
		return nil, errFormatf("gguf: bad magic (not a GGUF file)")
	}
	version := c.U32()
	if version != 2 && version != 3 {
		return nil, errFormatf("gguf: unsupported version %d (want 2 or 3)", version)
	}
	tensorCount := c.U64()
	kvCount := c.U64()

	// kvCount/tensorCount are untrusted: a hostile header can claim billions of
	// entries. Clamp the make() hints to what the input could hold; the loops
	// below still run to the true count and stop at EOF.
	g := &GGUFFile{Metadata: make(map[string]any, c.hintLen(kvCount, 13)), tensors: make(map[string]ggufTensorInfo, c.hintLen(tensorCount, 24))}
	for i := uint64(0); i < kvCount && c.Err == nil; i++ {
		key := c.str()
		vtype := c.U32()
		g.Metadata[key] = c.value(vtype)
	}
	if c.Err != nil {
		return nil, c.Err
	}

	for i := uint64(0); i < tensorCount && c.Err == nil; i++ {
		name := c.str()
		nd := int(c.U32())
		// Each dim is a u64; a count beyond remaining/8 can't be satisfied, so
		// reject it rather than make([]uint64, huge) and OOM.
		if nd < 0 || nd > c.Remaining()/8 {
			c.Err = errFormatf("gguf: tensor %q dim count %d exceeds remaining input", name, nd)
			break
		}
		dims := make([]uint64, nd)
		for d := range nd {
			dims[d] = c.U64()
		}
		typ := c.U32()
		off := c.U64()
		g.tensors[name] = ggufTensorInfo{dims: dims, typ: typ, offset: off}
	}
	if c.Err != nil {
		return nil, c.Err
	}

	// The tensor data section begins at the next `alignment` boundary after the
	// header (default 32; overridable via general.alignment).
	align := uint64(32)
	if a, ok := g.Uint("general.alignment"); ok && a > 0 {
		align = a
	}
	start := uint64(c.Pos)
	if start%align != 0 {
		start += align - start%align
	}
	if start > uint64(len(raw)) {
		return nil, errFormatf("gguf: data section start %d past EOF %d", start, len(raw))
	}
	g.data = raw[start:]
	return g, nil
}

// Names returns the tensor names present in the file.
func (g *GGUFFile) Names() []string {
	out := make([]string, 0, len(g.tensors))
	for n := range g.tensors {
		out = append(out, n)
	}
	return out
}

// Has reports whether a tensor is present.
func (g *GGUFFile) Has(name string) bool { _, ok := g.tensors[name]; return ok }

// Dims returns a tensor's dimensions (GGUF order: dims[0] innermost = input
// features) without reading or dequantizing its data — for cheap shape probes
// (e.g. deriving vocab size from the embedding) on big quantized tensors.
func (g *GGUFFile) Dims(name string) ([]int, bool) {
	info, ok := g.tensors[name]
	if !ok {
		return nil, false
	}
	dims := make([]int, len(info.dims))
	for i, d := range info.dims {
		// info.dims are untrusted uint64. A dim ≥ 2^63 wraps to a negative int,
		// which a caller feeding the result to make() or a shape product would
		// mishandle — report the tensor as unusable rather than hand back a
		// negative dim. (RowDequantizer validates dims against the data section;
		// Dims is the cheap metadata probe, so it just rejects the overflow.)
		if d > uint64(math.MaxInt) {
			return nil, false
		}
		dims[i] = int(d)
	}
	return dims, true
}

// Tensor dequantizes a tensor to float32 and returns its dimensions in GGUF
// order (dims[0] is the fastest/innermost = input features; dims[1] the row
// count = output features). The f32 data is row-major over the outer dims, i.e.
// for a 2-D weight it is [out, in] — the layout decoder.weightMat expects.
func (g *GGUFFile) Tensor(name string) (dims []int, data []float32, err error) {
	dims, into, err := g.RowDequantizer(name)
	if err != nil {
		return nil, nil, err
	}
	n := 1
	for _, d := range dims {
		n *= d
	}
	data = make([]float32, n)
	if err := into(0, data); err != nil {
		return nil, nil, fmt.Errorf("gguf: tensor %q: %w", name, err)
	}
	return dims, data, nil
}

// RowDequantizer resolves tensor `name` once and returns its dims plus a closure
// that dequantizes the element range [start, start+len(dst)) into dst. For
// quantized types start and len(dst) must be (super-)block-aligned — always true
// when dequantizing whole rows of a per-row-quantized weight (cols is a multiple
// of the block size). This lets a loader stream a big tensor row-by-row into a
// small scratch and re-quantize each row immediately, instead of materializing
// the whole f32 matrix (the load-time memory-bandwidth win). Tensor is the
// whole-tensor convenience built on top, so both share one dequant path.
func (g *GGUFFile) RowDequantizer(name string) (dims []int, into func(start int, dst []float32) error, err error) {
	info, ok := g.tensors[name]
	if !ok {
		return nil, nil, fmt.Errorf("gguf: tensor %q not found", name)
	}
	// ∏dims feeds make([]float32, n) and the byte-size arithmetic, and every dim
	// is an untrusted uint64. A hostile tensor can claim dims whose product
	// overflows int (wrapping the byte check and OOM-ing the make). Bound it:
	// the densest supported type is Q2_K at ~0.33 bytes/element (IQ2_S ~0.32),
	// so a tensor's element count can't exceed 4×|data section| — using 2× here
	// falsely rejected a legitimate Q2_K/IQ2_S tensor occupying >~65% of the
	// data section (tensorBytes would have validated it exactly). Check before
	// each multiply so the product itself can never overflow.
	sec := g.section(info)
	maxElems := 4*len(sec) + qkK
	n := 1
	dims = make([]int, len(info.dims))
	for i, d := range info.dims {
		if d > uint64(maxElems) {
			return nil, nil, fmt.Errorf("gguf: tensor %q dim %d (%d) exceeds data section (%d bytes)", name, i, d, len(sec))
		}
		di := int(d)
		if di != 0 && n > maxElems/di {
			return nil, nil, fmt.Errorf("gguf: tensor %q element count exceeds data section (%d bytes)", name, len(sec))
		}
		dims[i] = di
		n *= di
	}
	blockElems, ok := ggmlBlockElems(info.typ)
	if !ok {
		return nil, nil, fmt.Errorf("gguf: tensor %q unsupported ggml type %d (have F32/F16/Q8_0/Q4_0/Q5_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/IQ4_NL/IQ4_XS/IQ2_S/IQ3_S)", name, info.typ)
	}
	raw, err := g.tensorBytes(info, n)
	if err != nil {
		return nil, nil, fmt.Errorf("gguf: tensor %q: %w", name, err)
	}
	into = func(start int, dst []float32) error {
		// H6: raw aliases g's mmap region (OpenGGUFMmap installs a finalizer
		// that munmaps it). The closure captures raw, not g, so g could be
		// unreachable — and thus finalized/unmapped — while dequantRange still
		// reads raw, giving a SIGSEGV. Keep g alive across the read. This also
		// covers Tensor(), which dequantizes through this same closure.
		defer runtime.KeepAlive(g)
		if start < 0 || start+len(dst) > n {
			return fmt.Errorf("gguf: tensor %q range [%d:%d] out of [0:%d]", name, start, start+len(dst), n)
		}
		if blockElems > 1 && (start%blockElems != 0 || len(dst)%blockElems != 0) {
			return fmt.Errorf("gguf: tensor %q range [%d:%d] not aligned to block %d", name, start, start+len(dst), blockElems)
		}
		dequantRange(info.typ, raw, start, dst, blockElems)
		return nil
	}
	return dims, into, nil
}

// ggmlBlockElems returns the number of elements per (super-)block for a ggml type
// (1 for the unquantized F32/F16), and whether the type is supported.
func ggmlBlockElems(typ uint32) (int, bool) {
	switch typ {
	case ggmlTypeF32, ggmlTypeF16:
		return 1, true
	case ggmlTypeQ8_0, ggmlTypeQ4_0, ggmlTypeQ5_0, ggmlTypeIQ4NL, ggmlTypeMXFP4:
		return 32, true
	case ggmlTypeQ2_K, ggmlTypeQ3_K, ggmlTypeQ4_K, ggmlTypeQ5_K, ggmlTypeQ6_K, ggmlTypeIQ4XS, ggmlTypeIQ2S, ggmlTypeIQ3S:
		return qkK, true
	default:
		return 0, false
	}
}

// tensorBytes returns the raw bytes for a tensor, validating the element count
// against the type's block size.
func (g *GGUFFile) tensorBytes(info ggufTensorInfo, n int) ([]byte, error) {
	var nbytes int
	switch info.typ {
	case ggmlTypeF32:
		nbytes = n * 4
	case ggmlTypeF16:
		nbytes = n * 2
	case ggmlTypeQ8_0, ggmlTypeQ4_0, ggmlTypeQ5_0, ggmlTypeIQ4NL, ggmlTypeMXFP4:
		if n%32 != 0 {
			return nil, fmt.Errorf("element count %d not a multiple of 32 (block size)", n)
		}
		blocks := n / 32
		switch info.typ {
		case ggmlTypeQ8_0:
			nbytes = blocks * 34 // 2-byte f16 scale + 32 int8
		case ggmlTypeQ4_0, ggmlTypeIQ4NL:
			nbytes = blocks * 18 // 2-byte f16 scale + 16 packed nibbles
		case ggmlTypeMXFP4:
			nbytes = blocks * 17 // 1-byte e8m0 scale + 16 packed e2m1 nibbles
		default: // Q5_0
			nbytes = blocks * 22 // 2-byte f16 scale + 4-byte high bits + 16 packed nibbles
		}
	case ggmlTypeQ2_K, ggmlTypeQ3_K, ggmlTypeQ4_K, ggmlTypeQ5_K, ggmlTypeQ6_K, ggmlTypeIQ4XS, ggmlTypeIQ2S, ggmlTypeIQ3S:
		if n%qkK != 0 {
			return nil, fmt.Errorf("element count %d not a multiple of %d (super-block)", n, qkK)
		}
		nSuperblocks := n / qkK
		switch info.typ {
		case ggmlTypeIQ2S:
			nbytes = nSuperblocks * 82 // d(f16) + qs[32] + signs[32] + qh[8] + scales[8]
		case ggmlTypeIQ3S:
			nbytes = nSuperblocks * 110 // d(f16) + qs[64] + qh[8] + signs[32] + scales[4]
		case ggmlTypeIQ4XS:
			nbytes = nSuperblocks * 136 // d(f16) + scales_h(u16) + scales_l[4] + qs[128]
		case ggmlTypeQ2_K:
			nbytes = nSuperblocks * 84 // scales[16] + qs[64] + d + dmin (f16 each)
		case ggmlTypeQ3_K:
			nbytes = nSuperblocks * 110 // hmask[32] + qs[64] + scales[12] + d(f16)
		case ggmlTypeQ4_K:
			nbytes = nSuperblocks * 144 // d + dmin (f16 each) + scales[12] + qs[128]
		case ggmlTypeQ5_K:
			nbytes = nSuperblocks * 176 // d + dmin (f16 each) + scales[12] + qh[32] + qs[128]
		default: // Q6_K
			nbytes = nSuperblocks * 210 // ql[128] + qh[64] + scales[16] + d(f16)
		}
	default:
		return nil, fmt.Errorf("unsupported ggml type %d", info.typ)
	}
	// info.offset is an untrusted uint64; compare without adding so a near-2^64
	// offset can't wrap the sum past this guard and panic the slice below.
	sec := g.section(info)
	if info.offset > uint64(len(sec)) || uint64(nbytes) > uint64(len(sec))-info.offset {
		return nil, fmt.Errorf("data range [%d:+%d] past section end %d", info.offset, nbytes, len(sec))
	}
	return sec[info.offset : info.offset+uint64(nbytes)], nil
}

// --- typed metadata accessors ---

// Str returns a string metadata value.
func (g *GGUFFile) Str(key string) (string, bool) {
	v, ok := g.Metadata[key].(string)
	return v, ok
}

// Uint returns an integer metadata value, accepting any of GGUF's int widths.
func (g *GGUFFile) Uint(key string) (uint64, bool) {
	switch v := g.Metadata[key].(type) {
	case uint8:
		return uint64(v), true
	case uint16:
		return uint64(v), true
	case uint32:
		return uint64(v), true
	case uint64:
		return v, true
	case int8:
		return uint64(v), true
	case int16:
		return uint64(v), true
	case int32:
		return uint64(v), true
	case int64:
		return uint64(v), true
	default:
		return 0, false
	}
}

// Float returns a floating-point metadata value (f32 or f64).
func (g *GGUFFile) Float(key string) (float64, bool) {
	switch v := g.Metadata[key].(type) {
	case float32:
		return float64(v), true
	case float64:
		return v, true
	default:
		return 0, false
	}
}
