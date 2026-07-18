package embed

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"runtime"

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
)

// kvaluesIQ4NL is the 16-entry non-linear codebook shared by IQ4_NL and IQ4_XS:
// a 4-bit code indexes one of these int8 levels (ggml's kvalues_iq4nl), scaled by
// the block's f16 (and, for IQ4_XS, per-sub-block) scale. Unlike the linear
// Q4_* quants there is no `code-8` recentering — the codebook is the mapping.
var kvaluesIQ4NL = [16]int8{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}

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
}

// GGUFFile is a parsed GGUF checkpoint: its metadata (architecture config,
// tokenizer, …) and a directory of dequantizable tensors over the mapped data.
type GGUFFile struct {
	Metadata map[string]any
	tensors  map[string]ggufTensorInfo
	data     []byte // the tensor-data section (file bytes after the aligned header)
	mmap     []byte // full mmap region iff opened via OpenGGUFMmap; nil for OpenGGUF
}

// gcur is a little-endian cursor over a byte slice with bounds checks.
type gcur struct {
	b     []byte
	pos   int
	err   error
	depth int // nested-array recursion depth (see ggufMaxArrayDepth)
}

func (c *gcur) need(n int) bool {
	if c.err != nil {
		return false
	}
	// n<0 guards a length field that overflowed int when converted from uint64
	// (a hostile string/array length ≥ 2^63); compare against the remaining
	// span without adding, so c.pos+n can't itself overflow.
	if n < 0 || n > len(c.b)-c.pos {
		c.err = errFormatf("gguf: unexpected EOF (need %d at %d of %d)", n, c.pos, len(c.b))
		return false
	}
	return true
}

// remaining is the number of unread bytes — the ceiling on any element count
// or length, since every element/byte consumes at least one byte of input.
func (c *gcur) remaining() int { return len(c.b) - c.pos }

// hintLen clamps an untrusted element count to a safe make() preallocation size:
// never more than the bytes left. The parse loop stays bounded by the true count
// and stops at EOF, so this only prevents a hostile count from driving a giant
// allocation before that EOF is ever reached.
func (c *gcur) hintLen(n uint64) int {
	if r := c.remaining(); n > uint64(r) {
		return r
	}
	return int(n)
}

func (c *gcur) u8() uint8 {
	if !c.need(1) {
		return 0
	}
	v := c.b[c.pos]
	c.pos++
	return v
}
func (c *gcur) u16() uint16 {
	if !c.need(2) {
		return 0
	}
	v := binary.LittleEndian.Uint16(c.b[c.pos:])
	c.pos += 2
	return v
}
func (c *gcur) u32() uint32 {
	if !c.need(4) {
		return 0
	}
	v := binary.LittleEndian.Uint32(c.b[c.pos:])
	c.pos += 4
	return v
}
func (c *gcur) u64() uint64 {
	if !c.need(8) {
		return 0
	}
	v := binary.LittleEndian.Uint64(c.b[c.pos:])
	c.pos += 8
	return v
}
func (c *gcur) f32() float32 { return math.Float32frombits(c.u32()) }
func (c *gcur) f64() float64 { return math.Float64frombits(c.u64()) }

// str reads a gguf string: uint64 length + raw bytes.
func (c *gcur) str() string {
	n := int(c.u64())
	if !c.need(n) {
		return ""
	}
	s := string(c.b[c.pos : c.pos+n])
	c.pos += n
	return s
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
		return c.u8()
	case ggufInt8:
		return int8(c.u8())
	case ggufUint16:
		return c.u16()
	case ggufInt16:
		return int16(c.u16())
	case ggufUint32:
		return c.u32()
	case ggufInt32:
		return int32(c.u32())
	case ggufFloat32:
		return c.f32()
	case ggufBool:
		return c.u8() != 0
	case ggufString:
		return c.str()
	case ggufUint64:
		return c.u64()
	case ggufInt64:
		return int64(c.u64())
	case ggufFloat64:
		return c.f64()
	case ggufArray:
		// Bound recursion depth before descending: an array-of-arrays chain
		// nests one value() frame per level and would otherwise blow the
		// goroutine stack (see ggufMaxArrayDepth).
		if c.depth >= ggufMaxArrayDepth {
			c.err = fmt.Errorf("gguf: metadata array nesting exceeds %d levels", ggufMaxArrayDepth)
			return nil
		}
		c.depth++
		defer func() { c.depth-- }()
		et := c.u32()
		n := c.u64()
		// Each array element is ≥1 byte, so a count beyond the remaining input is
		// impossible — reject it rather than pre-allocate (or wrap int and panic).
		if n > uint64(c.remaining()) {
			c.err = fmt.Errorf("gguf: array length %d exceeds %d remaining bytes", n, c.remaining())
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
		for i := uint64(0); i < n && c.err == nil; i++ {
			arr = append(arr, c.value(et))
		}
		return arr
	default:
		c.err = fmt.Errorf("gguf: unknown metadata value type %d", vtype)
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
	if g.mmap == nil {
		return nil
	}
	err := mmap.Unmap(g.mmap)
	g.mmap = nil
	g.data = nil
	return err
}

func finalizeGGUFMmap(g *GGUFFile) { _ = g.Close() }

// parseGGUF parses the header, metadata key-values and tensor directory from an
// already-loaded (heap or mmap) byte slice. The data section is referenced in
// place via GGUFFile.data, so raw must outlive the GGUFFile's tensor reads.
func parseGGUF(raw []byte) (*GGUFFile, error) {
	c := &gcur{b: raw}
	if c.u32() != ggufMagic {
		return nil, errFormatf("gguf: bad magic (not a GGUF file)")
	}
	version := c.u32()
	if version != 2 && version != 3 {
		return nil, errFormatf("gguf: unsupported version %d (want 2 or 3)", version)
	}
	tensorCount := c.u64()
	kvCount := c.u64()

	// kvCount/tensorCount are untrusted: a hostile header can claim billions of
	// entries. Clamp the make() hints to what the input could hold; the loops
	// below still run to the true count and stop at EOF.
	g := &GGUFFile{Metadata: make(map[string]any, c.hintLen(kvCount)), tensors: make(map[string]ggufTensorInfo, c.hintLen(tensorCount))}
	for i := uint64(0); i < kvCount && c.err == nil; i++ {
		key := c.str()
		vtype := c.u32()
		g.Metadata[key] = c.value(vtype)
	}
	if c.err != nil {
		return nil, c.err
	}

	for i := uint64(0); i < tensorCount && c.err == nil; i++ {
		name := c.str()
		nd := int(c.u32())
		// Each dim is a u64; a count beyond remaining/8 can't be satisfied, so
		// reject it rather than make([]uint64, huge) and OOM.
		if nd < 0 || nd > c.remaining()/8 {
			c.err = fmt.Errorf("gguf: tensor %q dim count %d exceeds remaining input", name, nd)
			break
		}
		dims := make([]uint64, nd)
		for d := range nd {
			dims[d] = c.u64()
		}
		typ := c.u32()
		off := c.u64()
		g.tensors[name] = ggufTensorInfo{dims: dims, typ: typ, offset: off}
	}
	if c.err != nil {
		return nil, c.err
	}

	// The tensor data section begins at the next `alignment` boundary after the
	// header (default 32; overridable via general.alignment).
	align := uint64(32)
	if a, ok := g.Uint("general.alignment"); ok && a > 0 {
		align = a
	}
	start := uint64(c.pos)
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
	// even the densest supported type packs ≥ ~0.5 bytes/element, so a tensor's
	// element count can't exceed 2×|data section|. Check before each multiply so
	// the product itself can never overflow; tensorBytes does the exact check.
	maxElems := 2*len(g.data) + qkK
	n := 1
	dims = make([]int, len(info.dims))
	for i, d := range info.dims {
		if d > uint64(maxElems) {
			return nil, nil, fmt.Errorf("gguf: tensor %q dim %d (%d) exceeds data section (%d bytes)", name, i, d, len(g.data))
		}
		di := int(d)
		if di != 0 && n > maxElems/di {
			return nil, nil, fmt.Errorf("gguf: tensor %q element count exceeds data section (%d bytes)", name, len(g.data))
		}
		dims[i] = di
		n *= di
	}
	bs, ok := ggmlBlockElems(info.typ)
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
		if bs > 1 && (start%bs != 0 || len(dst)%bs != 0) {
			return fmt.Errorf("gguf: tensor %q range [%d:%d] not aligned to block %d", name, start, start+len(dst), bs)
		}
		dequantRange(info.typ, raw, start, dst, bs)
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
	case ggmlTypeQ8_0, ggmlTypeQ4_0, ggmlTypeQ5_0, ggmlTypeIQ4NL:
		return 32, true
	case ggmlTypeQ2_K, ggmlTypeQ3_K, ggmlTypeQ4_K, ggmlTypeQ5_K, ggmlTypeQ6_K, ggmlTypeIQ4XS, ggmlTypeIQ2S, ggmlTypeIQ3S:
		return qkK, true
	default:
		return 0, false
	}
}

// dequantRange dequantizes the block-aligned element range [start, start+len(dst))
// of a tensor's raw bytes into dst, dispatching to the per-block kernels. bs is the
// type's block size (from ggmlBlockElems). The caller validates alignment.
func dequantRange(typ uint32, raw []byte, start int, dst []float32, bs int) {
	switch typ {
	case ggmlTypeF32:
		for i := range dst {
			dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*(start+i):]))
		}
	case ggmlTypeF16:
		for i := range dst {
			dst[i] = halfBitsToF32(binary.LittleEndian.Uint16(raw[2*(start+i):]))
		}
	default:
		first := start / bs
		for k := 0; k*bs < len(dst); k++ {
			out := dst[k*bs : (k+1)*bs]
			switch typ {
			case ggmlTypeQ8_0:
				dequantQ8_0Block(raw, first+k, out)
			case ggmlTypeQ4_0:
				dequantQ4_0Block(raw, first+k, out)
			case ggmlTypeQ5_0:
				dequantQ5_0Block(raw, first+k, out)
			case ggmlTypeQ2_K:
				dequantQ2KBlock(raw, first+k, out)
			case ggmlTypeQ3_K:
				dequantQ3KBlock(raw, first+k, out)
			case ggmlTypeQ4_K:
				dequantQ4KBlock(raw, first+k, out)
			case ggmlTypeQ5_K:
				dequantQ5KBlock(raw, first+k, out)
			case ggmlTypeQ6_K:
				dequantQ6KBlock(raw, first+k, out)
			case ggmlTypeIQ4NL:
				dequantIQ4NLBlock(raw, first+k, out)
			case ggmlTypeIQ4XS:
				dequantIQ4XSBlock(raw, first+k, out)
			case ggmlTypeIQ2S:
				dequantIQ2SBlock(raw, first+k, out)
			case ggmlTypeIQ3S:
				dequantIQ3SBlock(raw, first+k, out)
			}
		}
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
	case ggmlTypeQ8_0, ggmlTypeQ4_0, ggmlTypeQ5_0, ggmlTypeIQ4NL:
		if n%32 != 0 {
			return nil, fmt.Errorf("element count %d not a multiple of 32 (block size)", n)
		}
		blocks := n / 32
		switch info.typ {
		case ggmlTypeQ8_0:
			nbytes = blocks * 34 // 2-byte f16 scale + 32 int8
		case ggmlTypeQ4_0, ggmlTypeIQ4NL:
			nbytes = blocks * 18 // 2-byte f16 scale + 16 packed nibbles
		default: // Q5_0
			nbytes = blocks * 22 // 2-byte f16 scale + 4-byte high bits + 16 packed nibbles
		}
	case ggmlTypeQ2_K, ggmlTypeQ3_K, ggmlTypeQ4_K, ggmlTypeQ5_K, ggmlTypeQ6_K, ggmlTypeIQ4XS, ggmlTypeIQ2S, ggmlTypeIQ3S:
		if n%qkK != 0 {
			return nil, fmt.Errorf("element count %d not a multiple of %d (super-block)", n, qkK)
		}
		sb := n / qkK
		switch info.typ {
		case ggmlTypeIQ2S:
			nbytes = sb * 82 // d(f16) + qs[32] + signs[32] + qh[8] + scales[8]
		case ggmlTypeIQ3S:
			nbytes = sb * 110 // d(f16) + qs[64] + qh[8] + signs[32] + scales[4]
		case ggmlTypeIQ4XS:
			nbytes = sb * 136 // d(f16) + scales_h(u16) + scales_l[4] + qs[128]
		case ggmlTypeQ2_K:
			nbytes = sb * 84 // scales[16] + qs[64] + d + dmin (f16 each)
		case ggmlTypeQ3_K:
			nbytes = sb * 110 // hmask[32] + qs[64] + scales[12] + d(f16)
		case ggmlTypeQ4_K:
			nbytes = sb * 144 // d + dmin (f16 each) + scales[12] + qs[128]
		case ggmlTypeQ5_K:
			nbytes = sb * 176 // d + dmin (f16 each) + scales[12] + qh[32] + qs[128]
		default: // Q6_K
			nbytes = sb * 210 // ql[128] + qh[64] + scales[16] + d(f16)
		}
	default:
		return nil, fmt.Errorf("unsupported ggml type %d", info.typ)
	}
	// info.offset is an untrusted uint64; compare without adding so a near-2^64
	// offset can't wrap the sum past this guard and panic the slice below.
	if info.offset > uint64(len(g.data)) || uint64(nbytes) > uint64(len(g.data))-info.offset {
		return nil, fmt.Errorf("data range [%d:+%d] past section end %d", info.offset, nbytes, len(g.data))
	}
	return g.data[info.offset : info.offset+uint64(nbytes)], nil
}

// dequantQ8_0Block dequantizes one 32-element Q8_0 block (b) into out[:32]: a
// f16 scale d then 32 int8 q; value = d*q.
func dequantQ8_0Block(raw []byte, b int, out []float32) {
	base := b * 34
	d := halfBitsToF32(binary.LittleEndian.Uint16(raw[base:]))
	qs := raw[base+2 : base+34]
	for i := range 32 {
		out[i] = float32(int8(qs[i])) * d
	}
}

// dequantQ4_0Block dequantizes one 32-element Q4_0 block (b) into out[:32]: a f16
// scale d then 16 packed bytes; low nibble of byte i is element i, high nibble is
// element i+16, each recentered by -8: value = d*(nibble-8).
func dequantQ4_0Block(raw []byte, b int, out []float32) {
	base := b * 18
	d := halfBitsToF32(binary.LittleEndian.Uint16(raw[base:]))
	qs := raw[base+2 : base+18]
	for i := range 16 {
		v := qs[i]
		out[i] = float32(int(v&0x0F)-8) * d
		out[i+16] = float32(int(v>>4)-8) * d
	}
}

// dequantQ5_0Block dequantizes one 32-element Q5_0 block (b) into out[:32]: a f16
// scale d, a 4-byte qh carrying each element's 5th (high) bit, then 16 packed low
// nibbles. For element j the code is (low nibble | high bit << 4) ∈ [0,31],
// recentered by -16: value = d*(code-16). Mirrors ggml's dequantize_row_q5_0.
func dequantQ5_0Block(raw []byte, b int, out []float32) {
	base := b * 22
	d := halfBitsToF32(binary.LittleEndian.Uint16(raw[base:]))
	qh := binary.LittleEndian.Uint32(raw[base+2:])
	qs := raw[base+6 : base+22]
	for j := range 16 {
		xh0 := byte(((qh >> uint(j)) << 4) & 0x10) // bit j → bit 4
		xh1 := byte((qh >> uint(j+12)) & 0x10)     // bit j+16 → bit 4
		q0 := int32((qs[j]&0x0F)|xh0) - 16
		q1 := int32((qs[j]>>4)|xh1) - 16
		out[j] = float32(q0) * d
		out[j+16] = float32(q1) * d
	}
}

// dequantQ6KBlock dequantizes one 256-element Q6_K super-block (sb) into out[:256].
// Layout (210 bytes): ql[128] (low 4 bits), qh[64] (high 2 bits), scales[16]
// (int8), d (f16 super-scale). Mirrors ggml's dequantize_row_q6_K: a 6-bit quant
// q∈[0,63] recentered by -32, scaled by its sub-block int8 scale and d.
func dequantQ6KBlock(raw []byte, sb int, out []float32) {
	base := sb * 210
	ql := raw[base : base+128]
	qh := raw[base+128 : base+192]
	sc := raw[base+192 : base+208] // int8 scales
	d := halfBitsToF32(binary.LittleEndian.Uint16(raw[base+208:]))
	for chunk := range 2 {
		n0 := chunk * 128
		qlo := ql[chunk*64:]
		qho := qh[chunk*32:]
		sco := sc[chunk*8:]
		for l := range 32 {
			is := l / 16
			q1 := int8((qlo[l]&0x0F)|(((qho[l]>>0)&3)<<4)) - 32
			q2 := int8((qlo[l+32]&0x0F)|(((qho[l]>>2)&3)<<4)) - 32
			q3 := int8((qlo[l]>>4)|(((qho[l]>>4)&3)<<4)) - 32
			q4 := int8((qlo[l+32]>>4)|(((qho[l]>>6)&3)<<4)) - 32
			out[n0+l+0] = d * float32(int8(sco[is+0])) * float32(q1)
			out[n0+l+32] = d * float32(int8(sco[is+2])) * float32(q2)
			out[n0+l+64] = d * float32(int8(sco[is+4])) * float32(q3)
			out[n0+l+96] = d * float32(int8(sco[is+6])) * float32(q4)
		}
	}
}

// dequantIQ4NLBlock dequantizes one 32-element IQ4_NL block (b) into out[:32].
// Layout (18 bytes, same size as Q4_0): a f16 scale d then 16 packed bytes; the
// low nibble of byte j indexes element j and the high nibble element j+16 — each
// nibble looked up in the kvaluesIQ4NL codebook: value = d·kvaluesIQ4NL[code].
func dequantIQ4NLBlock(raw []byte, b int, out []float32) {
	base := b * 18
	d := halfBitsToF32(binary.LittleEndian.Uint16(raw[base:]))
	qs := raw[base+2 : base+18]
	for j := range 16 {
		out[j] = d * float32(kvaluesIQ4NL[qs[j]&0x0F])
		out[j+16] = d * float32(kvaluesIQ4NL[qs[j]>>4])
	}
}

// dequantIQ4XSBlock dequantizes one 256-element IQ4_XS super-block (sb) into
// out[:256]. Layout (136 bytes): d (f16 super-scale), scales_h (u16), scales_l[4],
// qs[128]. The super-block splits into eight 32-element sub-blocks; sub-block ib
// has a 6-bit scale ls assembled from scales_l (low 4 bits) and scales_h (high 2
// bits), recentered by −32: dl = d·(ls−32). Each nibble of qs indexes the shared
// kvaluesIQ4NL codebook (low → element j, high → j+16 within the sub-block).
func dequantIQ4XSBlock(raw []byte, sb int, out []float32) {
	base := sb * 136
	d := halfBitsToF32(binary.LittleEndian.Uint16(raw[base:]))
	scalesH := binary.LittleEndian.Uint16(raw[base+2:])
	scalesL := raw[base+4 : base+8]
	qs := raw[base+8 : base+136]
	for ib := range 8 { // eight 32-element sub-blocks
		ls := int((scalesL[ib/2]>>(4*(ib%2)))&0x0F) | int((scalesH>>(2*ib))&3)<<4
		dl := d * float32(ls-32)
		q := qs[ib*16 : ib*16+16]
		o := out[ib*32 : ib*32+32]
		for j := range 16 {
			o[j] = dl * float32(kvaluesIQ4NL[q[j]&0x0F])
			o[j+16] = dl * float32(kvaluesIQ4NL[q[j]>>4])
		}
	}
}

// dequantIQ2SBlock dequantizes one 256-element IQ2_S super-block (sb) into
// out[:256]. Layout (82 bytes): d (f16), qs[32] (low 8 bits of each 8-wide grid
// index), signs[32] (per-element sign bits), qh[8] (high 2 bits of each index),
// scales[8] (4-bit sub-scales). The super-block is 16 sub-blocks of 16, each a
// 4-bit scale and two 8-wide codebook lookups (iq2sGrid, 1024×8); the per-element
// sign comes from the packed sign bits. Mirrors ggml's dequantize_row_iq2_s.
func dequantIQ2SBlock(raw []byte, sb int, out []float32) {
	base := sb * 82
	d := halfBitsToF32(binary.LittleEndian.Uint16(raw[base:]))
	qs := raw[base+2 : base+34]
	signs := raw[base+34 : base+66]
	qh := raw[base+66 : base+74]
	scales := raw[base+74 : base+82]
	for sub := range 16 {
		sc := int((scales[sub/2] >> (4 * (sub & 1))) & 0x0F)
		db := d * (0.5 + float32(sc)) * 0.25
		for pair := range 2 {
			k := sub*2 + pair
			idx := int(qs[k]) | int((qh[k/4]>>(2*(k&3)))&3)<<8
			g := iq2sGrid[idx*8 : idx*8+8]
			sg := signs[k]
			o := out[sub*16+pair*8:]
			for j := range 8 {
				v := db * float32(g[j])
				if (sg>>j)&1 != 0 {
					v = -v
				}
				o[j] = v
			}
		}
	}
}

// dequantIQ3SBlock dequantizes one 256-element IQ3_S super-block (sb) into
// out[:256]. Layout (110 bytes): d (f16), qs[64] (low 8 bits of each 4-wide grid
// index), qh[8] (1 high bit per index), signs[32] (per-element sign bits),
// scales[4] (4-bit sub-scales). 8 sub-blocks of 32, each a 4-bit scale and eight
// 4-wide codebook lookups (iq3sGrid, 512×4); the sign comes from the packed sign
// bits. Mirrors ggml's dequantize_row_iq3_s.
func dequantIQ3SBlock(raw []byte, sb int, out []float32) {
	base := sb * 110
	d := halfBitsToF32(binary.LittleEndian.Uint16(raw[base:]))
	qs := raw[base+2 : base+66]
	qh := raw[base+66 : base+74]
	signs := raw[base+74 : base+106]
	scales := raw[base+106 : base+110]
	for p := range 256 {
		sub := p / 32
		sc := int((scales[sub/2] >> (4 * (sub & 1))) & 0x0F)
		db := d * float32(1+2*sc)
		m := p / 4 // grid-index number (0..63)
		idx := int(qs[m]) | int((qh[m/8]>>(m&7))&1)<<8
		v := db * float32(iq3sGrid[idx*4+p%4])
		if (signs[p/8]>>(p&7))&1 != 0 {
			v = -v
		}
		out[p] = v
	}
}

// dequantQ4KBlock dequantizes one 256-element Q4_K super-block (sb) into out[:256].
// Layout (144 bytes): d (f16), dmin (f16), scales[12] (6-bit scales+mins, packed),
// qs[128] (4-bit quants). Mirrors ggml's dequantize_row_q4_K: y = d·scale·q −
// dmin·min, with scale/min unpacked by get_scale_min_k4.
func dequantQ4KBlock(raw []byte, sb int, out []float32) {
	base := sb * 144
	d := halfBitsToF32(binary.LittleEndian.Uint16(raw[base:]))
	dmin := halfBitsToF32(binary.LittleEndian.Uint16(raw[base+2:]))
	scales := raw[base+4 : base+16]
	qs := raw[base+16 : base+144]
	yi := 0
	for j := range 4 { // four 64-element groups
		is := 2 * j
		sc1, m1 := q4kScaleMin(is+0, scales)
		sc2, m2 := q4kScaleMin(is+1, scales)
		d1, off1 := d*float32(sc1), dmin*float32(m1)
		d2, off2 := d*float32(sc2), dmin*float32(m2)
		q := qs[j*32 : j*32+32]
		for l := range 32 {
			out[yi] = d1*float32(q[l]&0x0F) - off1
			yi++
		}
		for l := range 32 {
			out[yi] = d2*float32(q[l]>>4) - off2
			yi++
		}
	}
}

// q4kScaleMin unpacks the j-th 6-bit scale and min from a Q4_K/Q5_K super-block's
// 12-byte scales array (ggml's get_scale_min_k4).
func q4kScaleMin(j int, q []byte) (scale, min uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	scale = (q[j+4] & 0x0F) | ((q[j-4] >> 6) << 4)
	min = (q[j+4] >> 4) | ((q[j] >> 6) << 4)
	return scale, min
}

// dequantQ5KBlock dequantizes one 256-element Q5_K super-block (sb) into out[:256].
// Layout (176 bytes): d (f16), dmin (f16), scales[12] (6-bit scales+mins, same
// packing as Q4_K), qh[32] (the 5th/high bit per element), qs[128] (low 4 bits).
// Mirrors ggml's dequantize_row_q5_K: y = d·sc·q − dmin·m with q a 5-bit code
// (low nibble | high bit << 4). The high bit for each 32-wide half is selected by
// a mask that walks the qh byte two bits at a time across the four 64-elem groups.
func dequantQ5KBlock(raw []byte, sb int, out []float32) {
	base := sb * 176
	d := halfBitsToF32(binary.LittleEndian.Uint16(raw[base:]))
	dmin := halfBitsToF32(binary.LittleEndian.Uint16(raw[base+2:]))
	scales := raw[base+4 : base+16]
	qh := raw[base+16 : base+48]
	qs := raw[base+48 : base+176]
	yi := 0
	u1, u2 := byte(1), byte(2)
	for j := range 4 { // four 64-element groups
		is := 2 * j
		sc1, m1 := q4kScaleMin(is+0, scales)
		sc2, m2 := q4kScaleMin(is+1, scales)
		d1, off1 := d*float32(sc1), dmin*float32(m1)
		d2, off2 := d*float32(sc2), dmin*float32(m2)
		ql := qs[j*32 : j*32+32]
		for l := range 32 {
			var h float32
			if qh[l]&u1 != 0 {
				h = 16
			}
			out[yi] = d1*(float32(ql[l]&0x0F)+h) - off1
			yi++
		}
		for l := range 32 {
			var h float32
			if qh[l]&u2 != 0 {
				h = 16
			}
			out[yi] = d2*(float32(ql[l]>>4)+h) - off2
			yi++
		}
		u1 <<= 2
		u2 <<= 2
	}
}

// dequantQ2KBlock dequantizes one 256-element Q2_K super-block (sb) into out[:256].
// Layout (84 bytes): scales[16] (each byte a 4-bit scale in the low nibble and a
// 4-bit min in the high nibble), qs[64] (2-bit quants), d (f16 super-scale), dmin
// (f16 super-min). Mirrors ggml's dequantize_row_q2_K: y = d·scale·q2 − dmin·min,
// q2 the 2-bit code. No high-bit mask (unlike Q3_K) — the coarsest K-quant.
func dequantQ2KBlock(raw []byte, sb int, out []float32) {
	base := sb * 84
	scales := raw[base : base+16]
	qs := raw[base+16 : base+80]
	d := halfBitsToF32(binary.LittleEndian.Uint16(raw[base+80:]))
	dmin := halfBitsToF32(binary.LittleEndian.Uint16(raw[base+82:]))

	yi, is := 0, 0
	for n := range 2 { // two 128-element halves
		qb := n * 32 // qs advances by 32 each half
		shift := uint(0)
		for range 4 {
			sc := scales[is]
			is++
			dl, ml := d*float32(sc&0x0F), dmin*float32(sc>>4)
			for l := range 16 {
				out[yi] = dl*float32((qs[qb+l]>>shift)&3) - ml
				yi++
			}
			sc = scales[is]
			is++
			dl, ml = d*float32(sc&0x0F), dmin*float32(sc>>4)
			for l := range 16 {
				out[yi] = dl*float32((qs[qb+l+16]>>shift)&3) - ml
				yi++
			}
			shift += 2
		}
	}
}

// dequantQ3KBlock dequantizes one 256-element Q3_K super-block (sb) into out[:256].
// Layout (110 bytes): hmask[32] (the 3rd/high bit per element), qs[64] (low 2
// bits), scales[12] (16 six-bit sub-block scales, bit-packed), d (f16). Mirrors
// ggml's dequantize_row_q3_K: the 12 scale bytes are unpacked (the aux dance) into
// 16 int8 scales recentered by −32, and each element is a 2-bit code lifted to
// [−4,3] by the hmask bit: y = d·scale·(q2 − (hmask_bit ? 0 : 4)).
func dequantQ3KBlock(raw []byte, sb int, out []float32) {
	const (
		kmask1 = 0x03030303
		kmask2 = 0x0f0f0f0f
	)
	base := sb * 110
	hm := raw[base : base+32]
	q := raw[base+32 : base+96]
	scRaw := raw[base+96 : base+108]
	dAll := halfBitsToF32(binary.LittleEndian.Uint16(raw[base+108:]))

	// Unpack the 16 six-bit sub-block scales: the 12 packed bytes are read as three
	// little-endian uint32s, recombined (ggml's bit dance), and laid back down as a
	// 16-byte int8 buffer. Each scale is recentered by −32 at use.
	a0 := binary.LittleEndian.Uint32(scRaw[0:])
	a1 := binary.LittleEndian.Uint32(scRaw[4:])
	tmp := binary.LittleEndian.Uint32(scRaw[8:])
	na := [4]uint32{
		(a0 & kmask2) | (((tmp >> 0) & kmask1) << 4),
		(a1 & kmask2) | (((tmp >> 2) & kmask1) << 4),
		((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4),
		((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4),
	}
	var sc [16]int8
	for i, v := range na {
		sc[4*i+0] = int8(v)
		sc[4*i+1] = int8(v >> 8)
		sc[4*i+2] = int8(v >> 16)
		sc[4*i+3] = int8(v >> 24)
	}

	yi, is := 0, 0
	m := byte(1)
	for n := range 2 { // two 128-element halves
		qb := n * 32 // q advances by 32 each half
		shift := uint(0)
		for range 4 {
			dl := dAll * float32(int(sc[is])-32)
			is++
			for l := range 16 {
				var sub float32 = 4
				if hm[l]&m != 0 {
					sub = 0
				}
				out[yi] = dl * (float32((q[qb+l]>>shift)&3) - sub)
				yi++
			}
			dl = dAll * float32(int(sc[is])-32)
			is++
			for l := range 16 {
				var sub float32 = 4
				if hm[l+16]&m != 0 {
					sub = 0
				}
				out[yi] = dl * (float32((q[qb+l+16]>>shift)&3) - sub)
				yi++
			}
			shift += 2
			m <<= 1
		}
	}
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
