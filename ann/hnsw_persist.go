package ann

import (
	"encoding/binary"
	"io"
	"math"
	"math/rand/v2"
	"unsafe"

	"github.com/townsendmerino/aikit/internal/cursor"
)

// HNSW serialization. The format is versioned from day one so the on-disk /
// //go:embed-ed layout can evolve without silently mis-reading old blobs:
//
//	magic uint32 | version uint32 | flags uint32 (reserved, always 0 today)
//	dim, ndocs, m, m0, efConstruction, efSearch, entry, maxLayer  (int32 each)
//	mL float64 | seed uint64 | heuristic uint8 (0/1) | int8 uint8 (0/1)
//	pad: 2 zero bytes (so the vector block below starts 8-byte aligned)
//	vectors:  int8 mode → ndocs×dim int8 codes, zero pad to a multiple of 4 (v5), ndocs float32 scales;
//	          else      → ndocs × dim float32 (little-endian, row-major)
//	graph (v5, CSR): listStart (ndocs+1) × uint32 | idOff (nLists+1) × uint32 | ids nIDs × int32
//	          node d's layer lists are lists listStart[d] … listStart[d+1]-1, layer 0 first;
//	          list i is ids[idOff[i]:idOff[i+1]]; nLists = listStart[ndocs], nIDs = idOff[nLists]
//	graph (v4): per node — layer int32, then for l in 0..layer:
//	          nbrCount int32, then nbrCount × neighbor id int32
//
// All integers little-endian. entry is int32 (it is -1 for an empty index); every
// other count is non-negative. Load validates every length against the bytes that
// remain, so a corrupt or hostile blob returns an error rather than panicking or
// over-allocating.
// Format-stability policy: Load reads v5 (written since v1.71.0) and v4 (v1.x before
// it), because indexes on disk outlive the release that wrote them (ken keeps them);
// it rejects any other version with ErrFormat (loud, never a silent misread). See
// README "Serialized blob formats".
//
// v5 (memory campaign item 7) changed only the graph: v4's per-node runs of
// layer/count/ids became three flat, 4-byte-aligned arrays, which LoadHNSWMmap aliases
// from the mapping instead of copying (and Load copies as three allocations), and the
// int8 scales are padded to 4-byte alignment so they alias too. A v4 blob loads into the
// same in-memory form; saving it again writes v5.
//
// v4's format bump (was v3) bundled the two items the FORMAT-BUMP CHECKLIST had
// named as one deliberate future bump, so this note documents what landed rather
// than what's still pending:
//  1. a reserved uint32 flags word right after the version, so a later additive
//     change can extend via flags WITHOUT a version bump (the anti-churn
//     mechanism) — read and ignored today, since no flag bit is defined yet;
//  2. the header pads to end 8-byte aligned, so the float32 vector block (f32
//     storage mode) starts 8-byte — and so also 4-byte, the only real requirement
//     — aligned within any page-aligned mmap. This is what makes LoadHNSWMmap
//     (hnsw_mmap.go) able to alias vecs/bq directly from the mapping instead of
//     copying them, mirroring LoadFlatI8Mmap's int8 aliasing. int8-mode codes have
//     no alignment requirement of their own and were already safe to alias before
//     this bump; only f32 mode needed it.
const (
	hnswMagic   uint32 = 0x484E5357 // "HNSW"
	hnswVersion uint32 = 5          // v5: the graph as CSR (see above); v4 added the reserved flags word + alignment pad
	hnswV4      uint32 = 4          // still read
	// rngSplit matches NewHNSW's PCG seeding so a loaded index re-creates an
	// equivalently-seeded rng (for Add-after-load).
	rngSplit uint64 = 0x9e3779b97f4a7c15
)

// hnswPad is the v4 header's 2-byte alignment pad — always zero.
var hnswPad [2]byte

// blobSize returns the exact serialized length, which is both MarshalBinary's
// allocation size and WriteTo's reported total. Header + vectors + graph, where
// each neighbor id is 4 bytes and each node costs 4 for its layer plus 4 per
// layer for that layer's count.
func (h *HNSW) blobSize() int {
	n := h.count()
	nNbr, nLists := 0, 0
	for d := range n {
		layer := h.nodeLayer(d)
		nLists += layer + 1
		for l := 0; l <= layer; l++ {
			nNbr += len(h.nbrs(d, l))
		}
	}
	vecBytes := n * h.dim * 4
	if h.int8 {
		vecBytes = len(h.bq) + pad4(len(h.bq)) + len(h.scales)*4
	}
	return hnswHeaderBytes + vecBytes + (n+1)*4 + (nLists+1)*4 + nNbr*4
}

// pad4 is the zero padding that takes n bytes to a multiple of 4.
func pad4(n int) int { return (4 - n%4) % 4 }

// hnswHeaderBytes is the fixed prefix: magic+version+flags+8 int32 config
// scalars (11×4), mL and seed (2×8), heuristic and int8 mode (2×1), then a
// 2-byte pad — 64 total, a multiple of 8 (see the v4 format note above).
const hnswHeaderBytes = 11*4 + 2*8 + 2 + 2

// MarshalBinary serializes the built index — graph, vectors, and config — into a
// versioned byte blob that Load turns back into a query-ready *HNSW. It implements
// encoding.BinaryMarshaler, so the index also round-trips through gob and friends.
//
// The point is the //go:embed pattern: build the graph once offline, embed the
// bytes in the binary, and Load them at startup instead of rebuilding per process.
//
// This is a thin wrapper over WriteTo into an exactly-sized buffer, so the two
// cannot encode different bytes — there is only one encoder. Prefer WriteTo when
// the destination is a file or a network connection: this necessarily materializes
// a second full copy of the index (see WriteTo's doc for the numbers).
func (h *HNSW) MarshalBinary() ([]byte, error) {
	// Whole-blob mode: the destination IS the buffer, so the encoder writes each
	// byte exactly once — no staging copy, and one allocation at the exact size.
	// (The previous version's own capacity estimate was short — it budgeted 8
	// bytes of graph header per node when a node with L layers needs 4+4L — so
	// append doubled it: 131.0 MB allocated for a 58.3 MB blob.)
	bw := &hblobWriter{buf: make([]byte, h.blobSize())}
	h.encode(bw)
	if bw.err != nil {
		return nil, bw.err
	}
	return bw.buf[:bw.n], nil
}

// WriteTo streams the same bytes MarshalBinary returns, without building them.
//
// It implements io.WriterTo, so os.File, bufio.Writer, gzip.Writer and io.Copy
// all pick it up automatically.
//
// WHY IT EXISTS. MarshalBinary allocates the whole blob, so a caller writing an
// index to disk holds the index AND a second full copy of it at once. An HNSW
// index is the largest thing this package produces — at n=1M, dim=768 the f32
// vectors alone are 3.1 GB, and MarshalBinary doubles the peak to save it. The
// blob is not needed as a value by anyone who is only going to write it; it was
// simply the only shape offered (lens doc §4.3).
//
// Streaming changes the transient cost from a full copy to one 64 KiB buffer,
// whatever the index's size. In int8 mode the code block is written straight from
// the index's own memory in a single Write — the same int8/byte aliasing
// MarshalBinary and LoadFlatI8Mmap both already rely on — so only the scales and
// the graph pass through the buffer.
func (h *HNSW) WriteTo(w io.Writer) (int64, error) {
	bw := &hblobWriter{w: w, buf: make([]byte, hblobBufSize)}
	h.encode(bw)
	return bw.finish()
}

// hblobBufSize is the streaming staging buffer. Big enough that the syscall count
// is negligible (a 3 GB index is ~48k writes, and callers wrapping in a
// bufio.Writer pay nothing extra), small enough to be a fixed cost independent of
// index size — which is the entire point of WriteTo.
const hblobBufSize = 64 << 10

// encode is the single encoder both serialization surfaces run. There is
// deliberately no second copy of the format: MarshalBinary passes a writer whose
// buffer is the whole output, WriteTo passes one that flushes every 64 KiB, and
// the bytes are identical because the code producing them is identical.
func (h *HNSW) encode(bw *hblobWriter) {
	bw.u32(hnswMagic)
	bw.u32(hnswVersion)
	bw.u32(0) // reserved flags — always 0 today; a future additive change sets bits here instead of bumping the version
	bw.puti(h.dim)
	bw.puti(h.count())
	bw.puti(h.m)
	bw.puti(h.m0)
	bw.puti(h.efConstruction)
	bw.puti(h.efSearch)
	bw.puti(h.entry)
	bw.puti(h.maxLayer)
	bw.u64(math.Float64bits(h.mL))
	bw.u64(h.seed)
	bw.boolByte(h.heuristic)
	bw.boolByte(h.int8) // v3: storage mode
	bw.raw(hnswPad[:])  // v4: 2 zero bytes so the vector block below starts 8-byte aligned

	// Vectors: int8 codes + per-vector scales (int8 mode) or row-major f32.
	if h.int8 {
		if len(h.bq) > 0 {
			// int8 → byte is the two's-complement round-trip and the two have
			// identical layout, so the whole code block is one memmove out. Same
			// aliasing argument as flat_i8_persist.go:64.
			bw.raw(unsafe.Slice((*byte)(unsafe.Pointer(&h.bq[0])), len(h.bq)))
		}
		bw.raw(hnswPad4[:pad4(len(h.bq))]) // v5: so the scales start 4-byte aligned
		bw.f32s(h.scales)
	} else {
		// f32 is encoded, not aliased: unlike int8 a float32 view would bake in
		// the host's endianness, and the format is little-endian by contract.
		for _, v := range h.vecs {
			bw.f32s(v)
		}
	}
	// v5 graph: listStart, idOff, ids — written from either representation, without building the arrays.
	n := h.count()
	var acc uint64
	bw.u32(0)
	for d := range n {
		acc += uint64(h.nodeLayer(d) + 1)
		bw.u32(bw.within(acc))
	}
	acc = 0
	bw.u32(0)
	for d := range n {
		for l := 0; l <= h.nodeLayer(d); l++ {
			acc += uint64(len(h.nbrs(d, l)))
			bw.u32(bw.within(acc))
		}
	}
	for d := range n {
		for l := 0; l <= h.nodeLayer(d); l++ {
			bw.i32s(h.nbrs(d, l))
		}
	}
}

// within returns v as a uint32, or records an error (and returns 0) when the graph is too large for v5's 32-bit
// offsets — over 4,294,967,295 lists or neighbor ids, far beyond any index this package can hold in memory.
func (b *hblobWriter) within(v uint64) uint32 {
	if v > math.MaxUint32 && b.err == nil {
		b.err = errFormatf("ann: HNSW graph exceeds the v5 format's 2^32-1 offsets")
	}
	return uint32(v)
}

// hnswPad4 is the v5 int8 code block's alignment pad — always zero.
var hnswPad4 [3]byte

// hblobWriter is the encoder target for both serialization surfaces.
//
//	w != nil — STREAMING: buf is a fixed staging buffer, flushed to w when full.
//	w == nil — WHOLE-BLOB: buf is the entire output and is never flushed, so each
//	           byte is written exactly once and MarshalBinary needs no staging copy.
//
// Errors are sticky: the first failed Write is remembered and every later call is
// a no-op, so encode() reads as straight-line code instead of an error check per
// field.
type hblobWriter struct {
	w     io.Writer
	buf   []byte
	n     int
	total int64
	err   error
}

func (b *hblobWriter) flush() {
	if b.err != nil || b.n == 0 || b.w == nil {
		return
	}
	n, err := b.w.Write(b.buf[:b.n])
	b.total += int64(n)
	b.n = 0
	if err != nil {
		b.err = err
	}
}

// ensure makes room for at least need bytes and reports whether the caller may
// write them.
func (b *hblobWriter) ensure(need int) bool {
	if b.err != nil {
		return false
	}
	if b.n+need <= len(b.buf) {
		return true
	}
	if b.w == nil {
		// Whole-blob mode ran out, i.e. blobSize() under-counted. Grow rather than
		// fail: a miscount must not corrupt a caller's blob, and
		// TestHNSW_WriteToMatchesMarshal is what catches it as a defect.
		b.buf = append(b.buf, make([]byte, b.n+need-len(b.buf))...)
		return true
	}
	b.flush()
	return b.err == nil
}

func (b *hblobWriter) u32(v uint32) {
	if !b.ensure(4) {
		return
	}
	binary.LittleEndian.PutUint32(b.buf[b.n:], v)
	b.n += 4
}

func (b *hblobWriter) u64(v uint64) {
	if !b.ensure(8) {
		return
	}
	binary.LittleEndian.PutUint64(b.buf[b.n:], v)
	b.n += 8
}

func (b *hblobWriter) puti(v int) { b.u32(uint32(int32(v))) }

func (b *hblobWriter) boolByte(v bool) {
	if !b.ensure(1) {
		return
	}
	if v {
		b.buf[b.n] = 1
	} else {
		b.buf[b.n] = 0
	}
	b.n++
}

// f32s and i32s encode a whole slice at a time. The bulk form matters: the
// per-value entry point costs a space check and a sticky-error check per element,
// and at 12.8 M floats that overhead was 2× the encode itself. Here both checks
// are paid once per buffer-full, and the inner loop is a bounds-check-hoistable
// run of stores.
func (b *hblobWriter) f32s(vs []float32) {
	for len(vs) > 0 {
		if !b.ensure(4) {
			return
		}
		k := min(len(vs), (len(b.buf)-b.n)/4)
		dst := b.buf[b.n : b.n+k*4]
		for j, f := range vs[:k] {
			binary.LittleEndian.PutUint32(dst[j*4:], math.Float32bits(f))
		}
		b.n += k * 4
		vs = vs[k:]
	}
}

func (b *hblobWriter) i32s(vs []int32) {
	for len(vs) > 0 {
		if !b.ensure(4) {
			return
		}
		k := min(len(vs), (len(b.buf)-b.n)/4)
		dst := b.buf[b.n : b.n+k*4]
		for j, v := range vs[:k] {
			binary.LittleEndian.PutUint32(dst[j*4:], uint32(v))
		}
		b.n += k * 4
		vs = vs[k:]
	}
}

// raw emits p without staging it — for a block already contiguous in the index's
// own memory, where a copy through the buffer would be pointless. In whole-blob
// mode there is nowhere to stream to, so it copies into the output.
func (b *hblobWriter) raw(p []byte) {
	if b.w == nil {
		if !b.ensure(len(p)) {
			return
		}
		b.n += copy(b.buf[b.n:], p)
		return
	}
	b.flush()
	if b.err != nil {
		return
	}
	n, err := b.w.Write(p)
	b.total += int64(n)
	if err != nil {
		b.err = err
	}
}

func (b *hblobWriter) finish() (int64, error) {
	b.flush()
	return b.total, b.err
}

// hcur is a bounds-checked little-endian reader over the serialized blob —
// the shared primitive (internal/cursor.Cursor) plus HNSW-specific
// allocation-guarding readers. Counts are clamped to what the remaining
// bytes could hold before they drive an allocation.
type hcur struct {
	cursor.Cursor
}

func newHcur(data []byte) *hcur {
	return &hcur{cursor.Cursor{B: data, Context: "ann: HNSW blob", Errorf: errFormatf}}
}

// int8s reads n raw int8 code bytes (the int8 vector block).
func (c *hcur) int8s(n int) []int8 {
	raw := c.Bytes(n)
	if raw == nil {
		return nil
	}
	out := make([]int8, n)
	for i, b := range raw {
		out[i] = int8(b)
	}
	return out
}

// asInt reads a signed int32 (used for entry, which is -1 for an empty index).
func (c *hcur) asInt() int { return int(int32(c.U32())) }

// readLen reads an allocation-driving LENGTH (ndocs, dim, neighbor count) and
// rejects it if it can't fit the bytes that remain (every subsequent element is
// ≥4 bytes), so a hostile count can't drive a giant make() before the reads hit
// EOF.
func (c *hcur) readLen() int {
	v := int32(c.U32())
	if c.Err != nil {
		return 0
	}
	if v < 0 || int(v) > c.Remaining()/4 {
		c.Err = errFormatf("ann: HNSW blob count %d exceeds remaining bytes", v)
		return 0
	}
	return int(v)
}

// cfgMax bounds the config scalars (m, m0, efConstruction, efSearch). Unlike
// readLen() these aren't byte-sized lengths — they're tuning knobs — but efSearch
// sizes Query's candidate map, so an absurd value must not slip through and OOM.
const cfgMax = 1 << 20

// cfg reads a non-negative config scalar, rejecting negatives and absurd values.
func (c *hcur) cfg(name string) int {
	v := int32(c.U32())
	if c.Err != nil {
		return 0
	}
	if v < 0 || int(v) > cfgMax {
		c.Err = errFormatf("ann: HNSW %s %d out of [0,%d]", name, v, cfgMax)
		return 0
	}
	return int(v)
}

// scanGraph walks the graph section from pos without allocating and returns the
// exact number of neighbor ids and per-layer lists it contains, so a v4 load can
// size its CSR arrays in one shot.
//
// It cannot use a byte-count upper bound instead: ids are 4 bytes each in the blob
// so remaining/4 bounds them tightly, but each [][]int32 header is 24 bytes in
// memory, and sizing THAT from remaining/4 would turn a 6 GB blob into a 38 GB
// allocation — an amplification a hostile input could aim. An exact pre-pass has
// no such gap. It re-validates nothing: ok=false simply means "don't bother", and
// the real read reports the error.
func scanGraph(b []byte, pos, ndocs int) (nIDs, nLayers int, ok bool) {
	for range ndocs {
		if pos+4 > len(b) {
			return 0, 0, false
		}
		layer := int(int32(binary.LittleEndian.Uint32(b[pos:])))
		pos += 4
		if layer < 0 || layer > (len(b)-pos)/4 {
			return 0, 0, false
		}
		nLayers += layer + 1
		for range layer + 1 {
			if pos+4 > len(b) {
				return 0, 0, false
			}
			cnt := int(int32(binary.LittleEndian.Uint32(b[pos:])))
			pos += 4
			if cnt < 0 || cnt > (len(b)-pos)/4 {
				return 0, 0, false
			}
			nIDs += cnt
			pos += cnt * 4
		}
	}
	return nIDs, nLayers, true
}

// Load reconstructs an index from MarshalBinary's output — the //go:embed-an-index
// entry point. The returned *HNSW is query-ready and, like a freshly built one,
// read-only-safe for concurrent Query. The bytes are not retained (vectors are
// copied). Returns an error for a bad magic, an unsupported version, or any
// truncated/inconsistent blob — never a panic. Use LoadHNSWMmap (hnsw_mmap.go) to
// avoid the vector copy for a large embedded index.
func Load(data []byte) (*HNSW, error) {
	return loadHNSW(data, false)
}

// loadHNSW is Load's body, shared with LoadHNSWMmap: alias=false copies the
// vector block into the heap (Load's contract); alias=true aliases it directly
// from data instead (LoadHNSWMmap's contract — data is then the caller's mmap,
// kept alive by the returned *HNSW). Everything else — header validation, graph
// parsing, integrity checks — is identical either way, which is the point: two
// entry points, one parser, so they cannot silently diverge.
func loadHNSW(data []byte, alias bool) (*HNSW, error) {
	c := newHcur(data)
	if c.U32() != hnswMagic {
		return nil, errFormatf("ann: not an HNSW blob (bad magic)")
	}
	version := c.U32()
	if version != hnswVersion && version != hnswV4 {
		return nil, errFormatf("ann: unsupported HNSW format version %d (want %d or %d)", version, hnswVersion, hnswV4)
	}
	_ = c.U32() // reserved flags (v4) — read to advance the cursor, ignored: no flag bit is defined yet
	dim := c.readLen()
	ndocs := c.readLen()
	// match NewHNSW's clamp; a well-formed blob already has m ≥ 2
	m := max(c.cfg("m"), 2)
	m0 := c.cfg("m0")
	efc := c.cfg("efConstruction")
	efs := c.cfg("efSearch")
	entry := c.asInt()
	maxLayer := c.cfg("maxLayer")
	_ = c.U64() // consume the serialized mL to advance the cursor; recomputed below
	seed := c.U64()
	heuristic := c.U8() != 0
	int8mode := c.U8() != 0   // v3
	_ = c.Bytes(len(hnswPad)) // v4: alignment pad, discarded
	if c.Err != nil {
		return nil, c.Err
	}
	// Recompute mL from m rather than trust the serialized value: a crafted blob
	// could carry mL = +Inf/NaN/≤0, which randomLevel turns into an overflowing
	// level on Add-after-load (Load then Add is a documented capability) →
	// make([][]int32, level+1) panics. 1/ln(m) is deterministic and
	// round-trip-stable (NewHNSW stored exactly this), so a valid blob is
	// unaffected; Query never touches mL, so a load-only workflow was already safe.
	mL := 1.0 / math.Log(float64(m))

	var vecs [][]float32
	var bq []int8
	var scales []float32
	if int8mode {
		// Overflow-safe: the int8 codes + scales must fit the bytes before the
		// graph (computed in int64 so a hostile ndocs/dim can't wrap to a small
		// allocation).
		if int64(ndocs)*int64(dim)+int64(ndocs)*4 > int64(len(c.B)-c.Pos) {
			return nil, errFormatf("ann: HNSW int8 vector block (ndocs=%d dim=%d) exceeds remaining bytes", ndocs, dim)
		}
		if alias {
			// int8 has no alignment requirement, so aliasing straight from data is
			// safe regardless of offset — same reasoning as flat_i8_mmap.go.
			if raw := c.Bytes(ndocs * dim); len(raw) > 0 {
				bq = unsafe.Slice((*int8)(unsafe.Pointer(&raw[0])), len(raw))
			}
		} else {
			bq = c.int8s(ndocs * dim)
		}
		if version >= hnswVersion {
			_ = c.Bytes(pad4(ndocs * dim)) // v5: the scales start 4-byte aligned
		}
		if alias && version >= hnswVersion {
			scales = aliasU32s[float32](c, ndocs)
		} else {
			scales = make([]float32, ndocs)
			for i := range scales {
				scales[i] = c.F32()
			}
		}
	} else {
		// Overflow-safe product guard, mirroring the int8 branch above:
		// count() clamps ndocs and dim individually but not their product, so
		// without this a crafted ~1 MB header (dim≈ndocs≈250k) drives ~250 GB
		// of row allocations. 4 bytes per f32.
		if int64(ndocs)*int64(dim)*4 > int64(len(c.B)-c.Pos) {
			return nil, errFormatf("ann: HNSW f32 vector block (ndocs=%d dim=%d) exceeds remaining bytes", ndocs, dim)
		}
		vecs = make([][]float32, ndocs)
		if alias {
			// Alias the whole block as one []float32 arena, then sub-slice per row —
			// same row layout the copying path builds below, just backed by data
			// (the caller's mmap) instead of a fresh heap allocation. Safe ONLY
			// because the v4 header pads to an 8-byte boundary before this block
			// (see the format note above) — a float32 alias at an unaligned offset
			// is undefined behavior, unlike the int8 branch above.
			if raw := c.Bytes(ndocs * dim * 4); len(raw) > 0 {
				arena := unsafe.Slice((*float32)(unsafe.Pointer(&raw[0])), ndocs*dim)
				for d := range vecs {
					vecs[d] = arena[d*dim : (d+1)*dim : (d+1)*dim]
				}
			}
		} else {
			// One arena, sub-sliced per row, instead of ndocs separate allocations —
			// 1M rows was 1M allocations for a block that is contiguous in the blob
			// anyway. The guard above already proved ndocs*dim*4 fits the remaining
			// bytes, so this allocation is bounded by the input either way.
			arena := make([]float32, ndocs*dim)
			for d := range vecs {
				row := arena[d*dim : (d+1)*dim : (d+1)*dim]
				for j := range row {
					row[j] = c.F32()
				}
				vecs[d] = row
				if c.Err != nil { // stop reading once the input is exhausted
					return nil, c.Err
				}
			}
		}
	}
	if c.Err != nil {
		return nil, c.Err
	}
	var g *hnswCSR
	if version == hnswV4 {
		g = readGraphV4(c, ndocs)
	} else {
		g = readGraphV5(c, ndocs, alias)
	}
	if c.Err != nil {
		return nil, c.Err
	}
	if c.Pos != len(c.B) {
		return nil, errFormatf("ann: HNSW blob has %d trailing bytes", len(c.B)-c.Pos)
	}
	if err := validateGraph(g, ndocs, entry, maxLayer); err != nil {
		return nil, err
	}

	return &HNSW{
		vecs:           vecs,
		int8:           int8mode,
		bq:             bq,
		scales:         scales,
		csr:            g,
		dim:            dim,
		m:              m,
		m0:             m0,
		efConstruction: efc,
		efSearch:       efs,
		mL:             mL,
		entry:          entry,
		maxLayer:       maxLayer,
		seed:           seed,
		heuristic:      heuristic,
		rng:            rand.New(rand.NewPCG(seed, seed^rngSplit)),
	}, nil
}

// readGraphV4 reads v4's per-node graph runs into the CSR form. scanGraph sizes the three arrays exactly in one
// non-allocating pass; on a truncated blob it reports !ok and the arrays grow by append instead, and the read below
// ends in the error the cursor records.
func readGraphV4(c *hcur, ndocs int) *hnswCSR {
	nIDs, nLists, _ := scanGraph(c.B, c.Pos, ndocs)
	g := &hnswCSR{listStart: make([]uint32, 1, ndocs+1), idOff: make([]uint32, 1, nLists+1), ids: make([]int32, 0, nIDs)}
	for range ndocs {
		layer := c.readLen()
		for l := 0; l <= layer && c.Err == nil; l++ {
			cnt := c.readLen()
			for range cnt {
				g.ids = append(g.ids, int32(c.U32()))
			}
			g.idOff = append(g.idOff, uint32(len(g.ids)))
		}
		if c.Err != nil {
			return g
		}
		g.listStart = append(g.listStart, uint32(len(g.idOff)-1))
	}
	return g
}

// readGraphV5 reads v5's three CSR arrays, aliasing them from c's bytes (alias) or decoding copies. Each array's
// length comes from the one before it and is checked against the bytes that remain before anything is allocated;
// validateGraph checks their contents.
func readGraphV5(c *hcur, ndocs int, alias bool) *hnswCSR {
	read := func(n int) []uint32 {
		if c.Err == nil && n > c.Remaining()/4 {
			c.Err = errFormatf("ann: HNSW graph array of %d entries exceeds remaining bytes", n)
		}
		if c.Err != nil {
			return nil
		}
		if alias {
			return aliasU32s[uint32](c, n)
		}
		out := make([]uint32, n)
		for i := range out {
			out[i] = c.U32()
		}
		return out
	}
	g := &hnswCSR{listStart: read(ndocs + 1)}
	if c.Err != nil {
		return g
	}
	g.idOff = read(int(g.listStart[ndocs]) + 1)
	if c.Err != nil {
		return g
	}
	ids := read(int(g.idOff[len(g.idOff)-1]))
	if len(ids) > 0 {
		g.ids = unsafe.Slice((*int32)(unsafe.Pointer(&ids[0])), len(ids))
	}
	return g
}

// aliasU32s returns the next n 4-byte little-endian values of c as a []T aliasing c's bytes, which must be 4-byte
// aligned there (the v5 layout guarantees it relative to a page-aligned mapping). Like the f32 vector alias, it
// assumes a little-endian host.
func aliasU32s[T uint32 | float32](c *hcur, n int) []T {
	raw := c.Bytes(n * 4)
	if c.Err != nil || n == 0 {
		return nil
	}
	if uintptr(unsafe.Pointer(&raw[0]))%4 != 0 {
		c.Err = errFormatf("ann: HNSW blob section is not 4-byte aligned in the mapping")
		return nil
	}
	return unsafe.Slice((*T)(unsafe.Pointer(&raw[0])), n)
}

// validateGraph checks everything a query relies on, since Query indexes the arrays directly: offsets start at 0
// and never decrease, every node has at least its layer-0 list, the last offsets match the array lengths, every
// neighbor id is in range and exists at the layer that links it, and the entry point is consistent.
func validateGraph(g *hnswCSR, ndocs, entry, maxLayer int) error {
	if len(g.listStart) != ndocs+1 || g.listStart[0] != 0 {
		return errFormatf("ann: HNSW graph has a bad node table")
	}
	for d := range ndocs {
		if g.listStart[d+1] <= g.listStart[d] {
			return errFormatf("ann: HNSW node %d has no layer-0 list", d)
		}
	}
	if len(g.idOff) != int(g.listStart[ndocs])+1 || g.idOff[0] != 0 {
		return errFormatf("ann: HNSW graph has a bad list table")
	}
	for i := 1; i < len(g.idOff); i++ {
		if g.idOff[i] < g.idOff[i-1] {
			return errFormatf("ann: HNSW list %d ends before it starts", i-1)
		}
	}
	if int(g.idOff[len(g.idOff)-1]) != len(g.ids) {
		return errFormatf("ann: HNSW list table covers %d of %d ids", g.idOff[len(g.idOff)-1], len(g.ids))
	}
	layerOf := func(d int) int { return int(g.listStart[d+1]-g.listStart[d]) - 1 }
	if ndocs == 0 {
		if entry != -1 {
			return errFormatf("ann: empty HNSW must have entry -1, got %d", entry)
		}
		return nil
	}
	if entry < 0 || entry >= ndocs {
		return errFormatf("ann: HNSW entry %d out of [0,%d)", entry, ndocs)
	}
	if maxLayer > layerOf(entry) {
		return errFormatf("ann: HNSW maxLayer %d exceeds entry-node layer %d", maxLayer, layerOf(entry))
	}
	for d := range ndocs {
		for l := 0; l <= layerOf(d); l++ {
			for _, id := range g.list(g.listStart[d] + uint32(l)) {
				if id < 0 || int(id) >= ndocs {
					return errFormatf("ann: HNSW node %d layer %d neighbor id %d out of [0,%d)", d, l, id, ndocs)
				}
				if layerOf(int(id)) < l {
					return errFormatf("ann: HNSW node %d layer %d links node %d, which exists only to layer %d", d, l, id, layerOf(int(id)))
				}
			}
		}
	}
	return nil
}
