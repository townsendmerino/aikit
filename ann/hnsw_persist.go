package ann

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
)

// HNSW serialization. The format is versioned from day one so the on-disk /
// //go:embed-ed layout can evolve without silently mis-reading old blobs:
//
//	magic uint32 | version uint32
//	dim, ndocs, m, m0, efConstruction, efSearch, entry, maxLayer  (int32 each)
//	mL float64 | seed uint64
//	vectors:  ndocs × dim float32 (little-endian, row-major)
//	graph:    per node — layer int32, then for l in 0..layer:
//	          nbrCount int32, then nbrCount × neighbor id int32
//
// All integers little-endian. entry is int32 (it is -1 for an empty index); every
// other count is non-negative. Load validates every length against the bytes that
// remain, so a corrupt or hostile blob returns an error rather than panicking or
// over-allocating.
const (
	hnswMagic   uint32 = 0x484E5357 // "HNSW"
	hnswVersion uint32 = 1
	// rngSplit matches NewHNSW's PCG seeding so a loaded index re-creates an
	// equivalently-seeded rng (for Add-after-load).
	rngSplit uint64 = 0x9e3779b97f4a7c15
)

// MarshalBinary serializes the built index — graph, vectors, and config — into a
// versioned byte blob that Load turns back into a query-ready *HNSW. It implements
// encoding.BinaryMarshaler, so the index also round-trips through gob and friends.
//
// The point is the //go:embed pattern: build the graph once offline, embed the
// bytes in the binary, and Load them at startup instead of rebuilding per process.
func (h *HNSW) MarshalBinary() ([]byte, error) {
	// Preallocate: header + vectors + graph (each neighbor id is 4 bytes).
	nNbr := 0
	for _, nd := range h.nodes {
		for _, l := range nd.nbrs {
			nNbr += len(l)
		}
	}
	b := make([]byte, 0, 64+len(h.vecs)*h.dim*4+len(h.nodes)*8+nNbr*4)

	put32 := func(v uint32) { b = binary.LittleEndian.AppendUint32(b, v) }
	puti := func(v int) { put32(uint32(int32(v))) } // counts + the (-1-capable) entry

	put32(hnswMagic)
	put32(hnswVersion)
	puti(h.dim)
	puti(len(h.vecs))
	puti(h.m)
	puti(h.m0)
	puti(h.efConstruction)
	puti(h.efSearch)
	puti(h.entry)
	puti(h.maxLayer)
	b = binary.LittleEndian.AppendUint64(b, math.Float64bits(h.mL))
	b = binary.LittleEndian.AppendUint64(b, h.seed)

	for _, v := range h.vecs {
		for _, f := range v {
			put32(math.Float32bits(f))
		}
	}
	for _, nd := range h.nodes {
		puti(nd.layer)
		for l := 0; l <= nd.layer; l++ {
			puti(len(nd.nbrs[l]))
			for _, id := range nd.nbrs[l] {
				put32(uint32(id))
			}
		}
	}
	return b, nil
}

// hcur is a bounds-checked little-endian reader over the serialized blob. Every
// read goes through need(), so a truncated or hostile input sets err and yields
// zeros instead of panicking; counts are additionally clamped to what the
// remaining bytes could hold before they drive an allocation.
type hcur struct {
	b   []byte
	pos int
	err error
}

func (c *hcur) need(n int) bool {
	if c.err != nil {
		return false
	}
	if n < 0 || n > len(c.b)-c.pos {
		c.err = fmt.Errorf("ann: HNSW blob truncated (need %d at %d of %d)", n, c.pos, len(c.b))
		return false
	}
	return true
}

func (c *hcur) u32() uint32 {
	if !c.need(4) {
		return 0
	}
	v := binary.LittleEndian.Uint32(c.b[c.pos:])
	c.pos += 4
	return v
}

func (c *hcur) u64() uint64 {
	if !c.need(8) {
		return 0
	}
	v := binary.LittleEndian.Uint64(c.b[c.pos:])
	c.pos += 8
	return v
}

func (c *hcur) f32() float32 { return math.Float32frombits(c.u32()) }

// asInt reads a signed int32 (used for entry, which is -1 for an empty index).
func (c *hcur) asInt() int { return int(int32(c.u32())) }

// count reads an allocation-driving LENGTH (ndocs, dim, neighbor count) and
// rejects it if it can't fit the bytes that remain (every subsequent element is
// ≥4 bytes), so a hostile count can't drive a giant make() before the reads hit
// EOF.
func (c *hcur) count() int {
	v := int32(c.u32())
	if c.err != nil {
		return 0
	}
	if v < 0 || int(v) > (len(c.b)-c.pos)/4 {
		c.err = fmt.Errorf("ann: HNSW blob count %d exceeds remaining bytes", v)
		return 0
	}
	return int(v)
}

// cfgMax bounds the config scalars (m, m0, efConstruction, efSearch). Unlike
// count() these aren't byte-sized lengths — they're tuning knobs — but efSearch
// sizes Query's candidate map, so an absurd value must not slip through and OOM.
const cfgMax = 1 << 20

// cfg reads a non-negative config scalar, rejecting negatives and absurd values.
func (c *hcur) cfg(name string) int {
	v := int32(c.u32())
	if c.err != nil {
		return 0
	}
	if v < 0 || int(v) > cfgMax {
		c.err = fmt.Errorf("ann: HNSW %s %d out of [0,%d]", name, v, cfgMax)
		return 0
	}
	return int(v)
}

// Load reconstructs an index from MarshalBinary's output — the //go:embed-an-index
// entry point. The returned *HNSW is query-ready and, like a freshly built one,
// read-only-safe for concurrent Query. The bytes are not retained (vectors are
// copied). Returns an error for a bad magic, an unsupported version, or any
// truncated/inconsistent blob — never a panic.
func Load(data []byte) (*HNSW, error) {
	c := &hcur{b: data}
	if c.u32() != hnswMagic {
		return nil, fmt.Errorf("ann: not an HNSW blob (bad magic)")
	}
	if v := c.u32(); v != hnswVersion {
		return nil, fmt.Errorf("ann: unsupported HNSW format version %d (want %d)", v, hnswVersion)
	}
	dim := c.count()
	ndocs := c.count()
	m := c.cfg("m")
	m0 := c.cfg("m0")
	efc := c.cfg("efConstruction")
	efs := c.cfg("efSearch")
	entry := c.asInt()
	maxLayer := c.cfg("maxLayer")
	mL := math.Float64frombits(c.u64())
	seed := c.u64()
	if c.err != nil {
		return nil, c.err
	}

	vecs := make([][]float32, ndocs)
	for d := range vecs {
		row := make([]float32, dim)
		for j := range row {
			row[j] = c.f32()
		}
		vecs[d] = row
	}
	nodes := make([]hnswNode, ndocs)
	for d := range nodes {
		layer := c.count()
		nbrs := make([][]int32, layer+1)
		for l := 0; l <= layer; l++ {
			cnt := c.count()
			ids := make([]int32, cnt)
			for i := range ids {
				ids[i] = int32(c.u32())
			}
			nbrs[l] = ids
		}
		nodes[d] = hnswNode{layer: layer, nbrs: nbrs}
	}
	if c.err != nil {
		return nil, c.err
	}
	if c.pos != len(c.b) {
		return nil, fmt.Errorf("ann: HNSW blob has %d trailing bytes", len(c.b)-c.pos)
	}

	// Validate graph integrity: Query indexes vecs[id] and nodes[id].nbrs[layer]
	// directly, so a blob with out-of-range ids or layer-inconsistent edges would
	// panic mid-query. Reject it here instead.
	if ndocs == 0 {
		if entry != -1 {
			return nil, fmt.Errorf("ann: empty HNSW must have entry -1, got %d", entry)
		}
	} else {
		if entry < 0 || entry >= ndocs {
			return nil, fmt.Errorf("ann: HNSW entry %d out of [0,%d)", entry, ndocs)
		}
		if maxLayer > nodes[entry].layer {
			return nil, fmt.Errorf("ann: HNSW maxLayer %d exceeds entry-node layer %d", maxLayer, nodes[entry].layer)
		}
	}
	for d := range nodes {
		for l := 0; l <= nodes[d].layer; l++ {
			for _, id := range nodes[d].nbrs[l] {
				if id < 0 || int(id) >= ndocs {
					return nil, fmt.Errorf("ann: HNSW node %d layer %d neighbor id %d out of [0,%d)", d, l, id, ndocs)
				}
				if nodes[id].layer < l {
					return nil, fmt.Errorf("ann: HNSW node %d layer %d links node %d, which exists only to layer %d", d, l, id, nodes[id].layer)
				}
			}
		}
	}

	return &HNSW{
		vecs:           vecs,
		nodes:          nodes,
		dim:            dim,
		m:              m,
		m0:             m0,
		efConstruction: efc,
		efSearch:       efs,
		mL:             mL,
		entry:          entry,
		maxLayer:       maxLayer,
		seed:           seed,
		rng:            rand.New(rand.NewPCG(seed, seed^rngSplit)),
	}, nil
}
