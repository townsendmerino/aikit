package ann

import (
	"encoding/binary"
	"fmt"
	"math"
)

// FlatI8 serialization — the //go:embed-an-index entry point for the int8 index,
// the one you'd most want to embed (¼ the float32 memory at ~equal recall). Like
// the HNSW format it is versioned from day one so the on-disk layout can evolve
// without silently mis-reading old blobs:
//
//	magic uint32 | version uint32
//	dim int32 | n int32
//	codes:  n × dim int8 (row-major, one byte each)
//	scales: n × float32 (little-endian)
//
// All integers little-endian. LoadFlatI8 validates the payload size against the
// remaining bytes before allocating, so a corrupt or hostile blob returns an error
// rather than panicking or over-allocating.
const (
	flatI8Magic   uint32 = 0x46493800 // "FI8\0"
	flatI8Version uint32 = 1
)

// MarshalBinary serializes the int8 index (codes + per-vector scales + shape) into
// a versioned blob that LoadFlatI8 turns back into a query-ready *FlatI8. It
// implements encoding.BinaryMarshaler, so the index also round-trips through gob.
//
// The point is the //go:embed pattern: quantize the corpus once offline, embed the
// bytes, and Load at startup — no float32 vectors, no re-quantization per process.
func (f *FlatI8) MarshalBinary() ([]byte, error) {
	b := make([]byte, 0, 16+len(f.bq)+len(f.scales)*4)
	put32 := func(v uint32) { b = binary.LittleEndian.AppendUint32(b, v) }
	put32(flatI8Magic)
	put32(flatI8Version)
	put32(uint32(int32(f.dim)))
	put32(uint32(int32(f.n)))
	for _, code := range f.bq {
		b = append(b, byte(code)) // int8 → byte is the two's-complement round-trip
	}
	for _, s := range f.scales {
		put32(math.Float32bits(s))
	}
	return b, nil
}

// fcur is a bounds-checked little-endian reader over a FlatI8 blob (the per-format
// cursor convention, alongside HNSW's hcur and gguf's gcur). Every read goes
// through need(), so a truncated input sets err and yields zeros instead of
// panicking.
type fcur struct {
	b   []byte
	pos int
	err error
}

func (c *fcur) need(n int) bool {
	if c.err != nil {
		return false
	}
	if n < 0 || n > len(c.b)-c.pos {
		c.err = fmt.Errorf("ann: FlatI8 blob truncated (need %d at %d of %d)", n, c.pos, len(c.b))
		return false
	}
	return true
}

func (c *fcur) u32() uint32 {
	if !c.need(4) {
		return 0
	}
	v := binary.LittleEndian.Uint32(c.b[c.pos:])
	c.pos += 4
	return v
}

func (c *fcur) f32() float32 { return math.Float32frombits(c.u32()) }

func (c *fcur) u8() uint8 {
	if !c.need(1) {
		return 0
	}
	v := c.b[c.pos]
	c.pos++
	return v
}

// nonNeg reads an int32 shape scalar (dim, n) and rejects a negative value.
func (c *fcur) nonNeg(name string) int {
	v := int32(c.u32())
	if c.err != nil {
		return 0
	}
	if v < 0 {
		c.err = fmt.Errorf("ann: FlatI8 %s %d is negative", name, v)
		return 0
	}
	return int(v)
}

// LoadFlatI8 reconstructs an index from MarshalBinary's output. The returned
// *FlatI8 is query-ready and, like a freshly built one, read-only-safe for
// concurrent Query. The bytes are not retained. Returns an error for a bad magic,
// an unsupported version, or any truncated/inconsistent blob — never a panic.
func LoadFlatI8(data []byte) (*FlatI8, error) {
	c := &fcur{b: data}
	if c.u32() != flatI8Magic {
		return nil, fmt.Errorf("ann: not a FlatI8 blob (bad magic)")
	}
	if v := c.u32(); v != flatI8Version {
		return nil, fmt.Errorf("ann: unsupported FlatI8 format version %d (want %d)", v, flatI8Version)
	}
	dim := c.nonNeg("dim")
	n := c.nonNeg("n")
	if c.err != nil {
		return nil, c.err
	}
	if n > 0 && dim == 0 {
		return nil, fmt.Errorf("ann: FlatI8 has %d vectors but dim 0", n)
	}

	// Payload must be exactly n×dim code bytes + n×4 scale bytes. Computed in int64
	// so a hostile (n, dim) can't overflow into a small allocation; the exact-match
	// check also rejects truncation and trailing bytes in one shot.
	want := int64(n)*int64(dim) + int64(n)*4
	if got := int64(len(c.b) - c.pos); want != got {
		return nil, fmt.Errorf("ann: FlatI8 payload size %d != n*dim + n*4 = %d (n=%d dim=%d)", got, want, n, dim)
	}

	bq := make([]int8, n*dim)
	for i := range bq {
		bq[i] = int8(c.u8())
	}
	scales := make([]float32, n)
	for i := range scales {
		scales[i] = c.f32()
	}
	if c.err != nil {
		return nil, c.err
	}
	return &FlatI8{bq: bq, scales: scales, n: n, dim: dim}, nil
}
