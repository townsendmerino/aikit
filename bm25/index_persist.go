package bm25

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"runtime"
	"slices"
	"unsafe"

	"github.com/townsendmerino/aikit/internal/cursor"
	"github.com/townsendmerino/aikit/mmap"
)

// The saved Index (memory campaign item 6, roadmap N4). WriteTo writes it; ReadIndex reads it into the heap;
// LoadIndexMmap maps the file and reads the postings arena — the bulk of an index — straight from the mapping,
// with no copy and no decoding. Everything else (the term dictionary, the per-document lengths) is decoded into
// the heap, because the scorer needs a map and float norms anyway and they are small.
//
// Format, version 1. All integers little-endian.
//
//	magic uint32 "BM2I" | version uint32 | flags uint32 (reserved, 0 today) | pad uint32 (0)
//	K1 float64 | B float64
//	ndocs uint64 | nterms uint64 | npostings uint64
//	docLen:   ndocs × uint32, then zero padding to the next multiple of 8
//	postings: npostings × (doc int32, tf int32) — starts 8-byte aligned; each term's list in dictionary order,
//	          doc-ordered within a list
//	dictionary: nterms × (df uint32 | maxTf uint32 | minLen uint32 | keyLen uint32 | key bytes)
//
// A term's list starts where the previous term's ends, so the dictionary stores no offsets. avgdl and the length
// norms are recomputed from docLen on load, with the expression Build uses, so a loaded index scores bit-identically.
//
// Validation. A load checks everything a query relies on, so a corrupt or hostile file is an error wrapping
// ErrFormat, never a panic or a wrong answer: every count against the bytes that remain, every posting's doc
// against ndocs and its tf as positive, each list strictly doc-ordered, each term's df, maxTf and minLen against
// its list, the lists' total against npostings, each document's length against the sum of its postings' tf (every
// token is counted in exactly one posting), and no duplicate key. What it cannot see is a term key's bytes changed
// into another valid key: that is a different vocabulary, not an inconsistent one. For a mapped load that read touches every
// page of the arena once.
//
// Version-skew policy: the format is versioned and magic-tagged; a load rejects an unknown version loudly
// (ErrFormat). A later version keeps reading version 1, because an index on disk outlives the release that wrote
// it. A reserved flags word lets an additive change extend version 1 without a new version.
const (
	indexMagic      uint32 = 0x49324D42 // "BM2I" read little-endian
	indexVersion    uint32 = 1
	indexHeaderSize        = 4*4 + 2*8 + 3*8 // 56, a multiple of 8
)

// docLenPad is the zero padding after the docLen section, so the postings arena starts 8-byte aligned.
func docLenPad(ndocs int) int { return (8 - (4*ndocs)%8) % 8 }

// WriteTo writes the index in the saved format (see ReadIndex and LoadIndexMmap). It implements io.WriterTo.
func (ix *Index) WriteTo(w io.Writer) (int64, error) {
	ix.checkOpen()
	defer runtime.KeepAlive(ix)
	if len(ix.docLen) > math.MaxInt32 {
		return 0, fmt.Errorf("bm25: corpus exceeds 2^31-1 documents")
	}
	ew := &errWriter{w: w}
	var hdr [indexHeaderSize]byte
	binary.LittleEndian.PutUint32(hdr[0:], indexMagic)
	binary.LittleEndian.PutUint32(hdr[4:], indexVersion)
	binary.LittleEndian.PutUint64(hdr[16:], math.Float64bits(ix.K1))
	binary.LittleEndian.PutUint64(hdr[24:], math.Float64bits(ix.B))
	binary.LittleEndian.PutUint64(hdr[32:], uint64(len(ix.docLen)))
	binary.LittleEndian.PutUint64(hdr[40:], uint64(len(ix.entries)))
	binary.LittleEndian.PutUint64(hdr[48:], uint64(len(ix.postings)))
	ew.write(hdr[:])

	buf := make([]byte, 0, 64<<10)
	flush := func() {
		ew.write(buf)
		buf = buf[:0]
	}
	for _, l := range ix.docLen {
		if l > math.MaxUint32 {
			return ew.n, fmt.Errorf("bm25: document of %d tokens exceeds the format's 2^32-1", l)
		}
		buf = binary.LittleEndian.AppendUint32(buf, uint32(l))
		if len(buf) >= cap(buf)-8 {
			flush()
		}
	}
	buf = append(buf, make([]byte, docLenPad(len(ix.docLen)))...)
	flush()
	// The arena goes out as one block in the little-endian layout posting already has on every supported host.
	if len(ix.postings) > 0 {
		ew.write(unsafe.Slice((*byte)(unsafe.Pointer(&ix.postings[0])), len(ix.postings)*int(sizeofPosting)))
	}
	// Dictionary in arena order: entry i's list follows entry i-1's.
	keys := make([]string, len(ix.entries))
	for k, i := range ix.terms {
		keys[i] = k
	}
	for i := range ix.entries {
		e := &ix.entries[i]
		buf = binary.LittleEndian.AppendUint32(buf, uint32(e.df))
		buf = binary.LittleEndian.AppendUint32(buf, uint32(e.maxTf))
		buf = binary.LittleEndian.AppendUint32(buf, uint32(e.minLen))
		buf = binary.LittleEndian.AppendUint32(buf, uint32(len(keys[i])))
		buf = append(buf, keys[i]...)
		if len(buf) >= cap(buf)/2 {
			flush()
		}
	}
	flush()
	return ew.n, ew.err
}

// errWriter is a sticky-error writer: after the first failure every write is a no-op.
type errWriter struct {
	w   io.Writer
	n   int64
	err error
}

func (e *errWriter) write(p []byte) {
	if e.err != nil || len(p) == 0 {
		return
	}
	n, err := e.w.Write(p)
	e.n += int64(n)
	e.err = err
}

// ReadIndex reads an index WriteTo wrote, into the heap (nothing of r is retained). An invalid stream is an error
// wrapping ErrFormat; see LoadIndexMmap for a load that does not copy the postings.
func ReadIndex(r io.Reader) (*Index, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	return decodeIndex(data, false)
}

// LoadIndexMmap memory-maps an index file WriteTo wrote and reads its postings arena straight from the mapping:
// no copy and no decoding, so the bulk of the index lives in the OS page cache rather than the Go heap. The term
// dictionary and document lengths are decoded into the heap. On non-unix platforms the file is heap-read instead
// (same result; see package mmap).
//
// Lifetime: the index reads from the mapping, so keep it for as long as you query it; Close releases the mapping,
// and a query after Close panics. A mapped index that becomes unreachable without Close is unmapped by a finalizer.
// Every Index method that reads the postings keeps the index reachable until it returns, so the finalizer cannot
// unmap the arena under a running query.
func LoadIndexMmap(path string) (*Index, error) {
	data, err := mmap.MapReadOnly(path)
	if err != nil {
		return nil, err
	}
	ix, err := decodeIndex(data, true)
	if err != nil {
		_ = mmap.Unmap(data) //nolint:errcheck // a failed load: the format error is the one to report
		return nil, err
	}
	ix.mmap = data
	runtime.SetFinalizer(ix, (*Index).finalizeMmap)
	return ix, nil
}

// Close releases a LoadIndexMmap index's mapping; the index must not be queried afterwards (a query panics). It is a
// no-op for an index from Build, Builder or ReadIndex, and idempotent.
func (ix *Index) Close() error {
	if ix.mmap == nil || ix.closed {
		return nil
	}
	ix.closed = true
	ix.postings = nil
	runtime.SetFinalizer(ix, nil)
	return mmap.Unmap(ix.mmap)
}

func (ix *Index) finalizeMmap() {
	if !ix.closed && ix.mmap != nil {
		_ = mmap.Unmap(ix.mmap) //nolint:errcheck // finalizer: nowhere to report
	}
}

// checkOpen panics on a closed mapped index, rather than reading unmapped memory.
func (ix *Index) checkOpen() {
	if ix.closed {
		panic("bm25: use of an Index after Close (its mapping is released)")
	}
}

// decodeIndex parses and validates the saved format. alias=true reads the postings arena from data in place (data
// must then outlive the index: it is the caller's mapping); alias=false copies it.
func decodeIndex(data []byte, alias bool) (*Index, error) {
	c := &cursor.Cursor{B: data, Context: "bm25: index", Errorf: errFormatf}
	if c.U32() != indexMagic {
		return nil, errFormatf("bm25: not a saved index (bad magic)")
	}
	if v := c.U32(); v != indexVersion {
		return nil, errFormatf("bm25: unsupported index format version %d (want %d)", v, indexVersion)
	}
	_ = c.U32() // reserved flags, ignored: no flag bit is defined yet
	_ = c.U32() // pad
	k1, b := c.F64(), c.F64()
	ndocs, nterms, npost := c.U64(), c.U64(), c.U64()
	if c.Err != nil {
		return nil, c.Err
	}
	// Every count must fit what remains before it drives an allocation (each doc costs 4 bytes, each posting 8,
	// each term at least 16).
	rem := uint64(c.Remaining())
	if ndocs > math.MaxInt32 || ndocs > rem/4 {
		return nil, errFormatf("bm25: index document count %d exceeds the file", ndocs)
	}
	if npost > rem/8 || nterms > rem/16 {
		return nil, errFormatf("bm25: index counts (%d terms, %d postings) exceed the file", nterms, npost)
	}
	ix := &Index{K1: k1, B: b, docLen: make([]int, ndocs), terms: make(map[string]int32, nterms), entries: make([]termEntry, nterms)}
	for d := range ix.docLen {
		ix.docLen[d] = int(c.U32())
	}
	_ = c.Bytes(docLenPad(int(ndocs)))
	raw := c.Bytes(int(npost) * int(sizeofPosting))
	if c.Err != nil {
		return nil, c.Err
	}
	if npost > 0 {
		if alias {
			if uintptr(unsafe.Pointer(&raw[0]))%unsafe.Alignof(posting{}) != 0 {
				return nil, errFormatf("bm25: index postings are not aligned in the mapping")
			}
			ix.postings = unsafe.Slice((*posting)(unsafe.Pointer(&raw[0])), npost)
		} else {
			ix.postings = make([]posting, npost)
			copy(unsafe.Slice((*byte)(unsafe.Pointer(&ix.postings[0])), len(raw)), raw)
		}
	}
	// unaccounted[d] counts down docLen[d] by each of d's postings' tf. Every token of a document is counted in
	// exactly one of its postings, so a valid file leaves every count at zero; that is what ties docLen (which the
	// norms come from) to the postings.
	unaccounted := slices.Clone(ix.docLen)
	var off int64
	for i := range ix.entries {
		df, maxTf, minLen, klen := c.U32(), c.U32(), c.U32(), c.U32()
		key := c.Bytes(int(klen))
		if c.Err != nil {
			return nil, c.Err
		}
		if df == 0 || uint64(off)+uint64(df) > npost {
			return nil, errFormatf("bm25: index term %d has df %d at offset %d of %d postings", i, df, off, npost)
		}
		e := termEntry{off: off, df: int32(df), maxTf: int32(maxTf), minLen: int32(minLen)}
		if err := checkList(ix.postings[off:off+int64(df)], &e, ix.docLen, unaccounted); err != nil {
			return nil, fmt.Errorf("bm25: index term %d: %w", i, err)
		}
		k := string(key)
		if _, dup := ix.terms[k]; dup {
			return nil, errFormatf("bm25: index term %q appears twice", k)
		}
		ix.terms[k] = int32(i)
		ix.entries[i] = e
		off += int64(df)
	}
	if uint64(off) != npost {
		return nil, errFormatf("bm25: index lists cover %d of %d postings", off, npost)
	}
	for d, n := range unaccounted {
		if n != 0 {
			return nil, errFormatf("bm25: index document %d has length %d but its postings' tf sum to %d", d, ix.docLen[d], ix.docLen[d]-n)
		}
	}
	if c.Pos != len(c.B) {
		return nil, errFormatf("bm25: index has %d trailing bytes", len(c.B)-c.Pos)
	}
	ix.setLengths()
	return ix, nil
}

// checkList validates one term's posting list against its dictionary entry: docs in range and strictly increasing,
// tf positive, and the entry's maxTf and minLen equal to the list's. It subtracts each posting's tf from
// unaccounted[doc] (see decodeIndex).
func checkList(list []posting, e *termEntry, docLen, unaccounted []int) error {
	maxTf, minLen := int32(0), int32(math.MaxInt32)
	prev := int32(-1)
	for _, p := range list {
		if p.doc <= prev || int(p.doc) >= len(docLen) || p.tf <= 0 {
			return errFormatf("posting (doc %d, tf %d) after doc %d is out of order or range", p.doc, p.tf, prev)
		}
		prev = p.doc
		unaccounted[p.doc] -= int(p.tf)
		maxTf = max(maxTf, p.tf)
		minLen = min(minLen, int32(min(docLen[p.doc], math.MaxInt32)))
	}
	if e.maxTf != maxTf || e.minLen != minLen {
		return errFormatf("maxTf %d / minLen %d do not match the list's %d / %d", e.maxTf, e.minLen, maxTf, minLen)
	}
	return nil
}
