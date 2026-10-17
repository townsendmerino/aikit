package bm25

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"
)

// Item 6 (docs/internal/task-memory-campaign.md): Builder and the saved index.

// campaignBlob is the saved form of Build(campaignCorpus(20000, true)).
func campaignBlob(t testing.TB) []byte {
	t.Helper()
	var buf bytes.Buffer
	if _, err := Build(campaignCorpus(20000, true)).WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// writeTemp writes b to a fresh file and returns its path.
func writeTemp(t *testing.T, b []byte) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "index.bm25")
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// Gate (a): a Builder fed the documents one at a time gives results bit-identical to the pre-arena Build.
func TestBuilder_campaignHashMatchesBuild(t *testing.T) {
	b := NewBuilder()
	for _, d := range campaignCorpus(20000, true) {
		b.Add(d)
	}
	if b.N() != 20000 {
		t.Fatalf("N %d after 20,000 Adds", b.N())
	}
	if got := campaignHash(b.Build()); got != campaignHashBefore(t) {
		t.Fatalf("Builder hash %s, Build before the arena %s", got, campaignHashBefore(t))
	}
}

// heapInuseSampler records the peak HeapInuse while it runs.
func heapInuseSampler() (stop func() uint64) {
	var peak atomic.Uint64
	done, quit := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(done)
		var m runtime.MemStats
		for {
			runtime.ReadMemStats(&m)
			if m.HeapInuse > peak.Load() {
				peak.Store(m.HeapInuse)
			}
			select {
			case <-quit:
				return
			case <-time.After(2 * time.Millisecond):
			}
		}
	}()
	return func() uint64 { close(quit); <-done; return peak.Load() }
}

// Gate (b): the bytes a Builder build allocates are O(index), not O(corpus): TotalAlloc across the Adds and Build
// (the documents already exist, so they are not counted) ≤ 5 × the built index's bytes. Peak HeapInuse is reported
// beside Build's, which also holds the whole tokenized corpus.
func TestBuilder_allocatesOIndex(t *testing.T) {
	docs := campaignCorpus(20000, true)
	indexBytes := indexHeapBytes(docs)
	runtime.GC()
	var a, b runtime.MemStats
	runtime.ReadMemStats(&a)
	stop := heapInuseSampler()
	bl := NewBuilder()
	for _, d := range docs {
		bl.Add(d)
	}
	ix := bl.Build()
	peak := stop()
	runtime.ReadMemStats(&b)
	alloc := b.TotalAlloc - a.TotalAlloc
	ratio := float64(alloc) / float64(indexBytes)
	t.Logf("index %.2f MiB; Builder allocated %.2f MiB (%.2f×); peak HeapInuse %.2f MiB over a base of %.2f MiB",
		float64(indexBytes)/(1<<20), float64(alloc)/(1<<20), ratio, float64(peak)/(1<<20), float64(a.HeapInuse)/(1<<20))
	var corpus uint64
	for _, d := range docs {
		corpus += uint64(cap(d)) * uint64(unsafe.Sizeof(""))
	}
	t.Logf("for scale: the tokenized corpus Build needs at once holds %.2f MiB of string headers alone (the token bytes are shared vocabulary here)",
		float64(corpus)/(1<<20))
	if ratio > 5 {
		t.Fatalf("Builder allocated %.2f× the index's bytes, bar ≤ 5×", ratio)
	}
	runtime.KeepAlive(ix)
	runtime.KeepAlive(docs)
}

// Gate (c): a mapped load keeps no heap copy of the postings: the arena lies inside the mapping, and the heap the
// load adds is at most the dictionary and docLen bytes plus 1 MiB.
func TestLoadIndexMmap_postingsStayInTheMapping(t *testing.T) {
	blob := campaignBlob(t)
	p := writeTemp(t, blob)
	ndocs := int(leU64(blob[32:]))
	npost := int(leU64(blob[48:]))
	postStart := indexHeaderSize + 4*ndocs + docLenPad(ndocs)
	dictBytes := len(blob) - postStart - npost*8
	bound := uint64(dictBytes + 4*ndocs + 1<<20)

	runtime.GC()
	var a, b runtime.MemStats
	runtime.ReadMemStats(&a)
	ix, err := LoadIndexMmap(p)
	if err != nil {
		t.Fatal(err)
	}
	runtime.GC()
	runtime.ReadMemStats(&b)
	added := b.HeapAlloc - a.HeapAlloc
	t.Logf("postings %.2f MiB in the mapping; heap added by the load %.2f MiB (bound %.2f MiB: dictionary %.2f + docLen %.2f + 1)",
		float64(npost*8)/(1<<20), float64(added)/(1<<20), float64(bound)/(1<<20), float64(dictBytes)/(1<<20), float64(4*ndocs)/(1<<20))
	lo, hi := uintptr(unsafe.Pointer(&ix.mmap[0])), uintptr(unsafe.Pointer(&ix.mmap[0]))+uintptr(len(ix.mmap))
	if at := uintptr(unsafe.Pointer(&ix.postings[0])); at < lo || at >= hi {
		t.Fatal("the postings arena is not in the mapping")
	}
	if added > bound {
		t.Fatalf("a mapped load added %d B of heap, bound %d", added, bound)
	}
	if err := ix.Close(); err != nil {
		t.Fatal(err)
	}
}

func leU64(b []byte) uint64 {
	var v uint64
	for i := 7; i >= 0; i-- {
		v = v<<8 | uint64(b[i])
	}
	return v
}

// Gate (d): both loads give the built index's results bit for bit, and re-save to the same bytes.
func TestReadIndex_roundTrip(t *testing.T) {
	blob := campaignBlob(t)
	heap, err := ReadIndex(bytes.NewReader(blob))
	if err != nil {
		t.Fatal(err)
	}
	if got := campaignHash(heap); got != campaignHashBefore(t) {
		t.Fatalf("ReadIndex hash %s, want %s", got, campaignHashBefore(t))
	}
	mapped, err := LoadIndexMmap(writeTemp(t, blob))
	if err != nil {
		t.Fatal(err)
	}
	defer mapped.Close()
	if got := campaignHash(mapped); got != campaignHashBefore(t) {
		t.Fatalf("LoadIndexMmap hash %s, want %s", got, campaignHashBefore(t))
	}
	for name, ix := range map[string]*Index{"heap": heap, "mapped": mapped} {
		var again bytes.Buffer
		if _, err := ix.WriteTo(&again); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(again.Bytes(), blob) {
			t.Fatalf("%s: re-saved index differs from the original file", name)
		}
	}
}

// Edge shapes round-trip too: no documents, only empty documents, one document.
func TestReadIndex_edgeShapes(t *testing.T) {
	for name, docs := range map[string][][]string{
		"none":  nil,
		"empty": {nil, {}, nil},
		"one":   {{"a", "b", "a"}},
	} {
		ix := Build(docs)
		var buf bytes.Buffer
		if _, err := ix.WriteTo(&buf); err != nil {
			t.Fatal(err)
		}
		got, err := ReadIndex(&buf)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if got.N() != ix.N() || len(got.TopK([]string{"a"}, 5)) != len(ix.TopK([]string{"a"}, 5)) {
			t.Fatalf("%s: round trip differs", name)
		}
	}
}

// damageQueryHash is a cheap fingerprint for the damage test: TopK at k = 10 for the first 40 campaign queries.
func damageQueryHash(ix *Index) string {
	h := sha256.New()
	for _, q := range campaignQueries()[:40] {
		for _, r := range ix.TopK(q, 10) {
			hashInt(h, r.Doc)
			hashF64(h, r.Score)
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

// Gate (d): a truncated or corrupted file is an error wrapping ErrFormat, never a panic. On a 12-document index: every
// truncation point and a byte flip at every offset of the header, docLen and dictionary, and at every 7th offset of
// the postings block (7 is coprime with the 8-byte posting, so every byte of the struct is still hit) — which keeps
// the sweep to seconds under -race. A flip the loader accepts
// must be one no query can tell apart (K1/B are read back as written and so excluded), or a term key's bytes: that
// is a different but valid vocabulary, which no format can detect without a checksum. Anything else accepted is a
// validation gap and fails.
func TestReadIndex_rejectsDamage(t *testing.T) {
	var buf bytes.Buffer
	if _, err := Build(campaignCorpus(12, true)).WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	blob := buf.Bytes()
	ndocs, npost := int(leU64(blob[32:])), int(leU64(blob[48:]))
	dictStart := indexHeaderSize + 4*ndocs + docLenPad(ndocs) + 8*npost
	postStart := dictStart - 8*npost
	offsets := func(yield func(int) bool) {
		for off := 0; off < len(blob); off++ {
			if off >= postStart && off < dictStart && (off-postStart)%7 != 0 {
				continue
			}
			if !yield(off) {
				return
			}
		}
	}
	// decodeIndex directly (ReadIndex is io.ReadAll then this), so no case copies the file.
	for n := range offsets {
		if _, err := decodeIndex(blob[:n], false); !errors.Is(err, ErrFormat) {
			t.Fatalf("truncated to %d of %d bytes: err %v, want ErrFormat", n, len(blob), err)
		}
	}
	good, err := ReadIndex(bytes.NewReader(blob))
	if err != nil {
		t.Fatal(err)
	}
	want := damageQueryHash(good)
	keyFlips := 0
	for off := range offsets {
		blob[off] ^= 0x5a // flipped in place and restored below
		ix, err := decodeIndex(blob, false)
		blob[off] ^= 0x5a
		if err != nil {
			if !errors.Is(err, ErrFormat) {
				t.Fatalf("flip at %d: err %v, want ErrFormat", off, err)
			}
			continue
		}
		switch {
		case off >= 8 && off < 32: // flags, pad (ignored), K1, B (a valid tuning, read back as written)
		case damageQueryHash(ix) == want:
		case off >= dictStart:
			keyFlips++ // a renamed term: valid, just a different vocabulary
		default:
			t.Fatalf("flip at offset %d (before the dictionary at %d) was accepted and changes results", off, dictStart)
		}
	}
	t.Logf("%d-byte file: every truncation rejected; %d accepted flips rename a term", len(blob), keyFlips)
}

// Close releases the mapping; a query afterwards panics instead of reading unmapped memory; Close is idempotent and
// a no-op on a built index.
func TestLoadIndexMmap_close(t *testing.T) {
	ix, err := LoadIndexMmap(writeTemp(t, campaignBlob(t)))
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.Close(); err != nil {
		t.Fatal(err)
	}
	if err := ix.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("TopK after Close did not panic")
			}
		}()
		ix.TopK([]string{"t0000"}, 10)
	}()
	if err := Build(campaignCorpus(10, false)).Close(); err != nil {
		t.Fatalf("Close on a built index: %v", err)
	}
	if _, err := LoadIndexMmap(filepath.Join(t.TempDir(), "missing")); err == nil || errors.Is(err, ErrFormat) {
		t.Fatalf("missing file: err %v, want an I/O error", err)
	}
}

// FuzzReadIndex: no input makes ReadIndex panic.
func FuzzReadIndex(f *testing.F) {
	var buf bytes.Buffer
	if _, err := Build(campaignCorpus(40, true)).WriteTo(&buf); err != nil {
		f.Fatal(err)
	}
	f.Add(buf.Bytes())
	f.Add(buf.Bytes()[:indexHeaderSize])
	f.Add([]byte{})
	f.Fuzz(func(t *testing.T, b []byte) {
		ix, err := ReadIndex(bytes.NewReader(b))
		if err != nil {
			return
		}
		ix.TopK([]string{"t0000", "never"}, 5)
		_ = ix.Scores([]string{"t0001"})
		if math.IsNaN(ix.avgdl) {
			t.Fatal("NaN avgdl")
		}
	})
}
