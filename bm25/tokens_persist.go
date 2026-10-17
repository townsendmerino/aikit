// Package bm25's tokens_persist.go is a cache for a TOKENIZED CORPUS — the
// [][]string that Tokenize / TokenizePlain produce and Build consumes — not a
// serialized Index. (Since v1.71.0 the Index has a saved format of its own,
// index_persist.go; this cache remains for callers who rebuild on load.)
//
// docs/internal/perf-amdahl-linux-amd64.md's W3 cold-start measurement (on
// examples/embedded-corpus's real 1,747-chunk corpus) found Build itself is
// roughly HALF of BM25's real cold-start cost: `bm25.TokenizePlain` at 8.83 ms
// and `bm25.Build` at 8.78 ms, of 17.6 ms combined. A versioned Index format
// (the original N4 proposal, docs/internal/roadmap.md §2.14) would eliminate
// the whole 17.6 ms, but commits bm25.Index to a permanent compatibility
// promise: every future Index field carried, defaulted, and gated for old
// files, forever — for a package whose Build is already cheap. N4 was
// deferred on exactly that tradeoff.
//
// MarshalTokens/UnmarshalTokens take the smaller, reversible half instead:
// caching tokenization skips re-tokenizing raw text, but Build still runs (at
// its own already-cheap cost) on load. bm25.Index itself gains no format,
// no version, no compatibility promise — this can be reworked or dropped
// entirely without touching Index at all. Tokenizer-agnostic by construction:
// the cache stores plain token strings, produced by Tokenize, TokenizePlain,
// or any other []string-per-document analyzer; it doesn't know or care which.
//
// Typical use (the //go:embed pattern):
//
//	blob, err := bm25.MarshalTokens(tokenizedDocs) // once, offline
//	// ... embed blob, or write it to disk ...
//	docs, err := bm25.UnmarshalTokens(blob)        // at startup
//	ix := bm25.Build(docs)                         // Build still runs
package bm25

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/townsendmerino/aikit/internal/cursor"
)

// ErrFormat is returned (wrapped) by UnmarshalTokens, ReadIndex and
// LoadIndexMmap when the input is not valid serialized data: a bad magic, an
// unsupported format version, or a truncated/inconsistent blob. Test with
// errors.Is:
//
//	docs, err := bm25.UnmarshalTokens(blob)
//	if errors.Is(err, bm25.ErrFormat) {
//		// the bytes are not a usable token cache (corrupt, or a newer format)
//	}
var ErrFormat = errors.New("bm25: malformed or unsupported serialized data")

func errFormatf(format string, args ...any) error {
	return fmt.Errorf(format+": %w", append(args, ErrFormat)...)
}

// Format-stability policy matches ann's (see README "Serialized blob
// formats"): magic-tagged and versioned from day one, rebuild-per-minor,
// UnmarshalTokens rejects any other version loudly rather than a silent
// misread. A reserved flags word sits right after the version so a later
// additive change can extend via flags without a version bump — the
// anti-churn mechanism ann/hnsw_persist.go's v4 bump added after not having
// it from day one; this format starts with it instead of needing a v2 later
// just to add it.
//
//	magic uint32 | version uint32 | flags uint32 (reserved, 0 today)
//	ndocs uint32
//	per doc: ntokens uint32, then per token: length uint32 + UTF-8 bytes
//
// All integers little-endian.
const (
	tokensMagic   uint32 = 0x424D3254 // "BM2T"
	tokensVersion uint32 = 1
)

// MarshalTokens serializes a tokenized corpus — one []string per document, as
// produced by Tokenize, TokenizePlain, or any other analyzer — into a
// versioned byte blob. UnmarshalTokens reverses it.
func MarshalTokens(docs [][]string) ([]byte, error) {
	if len(docs) > math.MaxInt32 {
		return nil, fmt.Errorf("bm25: corpus exceeds 2^31-1 documents")
	}
	size := 4 + 4 + 4 + 4 // magic + version + flags + ndocs
	for _, d := range docs {
		if len(d) > math.MaxInt32 {
			return nil, fmt.Errorf("bm25: document exceeds 2^31-1 tokens")
		}
		size += 4 // ntokens
		for _, t := range d {
			size += 4 + len(t)
		}
	}
	b := make([]byte, size)
	pos := 0
	putU32 := func(v uint32) {
		binary.LittleEndian.PutUint32(b[pos:], v)
		pos += 4
	}
	putU32(tokensMagic)
	putU32(tokensVersion)
	putU32(0) // reserved flags
	putU32(uint32(len(docs)))
	for _, d := range docs {
		putU32(uint32(len(d)))
		for _, t := range d {
			putU32(uint32(len(t)))
			pos += copy(b[pos:], t)
		}
	}
	return b, nil
}

// UnmarshalTokens reverses MarshalTokens. Returns an error wrapping
// ErrFormat for a bad magic, unsupported version, or truncated/inconsistent
// blob — never a panic, matching every other persisted format in this repo.
func UnmarshalTokens(data []byte) ([][]string, error) {
	c := &cursor.Cursor{B: data, Context: "bm25: token cache", Errorf: errFormatf}
	if c.U32() != tokensMagic {
		return nil, errFormatf("bm25: not a token cache blob (bad magic)")
	}
	if v := c.U32(); v != tokensVersion {
		return nil, errFormatf("bm25: unsupported token cache format version %d (want %d)", v, tokensVersion)
	}
	_ = c.U32() // reserved flags, ignored: no flag bit is defined yet
	ndocs := readCount(c, 4)
	if c.Err != nil {
		return nil, c.Err
	}
	docs := make([][]string, ndocs)
	for i := range docs {
		ntok := readCount(c, 4)
		if c.Err != nil {
			return nil, c.Err
		}
		if ntok == 0 {
			continue
		}
		toks := make([]string, ntok)
		for j := range toks {
			tlen := readCount(c, 1)
			if c.Err != nil {
				return nil, c.Err
			}
			raw := c.Bytes(tlen)
			if c.Err != nil {
				return nil, c.Err
			}
			toks[j] = string(raw)
		}
		docs[i] = toks
	}
	if c.Pos != len(c.B) {
		return nil, errFormatf("bm25: token cache blob has %d trailing bytes", len(c.B)-c.Pos)
	}
	return docs, nil
}

// readCount reads an allocation-driving length and rejects it if it can't
// fit the bytes that remain, so a hostile count can't drive a giant make()
// before the reads hit EOF. minElemBytes is the cheapest possible per-element
// encoding: 4 for an element count (ndocs, ntokens — each element costs at
// least its own 4-byte length/count prefix), 1 for a raw byte length (a
// single token may consume up to the whole rest of the blob, no per-byte
// structure to divide by).
func readCount(c *cursor.Cursor, minElemBytes int) int {
	v := int32(c.U32())
	if c.Err != nil {
		return 0
	}
	if v < 0 || int(v) > c.Remaining()/minElemBytes {
		c.Err = errFormatf("bm25: token cache count %d exceeds remaining bytes", v)
		return 0
	}
	return int(v)
}
