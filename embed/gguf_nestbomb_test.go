package embed

import (
	"bytes"
	"encoding/binary"
	"runtime"
	"testing"
)

// TestParseGGUF_nestedArrayBomb guards the metadata array parser against the
// nested-array allocation blowup: a deeply nested array-of-arrays where every
// level claims a count near the remaining input. Each level's []any preallocation
// was bounded only by the remaining bytes, and the nesting depth is itself
// ~input/12 — so make([]any, 0, n) per level drove O(input²) allocation, parsing a
// ~1 MB file in ~700 ms (the FuzzParseGGUF "context deadline exceeded" slow path).
// With the capped preallocation, total allocation is linear in the input. We assert
// on bytes allocated (not wall time) so the gate is deterministic.
func TestParseGGUF_nestedArrayBomb(t *testing.T) {
	var b bytes.Buffer
	b.WriteString("GGUF")
	binary.Write(&b, binary.LittleEndian, uint32(3))
	binary.Write(&b, binary.LittleEndian, uint64(0)) // tensorCount
	binary.Write(&b, binary.LittleEndian, uint64(1)) // kvCount
	binary.Write(&b, binary.LittleEndian, uint64(1)) // key len
	b.WriteByte('x')
	binary.Write(&b, binary.LittleEndian, uint32(9)) // value type = array
	// ~960 KB of (et=array, n=1000) headers — each a nesting level claiming 1000.
	for range 80000 {
		binary.Write(&b, binary.LittleEndian, uint32(9))
		binary.Write(&b, binary.LittleEndian, uint64(1000))
	}
	data := b.Bytes()

	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	_, _ = OpenGGUFBytes(data) // must error or succeed, never blow up — and stay bounded
	runtime.ReadMemStats(&after)

	alloc := after.TotalAlloc - before.TotalAlloc
	// Quadratic was ~1 GB+ for this input; the linear path is tens of MB. 256 MB is
	// a wide ceiling that still fails hard on any return of the O(input²) behavior.
	const ceiling = 256 << 20
	t.Logf("input=%d B, allocated=%d MB", len(data), alloc>>20)
	if alloc > ceiling {
		t.Fatalf("parse allocated %d MB for a %d KB input — nested-array allocation blowup",
			alloc>>20, len(data)>>10)
	}
}

// TestParseGGUF_arrayDepthCap (H4) guards recursion DEPTH (distinct from the
// allocation blowup above): an array-of-arrays chain deeper than
// ggufMaxArrayDepth must return a clean error, not recurse until the goroutine
// stack aborts the process. A few hundred levels can't crash on its own, but it
// exercises the cap deterministically — the guard must fire regardless of depth.
func TestParseGGUF_arrayDepthCap(t *testing.T) {
	var b bytes.Buffer
	b.WriteString("GGUF")
	binary.Write(&b, binary.LittleEndian, uint32(3))
	binary.Write(&b, binary.LittleEndian, uint64(0)) // tensorCount
	binary.Write(&b, binary.LittleEndian, uint64(1)) // kvCount
	binary.Write(&b, binary.LittleEndian, uint64(1)) // key len
	b.WriteByte('x')
	binary.Write(&b, binary.LittleEndian, uint32(9)) // value type = array
	// ggufMaxArrayDepth (128) + slack levels of (et=array, n=1): the cap must
	// fire before the innermost element is ever reached.
	for range ggufMaxArrayDepth + 64 {
		binary.Write(&b, binary.LittleEndian, uint32(9)) // element type = array
		binary.Write(&b, binary.LittleEndian, uint64(1)) // count = 1
	}
	if _, err := OpenGGUFBytes(b.Bytes()); err == nil {
		t.Fatal("deep array nesting: OpenGGUFBytes returned nil, want a depth-cap error")
	}
}
