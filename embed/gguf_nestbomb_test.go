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
	for i := 0; i < 80000; i++ {
		binary.Write(&b, binary.LittleEndian, uint32(9))
		binary.Write(&b, binary.LittleEndian, uint64(1000))
	}
	data := b.Bytes()

	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	OpenGGUFBytes(data) // must error or succeed, never blow up — and stay bounded
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
