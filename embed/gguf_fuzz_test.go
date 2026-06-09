package embed

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// minimalGGUF builds the smallest well-formed GGUF: magic, version 3, zero
// tensors, zero KVs, padded to the 32-byte data alignment. A valid seed the
// fuzzer mutates from.
func minimalGGUF() []byte {
	var b bytes.Buffer
	b.WriteString("GGUF")
	binary.Write(&b, binary.LittleEndian, uint32(3)) // version
	binary.Write(&b, binary.LittleEndian, uint64(0)) // tensorCount
	binary.Write(&b, binary.LittleEndian, uint64(0)) // kvCount
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	return b.Bytes()
}

// ggufWithContent builds a valid GGUF carrying one string KV, one array KV, and
// one 2-D tensor — so the seed exercises the str/value/array/dims parse paths
// the fuzzer then mutates (lengths, counts, types).
func ggufWithContent() []byte {
	var b bytes.Buffer
	str := func(s string) {
		binary.Write(&b, binary.LittleEndian, uint64(len(s)))
		b.WriteString(s)
	}
	b.WriteString("GGUF")
	binary.Write(&b, binary.LittleEndian, uint32(3)) // version
	binary.Write(&b, binary.LittleEndian, uint64(1)) // tensorCount
	binary.Write(&b, binary.LittleEndian, uint64(2)) // kvCount

	// KV 1: string
	str("general.name")
	binary.Write(&b, binary.LittleEndian, uint32(ggufString))
	str("demo")
	// KV 2: array of 2 uint32
	str("some.array")
	binary.Write(&b, binary.LittleEndian, uint32(ggufArray))
	binary.Write(&b, binary.LittleEndian, uint32(ggufUint32)) // element type
	binary.Write(&b, binary.LittleEndian, uint64(2))          // count
	binary.Write(&b, binary.LittleEndian, uint32(7))
	binary.Write(&b, binary.LittleEndian, uint32(9))

	// One tensor: 2 dims, type 0 (F32), offset 0
	str("weight")
	binary.Write(&b, binary.LittleEndian, uint32(2)) // n dims
	binary.Write(&b, binary.LittleEndian, uint64(4))
	binary.Write(&b, binary.LittleEndian, uint64(3))
	binary.Write(&b, binary.LittleEndian, uint32(0)) // type
	binary.Write(&b, binary.LittleEndian, uint64(0)) // offset
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	binary.Write(&b, binary.LittleEndian, make([]byte, 4*3*4)) // f32 data section
	return b.Bytes()
}

// FuzzParseGGUF asserts the GGUF header/metadata/tensor-directory parser never
// panics on arbitrary bytes — it must return an error or a usable file. These
// parse untrusted, possibly hostile files; the contract is "no panic, ever".
// (Dequant-path fuzzing via Tensor()/RowDequantizer is a separate target — see
// the roadmap §3.1 follow-up.)
func FuzzParseGGUF(f *testing.F) {
	f.Add(minimalGGUF())
	f.Add(ggufWithContent())
	f.Add([]byte("GGUF"))                           // truncated right after magic
	f.Add([]byte("GGUF\x03\x00\x00\x00"))           // magic + version, then EOF
	f.Add([]byte("not a gguf file at all, really")) // bad magic
	f.Add([]byte{})                                 // empty

	f.Fuzz(func(t *testing.T, data []byte) {
		g, err := OpenGGUFBytes(data)
		if err != nil {
			return
		}
		// A successful parse must expose a consistent, panic-free surface.
		for _, n := range g.Names() {
			if !g.Has(n) {
				t.Fatalf("Names() returned %q but Has(%q)=false", n, n)
			}
			g.Dims(n)
		}
	})
}
