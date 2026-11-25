package embed

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"testing"
)

// dequantGoldenCase is one type's case in testdata/iq_dequant_golden.json: raw blocks and gguf-py's
// dequantization of them. Regenerate with scripts/oracle/pin_iq_dequant.py.
type dequantGoldenCase struct {
	Type     string    `json:"type"`
	GGMLType uint32    `json:"ggml_type"`
	Elems    int       `json:"elems"`
	RawHex   string    `json:"raw_hex"`
	Expected []float32 `json:"expected"`
}

func loadDequantGolden(t *testing.T) []dequantGoldenCase {
	t.Helper()
	raw, err := os.ReadFile("../testdata/iq_dequant_golden.json")
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g struct {
		Cases []dequantGoldenCase `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if len(g.Cases) == 0 {
		t.Fatal("golden has no cases")
	}
	return g.Cases
}

func dequantGoldenCaseOf(t *testing.T, typ string) dequantGoldenCase {
	t.Helper()
	for _, c := range loadDequantGolden(t) {
		if c.Type == typ {
			return c
		}
	}
	t.Fatalf("golden has no %s case", typ)
	return dequantGoldenCase{}
}

// dequant runs the case's raw bytes through dequantRange, the path every tensor read takes.
func (c dequantGoldenCase) dequant(t *testing.T) []float32 {
	t.Helper()
	blk, err := hex.DecodeString(c.RawHex)
	if err != nil {
		t.Fatalf("decode raw_hex: %v", err)
	}
	bs, ok := ggmlBlockElems(c.GGMLType)
	if !ok {
		t.Fatalf("type %d (%s) not supported", c.GGMLType, c.Type)
	}
	if len(c.Expected) != c.Elems {
		t.Fatalf("%s: golden has %d values for %d elems", c.Type, len(c.Expected), c.Elems)
	}
	got := make([]float32, c.Elems)
	dequantRange(c.GGMLType, blk, 0, got, bs)
	return got
}

// firstMismatch returns the first index whose bits differ from the reference, or -1.
func (c dequantGoldenCase) firstMismatch(got []float32) int {
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(c.Expected[i]) {
			return i
		}
	}
	return -1
}

// TestIQDequant_matchesReference holds every GGUF codebook and FP4 dequantizer bit-exact against
// llama.cpp's gguf Python reference (gguf.quants.dequantize) on the committed raw blocks.
func TestIQDequant_matchesReference(t *testing.T) {
	for _, c := range loadDequantGolden(t) {
		t.Run(c.Type, func(t *testing.T) {
			got := c.dequant(t)
			if i := c.firstMismatch(got); i >= 0 {
				t.Fatalf("%s[%d] = %v (%#08x), want %v (%#08x)", c.Type, i,
					got[i], math.Float32bits(got[i]), c.Expected[i], math.Float32bits(c.Expected[i]))
			}
			t.Logf("%s: %d values bit-identical to the gguf reference", c.Type, c.Elems)
		})
	}
}

// oneTensorGGUF builds a GGUF v3 file holding one 1-D tensor "w" of ggml type typ with n elements and the
// given data section.
func oneTensorGGUF(typ uint32, n int, data []byte) []byte {
	var b bytes.Buffer
	le := func(v any) { binary.Write(&b, binary.LittleEndian, v) }
	b.WriteString("GGUF")
	le(uint32(3)) // version
	le(uint64(1)) // tensor count
	le(uint64(0)) // KV count
	le(uint64(1))
	b.WriteString("w")
	le(uint32(1)) // n dims
	le(uint64(n))
	le(typ)
	le(uint64(0)) // offset
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	b.Write(data)
	return b.Bytes()
}

// TestIQDequant_goldenThroughGGUF reads each golden case as a GGUF tensor, so each type's byte size and
// block size are checked along with its kernel, and a data section one byte short is refused.
func TestIQDequant_goldenThroughGGUF(t *testing.T) {
	for _, c := range loadDequantGolden(t) {
		t.Run(c.Type, func(t *testing.T) {
			raw, err := hex.DecodeString(c.RawHex)
			if err != nil {
				t.Fatalf("decode raw_hex: %v", err)
			}
			g, err := OpenGGUFBytes(oneTensorGGUF(c.GGMLType, c.Elems, raw))
			if err != nil {
				t.Fatalf("open: %v", err)
			}
			_, got, err := g.Tensor("w")
			if err != nil {
				t.Fatalf("Tensor: %v", err)
			}
			if i := c.firstMismatch(got); i >= 0 {
				t.Fatalf("%s[%d] = %v, want %v", c.Type, i, got[i], c.Expected[i])
			}
			short, err := OpenGGUFBytes(oneTensorGGUF(c.GGMLType, c.Elems, raw[:len(raw)-1]))
			if err != nil {
				return // refused at open: as good as refused at read
			}
			if _, _, err := short.Tensor("w"); err == nil {
				t.Fatalf("%s: a data section one byte short was read", c.Type)
			}
		})
	}
}

// requirePlantCaught fails unless got (a dequantization with one planted defect) disagrees with the golden.
func requirePlantCaught(t *testing.T, c dequantGoldenCase, plant string, got []float32) {
	t.Helper()
	if c.firstMismatch(got) < 0 {
		t.Fatalf("%s: plant %q left the golden green", c.Type, plant)
	}
}

// plantTableEntry sets *entry to v for the duration of f. The tables are package state, so a test using
// it must not run in parallel with any other dequant test.
func plantTableEntry[T any](entry *T, v T, f func()) {
	saved := *entry
	*entry = v
	defer func() { *entry = saved }()
	f()
}

// TestNVFP4Dequant_goldenCatchesPlants: one wrong e2m1 entry, ModelOpt's sequential nibble order in place of
// ggml's split order, and a decode that honors the ue4m3 sign bit each turn the NVFP4 golden red.
func TestNVFP4Dequant_goldenCatchesPlants(t *testing.T) {
	c := dequantGoldenCaseOf(t, "NVFP4")

	var got []float32
	plantTableEntry(&mxfp4KValues[7], 11, func() { got = c.dequant(t) })
	requirePlantCaught(t, c, "e2m1 code 7 = 11", got)

	right := c.dequant(t)
	seq := make([]float32, len(right))
	for s := 0; s < len(right); s += 16 {
		for j := range 8 {
			seq[s+2*j] = right[s+j]
			seq[s+2*j+1] = right[s+j+8]
		}
	}
	requirePlantCaught(t, c, "sequential nibble order", seq)

	raw, err := hex.DecodeString(c.RawHex)
	if err != nil {
		t.Fatalf("decode raw_hex: %v", err)
	}
	signed := make([]float32, len(right))
	copy(signed, right)
	for blk := range len(raw) / 36 {
		for s := range 4 {
			if raw[blk*36+s]&0x80 == 0 {
				continue
			}
			for j := range 16 {
				signed[blk*64+16*s+j] = -signed[blk*64+16*s+j]
			}
		}
	}
	requirePlantCaught(t, c, "ue4m3 sign bit honored", signed)
}
