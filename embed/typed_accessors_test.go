package embed

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"testing"
	"testing/fstest"
)

// stEntry is one tensor for the multi-dtype builder below.
type stEntry struct {
	dtype string
	shape []int
	raw   []byte
}

// buildSafetensors writes a safetensors blob with arbitrary dtypes (the sharded test's
// builder is F32-only). Names are header keys; payload is concatenated in name order.
func buildSafetensors(t map[string]stEntry) []byte {
	names := make([]string, 0, len(t))
	for n := range t {
		names = append(names, n)
	}
	for i := 1; i < len(names); i++ {
		for j := i; j > 0 && names[j] < names[j-1]; j-- {
			names[j], names[j-1] = names[j-1], names[j]
		}
	}
	header := map[string]any{}
	var payload []byte
	off := 0
	for _, n := range names {
		e := t[n]
		header[n] = map[string]any{"dtype": e.dtype, "shape": e.shape, "data_offsets": []int{off, off + len(e.raw)}}
		payload = append(payload, e.raw...)
		off += len(e.raw)
	}
	hb, _ := json.Marshal(header)
	out := make([]byte, 8, 8+len(hb)+len(payload))
	binary.LittleEndian.PutUint64(out, uint64(len(hb)))
	out = append(out, hb...)
	return append(out, payload...)
}

func f32raw(v ...float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return b
}
func i32raw(v ...int32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], uint32(x))
	}
	return b
}
func f16raw(bits ...uint16) []byte { // caller passes IEEE-754 half bit patterns
	b := make([]byte, 2*len(bits))
	for i, x := range bits {
		binary.LittleEndian.PutUint16(b[2*i:], x)
	}
	return b
}

func TestSafetensorsFile_typedAccessors(t *testing.T) {
	blob := buildSafetensors(map[string]stEntry{
		"w":    {"F32", []int{2, 3}, f32raw(1, 2, 3, 4, 5, 6)},
		"half": {"F16", []int{3}, f16raw(0x3C00, 0x4000, 0xBC00)}, // 1, 2, -1
		"ids":  {"I32", []int{3}, i32raw(7, -8, 9)},
	})
	sf, err := OpenSafetensorsFromFS(fstest.MapFS{"m.safetensors": &fstest.MapFile{Data: blob}}, "m.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	defer sf.Close()

	// F32 read, with and without a shape check.
	got, err := sf.TensorF32("w", 2, 3)
	if err != nil {
		t.Fatalf("TensorF32(w, 2,3): %v", err)
	}
	if want := []float32{1, 2, 3, 4, 5, 6}; !eqF32(got, want) {
		t.Errorf("TensorF32(w)=%v want %v", got, want)
	}
	if _, err := sf.TensorF32("w"); err != nil { // no want → no check
		t.Errorf("TensorF32(w) no-want: %v", err)
	}
	// Wrong shape → error.
	if _, err := sf.TensorF32("w", 3, 2); err == nil {
		t.Error("TensorF32(w, 3,2): expected shape mismatch error")
	}
	// Missing tensor → error.
	if _, err := sf.TensorF32("nope"); err == nil {
		t.Error("TensorF32(nope): expected not-found error")
	}
	// F16 widens to f32.
	if got, err := sf.TensorF32("half", 3); err != nil || !eqF32(got, []float32{1, 2, -1}) {
		t.Errorf("TensorF32(half)=%v err=%v want [1 2 -1]", got, err)
	}
	// I32 read.
	if ids, err := sf.TensorI32("ids", 3); err != nil || ids[0] != 7 || ids[1] != -8 || ids[2] != 9 {
		t.Errorf("TensorI32(ids)=%v err=%v want [7 -8 9]", ids, err)
	}
	// F32 read of an I32 tensor → unsupported-dtype error.
	if _, err := sf.TensorF32("ids"); err == nil {
		t.Error("TensorF32(ids): expected unsupported-dtype error")
	}
	// errors.Is sanity: these are plain (not ErrFormat — they're caller/lookup errors).
	_, e := sf.TensorF32("nope")
	if errors.Is(e, ErrFormat) {
		t.Error("not-found should not be ErrFormat")
	}
}

func eqF32(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
