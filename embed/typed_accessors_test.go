package embed

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"testing"
	"testing/fstest"
	"unsafe"
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

// TestParseSafetensors_shapeDtypeCrossValidation (H2): the header's declared
// shape × dtype must match the byte range, so a hostile file can't pair a giant
// shape with a tiny byte range (which parsed before, then panicked at inference
// when a caller indexed by shape). Unknown dtypes are exempt (rejected later at
// read); negative dims and shape-product overflow are rejected too.
func TestParseSafetensors_shapeDtypeCrossValidation(t *testing.T) {
	load := func(e stEntry) error {
		_, err := parseSafetensors(buildSafetensors(map[string]stEntry{"w": e}))
		return err
	}

	// The H2 exemplar: shape [4096,4096] F32 but only 4 bytes of payload.
	if err := load(stEntry{"F32", []int{4096, 4096}, f32raw(1)}); !errors.Is(err, ErrFormat) {
		t.Errorf("giant shape / tiny bytes: want ErrFormat, got %v", err)
	}
	// Off-by-one element count.
	if err := load(stEntry{"F32", []int{2, 3}, f32raw(1, 2, 3, 4, 5)}); !errors.Is(err, ErrFormat) {
		t.Errorf("shape 2×3 with 5 elems: want ErrFormat, got %v", err)
	}
	// Negative dim.
	if err := load(stEntry{"F32", []int{-1, 3}, f32raw(1, 2, 3)}); !errors.Is(err, ErrFormat) {
		t.Errorf("negative dim: want ErrFormat, got %v", err)
	}
	// Shape-product overflow (each dim in-range for int, product wraps).
	if err := load(stEntry{"F32", []int{1 << 40, 1 << 40}, f32raw(1)}); !errors.Is(err, ErrFormat) {
		t.Errorf("shape overflow: want ErrFormat, got %v", err)
	}
	// Valid tensors still parse: exact match, an empty (0-element) tensor, and a
	// scalar (empty shape → 1 element).
	for _, ok := range []stEntry{
		{"F32", []int{2, 3}, f32raw(1, 2, 3, 4, 5, 6)},
		{"F32", []int{0}, nil},
		{"F32", []int{}, f32raw(42)},
		{"I64", []int{2}, make([]byte, 16)},
	} {
		if err := load(ok); err != nil {
			t.Errorf("valid tensor %v/%v: unexpected error %v", ok.dtype, ok.shape, err)
		}
	}
	// Unknown dtype is exempt from the byte-range check (parses; rejected at read).
	if err := load(stEntry{"BOOL", []int{2, 3}, []byte{1}}); err != nil {
		t.Errorf("unknown dtype should skip the shape check, got %v", err)
	}
}

// TestReinterpretLE_misalignedCopies (H3): a tensor whose bytes are not
// element-aligned must decode via a copy, not a misaligned unsafe.Pointer
// conversion (an unrecoverable checkptr/-race throw, SIGBUS on strict-alignment
// ports). Run under -race to exercise checkptr on the fast path. The copy must
// be value-correct and independent of the source bytes.
func TestReinterpretLE_misalignedCopies(t *testing.T) {
	want := []float32{1, -2, 3.5, 4, 5, 6, 7, 8}
	enc := make([]byte, 4*len(want))
	for i, v := range want {
		binary.LittleEndian.PutUint32(enc[4*i:], math.Float32bits(v))
	}
	// Slice a padded buffer at a start that is not 4-aligned.
	buf := make([]byte, len(enc)+8)
	off := 0
	for off < 8 && uintptr(unsafe.Pointer(&buf[off]))%4 == 0 {
		off++
	}
	raw := buf[off : off+len(enc)]
	if uintptr(unsafe.Pointer(&raw[0]))%4 == 0 {
		t.Skip("could not obtain a misaligned buffer on this allocator")
	}
	copy(raw, enc)

	got, err := reinterpretLE[float32]("t", raw)
	if err != nil {
		t.Fatalf("reinterpretLE(misaligned): %v", err)
	}
	if !eqF32(got, want) {
		t.Fatalf("misaligned decode = %v, want %v", got, want)
	}
	// It must be a copy: mutating the source doesn't change the result.
	raw[0] ^= 0xFF
	if !eqF32(got, want) {
		t.Errorf("result aliases the misaligned source; want an independent copy")
	}

	// Aligned input still takes the zero-copy view (and aliases).
	al, err := reinterpretLE[float32]("t", enc)
	if err != nil {
		t.Fatalf("reinterpretLE(aligned): %v", err)
	}
	if uintptr(unsafe.Pointer(&enc[0]))%4 == 0 {
		enc[0] ^= 0xFF
		if al[0] == want[0] {
			t.Errorf("aligned path should return a view that aliases the source")
		}
	}
}

// TestTensor_ElementsOverflow (F2): Elements() returns -1 on a shape whose
// product overflows int (reachable for an unknown-dtype tensor that skips the H2
// check), not a silently wrapped value.
func TestTensor_ElementsOverflow(t *testing.T) {
	if got := (Tensor{Shape: []int{1 << 40, 1 << 40}}).Elements(); got != -1 {
		t.Errorf("overflow shape: Elements() = %d, want -1", got)
	}
	if got := (Tensor{Shape: []int{-1, 4}}).Elements(); got != -1 {
		t.Errorf("negative dim: Elements() = %d, want -1", got)
	}
	if got := (Tensor{Shape: []int{2, 3, 4}}).Elements(); got != 24 {
		t.Errorf("valid shape: Elements() = %d, want 24", got)
	}
	if got := (Tensor{Shape: []int{5, 0}}).Elements(); got != 0 {
		t.Errorf("zero dim: Elements() = %d, want 0", got)
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
