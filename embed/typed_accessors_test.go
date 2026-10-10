package embed

import (
	"bytes"
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
func bf16raw(v ...float32) []byte { // bf16 = high 16 bits of the f32 pattern
	b := make([]byte, 2*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint16(b[2*i:], uint16(math.Float32bits(x)>>16))
	}
	return b
}

// TestSafetensorsFile_SubF32 gates the per-slice widener: SubF32(start,count) reads
// a contiguous element range without materializing the whole tensor, across F32 (a
// zero-copy view), BF16, and F16. This is what lets a fused [nExpert, …] weight be
// widened/quantized one expert at a time. Values (1..6) are integers, exact in all
// three dtypes, so the per-range read must equal the whole-tensor read sliced.
func TestSafetensorsFile_SubF32(t *testing.T) {
	blob := buildSafetensors(map[string]stEntry{
		"f32":  {"F32", []int{2, 3}, f32raw(1, 2, 3, 4, 5, 6)}, // 2 "experts" of 3
		"bf16": {"BF16", []int{2, 3}, bf16raw(1, 2, 3, 4, 5, 6)},
		"f16":  {"F16", []int{2, 3}, f16raw(0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600)}, // 1..6
	})
	sf, err := OpenSafetensorsFromFS(fstest.MapFS{"m.safetensors": &fstest.MapFile{Data: blob}}, "m.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	defer sf.Close()

	for _, name := range []string{"f32", "bf16", "f16"} {
		tn, err := sf.Tensor(name)
		if err != nil {
			t.Fatalf("Tensor(%s): %v", name, err)
		}
		// Each "expert" row [3 elements]; the concatenation of the per-row SubF32 reads
		// must equal the whole tensor.
		var got []float32
		for e := range 2 {
			row, err := tn.SubF32(e*3, 3)
			if err != nil {
				t.Fatalf("%s SubF32(%d,3): %v", name, e*3, err)
			}
			if len(row) != 3 {
				t.Fatalf("%s SubF32 len=%d want 3", name, len(row))
			}
			got = append(got, row...)
		}
		if want := []float32{1, 2, 3, 4, 5, 6}; !eqF32(got, want) {
			t.Errorf("%s per-row SubF32 = %v, want %v", name, got, want)
		}
		// Out-of-range must error, not panic.
		if _, err := tn.SubF32(4, 3); err == nil {
			t.Errorf("%s SubF32(4,3): expected out-of-bounds error", name)
		}
		if _, err := tn.SubF32(-1, 2); err == nil {
			t.Errorf("%s SubF32(-1,2): expected out-of-bounds error", name)
		}
	}

	// F32 SubF32 sub-slices the full read (a view when the mapping is 4-aligned, like
	// Float32s; a copy otherwise — alignment isn't guaranteed for a heap MapFS). Either
	// way the sub-range must equal the corresponding slice of the whole tensor.
	tn, _ := sf.Tensor("f32")
	full, _ := tn.Float32s()
	sub, _ := tn.SubF32(3, 3)
	if !eqF32(sub, full[3:6]) {
		t.Errorf("F32 SubF32(3,3)=%v want %v", sub, full[3:6])
	}
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

// TestUint8s covers the byte-wide accessor added for block-quantized payloads
// (MXFP4 safetensors ships packed 4-bit codes in a U8 `*_blocks` tensor and their
// E8M0 exponents in a separate U8 `*_scales` tensor).
func TestUint8s(t *testing.T) {
	blob := buildSafetensors(map[string]stEntry{
		"blocks": {dtype: "U8", shape: []int{2, 3}, raw: []byte{1, 2, 3, 250, 251, 252}},
		"flags":  {dtype: "BOOL", shape: []int{2}, raw: []byte{0, 1}},
		"signed": {dtype: "I8", shape: []int{2}, raw: []byte{0x80, 0x7f}},
		"floats": {dtype: "F32", shape: []int{2}, raw: f32raw(1, 2)},
	})
	f, err := parseSafetensors(blob)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	for _, tc := range []struct {
		name string
		want []byte
	}{
		{"blocks", []byte{1, 2, 3, 250, 251, 252}},
		{"flags", []byte{0, 1}},
		{"signed", []byte{0x80, 0x7f}},
	} {
		tn, err := f.Tensor(tc.name)
		if err != nil {
			t.Fatalf("Tensor(%s): %v", tc.name, err)
		}
		got, err := tn.Uint8s()
		if err != nil {
			t.Fatalf("Uint8s(%s): %v", tc.name, err)
		}
		if !bytes.Equal(got, tc.want) {
			t.Errorf("%s = %v, want %v", tc.name, got, tc.want)
		}
	}

	// Non-byte dtypes are refused rather than reinterpreted: handing back raw bytes for
	// an F32 would be a little-endian-dependent view, and those dtypes already have
	// typed accessors that decode properly.
	tn, err := f.Tensor("floats")
	if err != nil {
		t.Fatalf("Tensor(floats): %v", err)
	}
	if _, err := tn.Uint8s(); err == nil {
		t.Error("Uint8s on an F32 tensor succeeded; want an error")
	}

	// A header whose shape disagrees with its payload must be refused AT READ TIME.
	// Byte-wide dtypes are deliberately left unsized in dtypeSize (so opening an
	// unrelated file with an inconsistent, unread U8 tensor keeps working), which makes
	// this accessor the only place that check can happen.
	bad := buildSafetensors(map[string]stEntry{
		"blocks": {dtype: "U8", shape: []int{4, 4}, raw: []byte{1, 2, 3}}, // says 16, has 3
	})
	bf, err := parseSafetensors(bad)
	if err != nil {
		t.Fatalf("parse (mismatched U8 should still OPEN): %v", err)
	}
	bt, err := bf.Tensor("blocks")
	if err != nil {
		t.Fatalf("Tensor: %v", err)
	}
	if _, err := bt.Uint8s(); err == nil {
		t.Error("Uint8s accepted a shape/payload mismatch; want an error")
	}
}

// TestTensor_SubF32Transposed: SubF32Transposed(start, rows, cols) is SubF32(start, rows*cols) transposed, bit for bit,
// in every dtype, for shapes on and off the tile edge and for a matrix that starts inside the tensor; and a range
// outside the tensor is an error.
func TestTensor_SubF32Transposed(t *testing.T) {
	const lead, maxRows, maxCols = 7, 37, 50
	n := lead + maxRows*maxCols
	f32s, halves := make([]float32, n), make([]uint16, n)
	for i := range n {
		f32s[i] = float32(i%251-125) * 0.375    // exact in bf16's 8 significant bits
		halves[i] = uint16(i * 2654435761 >> 7) // every binary16 class, NaN and Inf included
	}
	halves[lead], halves[lead+1], halves[lead+2] = 0x0001, 0x7C00, 0x7E01 // subnormal, +Inf, NaN at known places
	blob := buildSafetensors(map[string]stEntry{
		"f32":  {"F32", []int{n}, f32raw(f32s...)},
		"bf16": {"BF16", []int{n}, bf16raw(f32s...)},
		"f16":  {"F16", []int{n}, f16raw(halves...)},
		"i32":  {"I32", []int{4}, i32raw(1, 2, 3, 4)},
	})
	sf, err := OpenSafetensorsFromFS(fstest.MapFS{"m.safetensors": &fstest.MapFile{Data: blob}}, "m.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	defer sf.Close()

	for _, name := range []string{"f32", "bf16", "f16"} {
		tn, err := sf.Tensor(name)
		if err != nil {
			t.Fatalf("Tensor(%s): %v", name, err)
		}
		for _, sh := range [][3]int{
			{lead, 37, 50}, {lead, 50, 37}, {0, 16, 16}, {3, 32, 48}, {0, 1, 50}, {lead, 37, 1}, {5, 17, 15}, {0, 0, 9}, {0, 9, 0},
		} {
			start, rows, cols := sh[0], sh[1], sh[2]
			flat, err := tn.SubF32(start, rows*cols)
			if err != nil {
				t.Fatalf("%s SubF32(%d, %d): %v", name, start, rows*cols, err)
			}
			got, err := tn.SubF32Transposed(start, rows, cols)
			if err != nil {
				t.Fatalf("%s SubF32Transposed(%d, %d, %d): %v", name, start, rows, cols, err)
			}
			if len(got) != rows*cols {
				t.Fatalf("%s [%d x %d]: %d elements, want %d", name, rows, cols, len(got), rows*cols)
			}
			for r := range rows {
				for c := range cols {
					if w, g := math.Float32bits(flat[r*cols+c]), math.Float32bits(got[c*rows+r]); w != g {
						t.Fatalf("%s [%d x %d] at %d: out[%d,%d] = %08x, want element (%d,%d) = %08x", name, rows, cols, start, c, r, g, r, c, w)
					}
				}
			}
		}
		if rows, cols := 4, 5; name == "f32" {
			got, err := tn.SubF32Transposed(0, rows, cols)
			if err != nil {
				t.Fatal(err)
			}
			got[0] = 12345 // a fresh slice: writing it must not reach the tensor
			if again, _ := tn.SubF32(0, 1); again[0] != f32s[0] {
				t.Errorf("SubF32Transposed aliased the tensor: element 0 reads %g after a write to the result", again[0])
			}
		}
		for _, bad := range [][3]int{{0, maxRows + 1, maxCols}, {lead + 1, maxRows, maxCols}, {-1, 2, 2}, {0, -1, 2}, {0, 2, -1}, {n + 1, 0, 0}, {0, 1 << 62, 4}} {
			if _, err := tn.SubF32Transposed(bad[0], bad[1], bad[2]); err == nil {
				t.Errorf("%s SubF32Transposed(%d, %d, %d): expected an out-of-bounds error", name, bad[0], bad[1], bad[2])
			}
		}
	}
	tn, err := sf.Tensor("i32")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tn.SubF32Transposed(0, 2, 2); err == nil {
		t.Error("an I32 tensor: expected an unsupported-dtype error")
	}
}
