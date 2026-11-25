package half_test

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
	"testing/fstest"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/internal/half"
	"github.com/townsendmerino/aikit/linalg"
)

// TestCallSitesAgree_allHalves: the decodes linalg and embed expose agree with half.ToF32, bit for bit, on all 65,536
// binary16 patterns: linalg.F16ToF32 (the scalar), linalg.F16ToF32Slice (the F16C/FCVTL kernels where present; a NaN
// may come back quieted, so a NaN matches any NaN there), and embed's F16 safetensors read (Tensor.Float16sToF32),
// heap and FS, with NaNs bit for bit.
func TestCallSitesAgree_allHalves(t *testing.T) {
	const n = 1 << 16
	src := make([]uint16, n)
	raw := make([]byte, 2*n)
	for i := range n {
		src[i] = uint16(i)
		binary.LittleEndian.PutUint16(raw[2*i:], uint16(i))
	}
	want := make([]uint32, n)
	for i, h := range src {
		want[i] = math.Float32bits(half.ToF32(h))
	}
	isNaN := func(b uint32) bool { return b&0x7f800000 == 0x7f800000 && b&0x7fffff != 0 }
	check := func(site string, got []float32, nanAnyPayload bool) {
		t.Helper()
		bad := 0
		for i, g := range got {
			gb := math.Float32bits(g)
			if gb == want[i] || (nanAnyPayload && isNaN(gb) && isNaN(want[i])) {
				continue
			}
			if bad++; bad <= 3 {
				t.Errorf("%s(%#04x) = %#08x, half.ToF32 = %#08x", site, src[i], gb, want[i])
			}
		}
		if bad > 0 {
			t.Errorf("%s: %d of %d patterns differ", site, bad, n)
		}
	}

	scalar := make([]float32, n)
	for i, h := range src {
		scalar[i] = linalg.F16ToF32(h)
	}
	check("linalg.F16ToF32", scalar, false)

	slice := make([]float32, n)
	linalg.F16ToF32Slice(slice, src)
	check("linalg.F16ToF32Slice", slice, true)

	header := []byte(`{"h":{"dtype":"F16","shape":[65536],"data_offsets":[0,131072]}}`)
	file := binary.LittleEndian.AppendUint64(nil, uint64(len(header)))
	file = append(append(file, header...), raw...)
	path := filepath.Join(t.TempDir(), "h.safetensors")
	if err := os.WriteFile(path, file, 0o644); err != nil {
		t.Fatal(err)
	}
	for _, open := range []struct {
		name string
		f    func() (*embed.SafetensorsFile, error)
	}{
		{"embed (heap)", func() (*embed.SafetensorsFile, error) { return embed.OpenSafetensors(path) }},
		{"embed (FS)", func() (*embed.SafetensorsFile, error) {
			return embed.OpenSafetensorsFromFS(fstest.MapFS{"h.safetensors": {Data: file}}, "h.safetensors")
		}},
	} {
		sf, err := open.f()
		if err != nil {
			t.Fatalf("%s: %v", open.name, err)
		}
		tn, err := sf.Tensor("h")
		if err != nil {
			t.Fatalf("%s: %v", open.name, err)
		}
		got, err := tn.Float16sToF32()
		if err != nil {
			t.Fatalf("%s: %v", open.name, err)
		}
		check(open.name+" Tensor.Float16sToF32", got, false)
	}
}
