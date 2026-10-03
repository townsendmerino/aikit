package embed

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ggufKV is one metadata entry for writeTestGGUF: a string, a uint16 or a uint32.
type ggufKV struct {
	key string
	val any
}

// ggufF32 is one F32 tensor for writeTestGGUF.
type ggufF32 struct {
	name string
	dims []uint64
	vals []float32
}

// writeTestGGUF writes a GGUF v3 file of F32 tensors, laid out as llama.cpp does: header, metadata, tensor directory,
// then the data section at the next 32-byte boundary with each tensor padded to 32.
func writeTestGGUF(t *testing.T, path string, kvs []ggufKV, tensors []ggufF32) {
	t.Helper()
	var b bytes.Buffer
	le := func(v any) { _ = binary.Write(&b, binary.LittleEndian, v) }
	str := func(s string) { le(uint64(len(s))); b.WriteString(s) }
	le(uint32(ggufMagic))
	le(uint32(3))
	le(uint64(len(tensors)))
	le(uint64(len(kvs)))
	for _, kv := range kvs {
		str(kv.key)
		switch v := kv.val.(type) {
		case string:
			le(uint32(8))
			str(v)
		case uint16:
			le(uint32(2))
			le(v)
		case uint32:
			le(uint32(4))
			le(v)
		default:
			t.Fatalf("writeTestGGUF: unsupported value %T", v)
		}
	}
	var off uint64
	for _, tn := range tensors {
		str(tn.name)
		le(uint32(len(tn.dims)))
		for _, d := range tn.dims {
			le(d)
		}
		le(uint32(ggmlTypeF32))
		le(off)
		off += uint64(len(tn.vals) * 4)
		off = (off + 31) &^ 31
	}
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	for _, tn := range tensors {
		le(tn.vals)
		for b.Len()%32 != 0 {
			b.WriteByte(0)
		}
	}
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

// splitFixture is a small model as one file and as three shards the way llama-gguf-split writes them: the first shard
// carries the model's metadata plus split.no / split.count / split.tensors.count, the others only split.no and
// split.count, and each holds a contiguous run of the tensors.
func splitFixture(t *testing.T) (single string, shards []string, tensors []ggufF32) {
	t.Helper()
	dir := t.TempDir()
	for i := range 5 {
		n := 3 + i*7 // odd sizes, so the 32-byte padding between tensors is exercised
		v := make([]float32, n*2)
		for j := range v {
			v[j] = float32(i*1000+j) * 0.25
		}
		tensors = append(tensors, ggufF32{name: fmt.Sprintf("blk.%d.w", i), dims: []uint64{uint64(n), 2}, vals: v})
	}
	model := []ggufKV{{"general.architecture", "llama"}, {"llama.block_count", uint32(5)}}
	single = filepath.Join(dir, "m.gguf")
	writeTestGGUF(t, single, model, tensors)
	parts := [][]ggufF32{tensors[:2], tensors[2:4], tensors[4:]}
	for i, p := range parts {
		kvs := []ggufKV{{"split.no", uint16(i)}, {"split.count", uint16(len(parts))}}
		if i == 0 {
			kvs = append(append([]ggufKV{}, model...), append(kvs, ggufKV{"split.tensors.count", uint32(len(tensors))})...)
		}
		path := filepath.Join(dir, fmt.Sprintf("m-%05d-of-%05d.gguf", i+1, len(parts)))
		writeTestGGUF(t, path, kvs, p)
		shards = append(shards, path)
	}
	return single, shards, tensors
}

// TestOpenGGUFSplitMmap_matchesSingleFile: the shard set reads as the model it was split from. Every tensor's dims
// and values, through Tensor and the streaming RowDequantizer, are bit-identical to the single file's, and the
// model's metadata comes from the first shard.
func TestOpenGGUFSplitMmap_matchesSingleFile(t *testing.T) {
	single, shards, tensors := splitFixture(t)
	one, err := OpenGGUFMmap(single)
	if err != nil {
		t.Fatal(err)
	}
	defer one.Close()
	set, err := OpenGGUFSplitMmap(shards)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	if a, _ := set.Str("general.architecture"); a != "llama" {
		t.Errorf("architecture = %q, want the first shard's llama", a)
	}
	if n, _ := set.Uint("llama.block_count"); n != 5 {
		t.Errorf("block_count = %d, want 5", n)
	}
	if len(set.Names()) != len(tensors) {
		t.Fatalf("set has %d tensors, want %d", len(set.Names()), len(tensors))
	}
	for _, tn := range tensors {
		wd, want, err := one.Tensor(tn.name)
		if err != nil {
			t.Fatal(err)
		}
		gd, got, err := set.Tensor(tn.name)
		if err != nil {
			t.Fatalf("%s from the set: %v", tn.name, err)
		}
		if fmt.Sprint(gd) != fmt.Sprint(wd) || len(got) != len(want) {
			t.Fatalf("%s: dims %v len %d, want %v len %d", tn.name, gd, len(got), wd, len(want))
		}
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) || want[i] != tn.vals[i] {
				t.Fatalf("%s[%d] = %v, single file %v, written %v", tn.name, i, got[i], want[i], tn.vals[i])
			}
		}
		_, into, err := set.RowDequantizer(tn.name)
		if err != nil {
			t.Fatal(err)
		}
		row := make([]float32, 2)
		if err := into(len(want)-2, row); err != nil || row[1] != want[len(want)-1] {
			t.Fatalf("%s last row through RowDequantizer = %v, %v; want %v", tn.name, row, err, want[len(want)-2:])
		}
	}
}

// TestOpenGGUFSplitMmap_refusesABadSet: each way a set can be wrong is refused at open, before any weight is read: a
// missing shard, shards out of order, a tensor in two shards, and a set whose tensors fall short of the first shard's
// split.tensors.count.
func TestOpenGGUFSplitMmap_refusesABadSet(t *testing.T) {
	_, shards, tensors := splitFixture(t)
	dir := filepath.Dir(shards[0])
	dup := filepath.Join(dir, "dup-00002-of-00003.gguf")
	writeTestGGUF(t, dup, []ggufKV{{"split.no", uint16(1)}, {"split.count", uint16(3)}}, tensors[1:4]) // blk.1 again
	short := filepath.Join(dir, "short-00003-of-00003.gguf")
	writeTestGGUF(t, short, []ggufKV{{"split.no", uint16(2)}, {"split.count", uint16(3)}}, nil)
	for _, c := range []struct {
		name  string
		paths []string
		want  string
	}{
		{"a missing shard", shards[:2], "not shard 1 of 2"},
		{"out of order", []string{shards[0], shards[2], shards[1]}, "not shard 2 of 3"},
		{"a tensor in two shards", []string{shards[0], dup, shards[2]}, "more than one shard"},
		{"fewer tensors than the first shard records", []string{shards[0], shards[1], short}, "split.tensors.count says 5"},
		{"no shards", nil, "at least one shard"},
	} {
		t.Run(c.name, func(t *testing.T) {
			g, err := OpenGGUFSplitMmap(c.paths)
			if err == nil {
				g.Close()
				t.Fatalf("opened %v; want a refusal containing %q", c.paths, c.want)
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Fatalf("error %q, want it to contain %q", err, c.want)
			}
		})
	}
}

// TestOpenGGUFSplitMmap_readAfterCloseFails: Close unmaps every shard, and a tensor read afterwards is an error, the
// way it is for a single file, not a read of an unmapped page.
func TestOpenGGUFSplitMmap_readAfterCloseFails(t *testing.T) {
	_, shards, tensors := splitFixture(t)
	set, err := OpenGGUFSplitMmap(shards)
	if err != nil {
		t.Fatal(err)
	}
	if err := set.Close(); err != nil {
		t.Fatal(err)
	}
	if _, _, err := set.Tensor(tensors[3].name); err == nil {
		t.Fatal("Tensor after Close succeeded; want an error")
	}
	if err := set.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}
