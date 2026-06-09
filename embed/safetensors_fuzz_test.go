package embed

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// minimalSafetensors builds a valid safetensors blob: 8-byte little-endian
// header length + JSON header. With one F32 tensor whose 8-byte payload follows.
func minimalSafetensors(header string, payload []byte) []byte {
	var b bytes.Buffer
	binary.Write(&b, binary.LittleEndian, uint64(len(header)))
	b.WriteString(header)
	b.Write(payload)
	return b.Bytes()
}

// FuzzParseSafetensors asserts the safetensors header parser never panics. The
// header length is an untrusted uint64 used in slice bounds, and tensor offsets
// come from untrusted JSON — both must be validated, not trusted.
func FuzzParseSafetensors(f *testing.F) {
	f.Add(minimalSafetensors(`{}`, nil))
	f.Add(minimalSafetensors(`{"__metadata__":{"k":"v"}}`, nil))
	f.Add(minimalSafetensors(`{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`, make([]byte, 8)))
	f.Add(minimalSafetensors(`{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,9999]}}`, make([]byte, 8))) // bad offset
	f.Add([]byte{0, 0, 0, 0, 0, 0, 0, 0})                                                                   // headerLen 0
	f.Add([]byte("short"))                                                                                  // < 8 bytes
	f.Add([]byte{})

	f.Fuzz(func(t *testing.T, data []byte) {
		ff, err := parseSafetensors(data)
		if err != nil {
			return
		}
		for _, n := range ff.Names() {
			if _, err := ff.Tensor(n); err != nil {
				t.Fatalf("Names() returned %q but Tensor(%q) errored: %v", n, n, err)
			}
		}
	})
}

// FuzzParseShardIndex fuzzes the JSON shard-index parser (model.safetensors.
// index.json). json.Unmarshal is panic-safe, so this mainly guards the
// post-unmarshal dedup/sort logic.
func FuzzParseShardIndex(f *testing.F) {
	f.Add([]byte(`{"weight_map":{"a.weight":"model-00001.safetensors"}}`))
	f.Add([]byte(`{"weight_map":{"a":"f1","b":"f1","c":"f2"}}`)) // dedup path
	f.Add([]byte(`{"weight_map":{}}`))
	f.Add([]byte(`{}`))
	f.Add([]byte(`not json`))
	f.Add([]byte{})

	f.Fuzz(func(t *testing.T, data []byte) {
		files, weightMap, err := parseShardIndex(data)
		if err != nil {
			return
		}
		// Every file referenced by the weight map must appear in files, exactly once.
		seen := map[string]int{}
		for _, fn := range files {
			seen[fn]++
		}
		for _, fn := range weightMap {
			if seen[fn] == 0 {
				t.Fatalf("weight_map references %q but it is missing from files %v", fn, files)
			}
		}
	})
}
