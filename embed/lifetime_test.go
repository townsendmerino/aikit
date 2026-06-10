package embed

import "testing"

// TestSafetensors_tensorAfterClose pins the §3.3 guard: requesting a tensor after
// Close returns a clean error rather than handing out an alias into a region the
// mmap path may have unmapped. (The held-slice-after-close trap can't be caught at
// the API and is covered by the Tensor doc's WRONG/RIGHT example.)
func TestSafetensors_tensorAfterClose(t *testing.T) {
	blob := minimalSafetensors(`{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`, make([]byte, 8))
	sf, err := parseSafetensors(blob)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sf.Tensor("w"); err != nil {
		t.Fatalf("Tensor before Close errored: %v", err)
	}
	if err := sf.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := sf.Tensor("w"); err == nil {
		t.Error("Tensor after Close returned no error — use-after-close guard missing")
	}
	if err := sf.Close(); err != nil { // idempotent
		t.Errorf("second Close errored: %v", err)
	}
}
