package linalg

import (
	"fmt"
	"runtime"
	"strings"
	"testing"
)

// refusal returns the text of fn's panic, failing t if fn returned or if the panic was a
// runtime error (a crash, not a refusal of its arguments).
func refusal(t *testing.T, name string, fn func()) string {
	t.Helper()
	var r any
	func() {
		defer func() { r = recover() }()
		fn()
	}()
	if r == nil {
		t.Errorf("%s: returned; want it to refuse", name)
		return ""
	}
	if _, ok := r.(runtime.Error); ok {
		t.Errorf("%s: crashed with a runtime error: %v", name, r)
		return ""
	}
	return fmt.Sprint(r)
}

// A group size of 0 (or less) is refused by name with the group in the message, as
// QuantizeGroupsInt4 does, rather than dividing by zero.
func TestGroupZero_refusedByName(t *testing.T) {
	const K = 8
	a, aq := make([]float32, K), make([]int8, K)
	for _, group := range []int{0, -1} {
		cases := map[string]func(){
			"DequantizeRowInt4":              func() { DequantizeRowInt4(make([]byte, K/2), make([]float32, K), group, K, make([]float32, K)) },
			"QuantizeGroupInt4Row":           func() { QuantizeGroupInt4Row(a, K, group, make([]byte, K/2), make([]float32, K)) },
			"QuantizeActivationsGroupedInto": func() { QuantizeActivationsGroupedInto(aq, make([]float32, K), a, 1, K, group) },
			"SumActGroupsInto":               func() { SumActGroupsInto(make([]int32, K), aq, 1, K, group) },
		}
		for name, fn := range cases {
			msg := refusal(t, fmt.Sprintf("%s(group=%d)", name, group), fn)
			if want := fmt.Sprintf("group must be > 0, got %d", group); msg != "" && !strings.Contains(msg, want) {
				t.Errorf("%s(group=%d): refused with %q, want it to name the group (%q)", name, group, msg, want)
			}
		}
	}
}

// An empty source converts to nothing: F32ToF16Slice and QuantizeF16 accept zero elements.
func TestF16_emptySource(t *testing.T) {
	F32ToF16Slice(nil, nil)
	F32ToF16Slice(make([]uint16, 3), []float32{})
	for _, s := range [][2]int{{0, 0}, {0, 4}, {1, 0}} {
		w := QuantizeF16(nil, s[0], s[1])
		if w.Rows() != s[0] || w.Cols() != s[1] {
			t.Errorf("QuantizeF16(nil, %d, %d): shape %dx%d", s[0], s[1], w.Rows(), w.Cols())
		}
	}
	// A short dst is still refused.
	if msg := func() (r any) {
		defer func() { r = recover() }()
		F32ToF16Slice(make([]uint16, 1), []float32{1, 2})
		return nil
	}(); msg == nil {
		t.Error("F32ToF16Slice with a short dst returned")
	}
}
