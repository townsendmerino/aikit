package outputcontract

import (
	"math"
	"testing"
	"unsafe"
)

// TestOutputContract_harnessCanFail shows the gate goes red: a kernel that skips one element, one that accumulates into dst, one that writes past its region, and a scratch-dependent one are each caught,
// and the same shapes done right are not.
func TestOutputContract_harnessCanFail(t *testing.T) {
	const n = 37
	mk := func(name string, kernel func(dst, scratch []float32, src []float32)) Case {
		return Case{name, func() (func(), []*OutBuf) {
			src := make([]float32, n)
			for i := range src {
				src[i] = float32(i%7) - 3
			}
			dst, o := MkOut[float32]("dst", n)
			scr, so := MkOut[float32]("scratch", n)
			return func() { kernel(dst, scr, src) }, []*OutBuf{o, so.AsScratch()}
		}}
	}
	sound := func(dst, _, src []float32) {
		for i := range dst {
			dst[i] = src[i] * 2
		}
	}
	if v := Violations(mk("sound", sound)); len(v) != 0 {
		t.Fatalf("a sound kernel was flagged: %v", v)
	}
	for _, bad := range []struct {
		name string
		k    func(dst, scratch []float32, src []float32)
	}{
		{"skips-the-last-element", func(dst, _, src []float32) {
			for i := 0; i < len(dst)-1; i++ {
				dst[i] = src[i] * 2
			}
		}},
		{"accumulates", func(dst, _, src []float32) {
			for i := range dst {
				dst[i] += src[i] * 2
			}
		}},
		{"overruns", func(dst, _, src []float32) {
			for i := range dst {
				dst[i] = src[i] * 2
			}
			*(*float32)(unsafe.Add(unsafe.Pointer(&dst[0]), len(dst)*4)) = 1 // one past the region
		}},
		{"reads-scratch", func(dst, scr, src []float32) {
			for i := range dst {
				dst[i] = src[i]*2 + scr[0]*0 + float32(math.Float32bits(scr[0])&1)
			}
		}},
	} {
		if v := Violations(mk(bad.name, bad.k)); len(v) == 0 {
			t.Errorf("%s was NOT flagged: the gate cannot go red for it", bad.name)
		}
	}
}
