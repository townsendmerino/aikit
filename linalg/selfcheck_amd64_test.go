//go:build amd64

package linalg

import (
	"slices"
	"testing"
)

// Repair on amd64: a kernel that is wrong only while AVX2 is on makes SelfCheck turn AVX2 (and Q4K's flag, derived from it) off, after which the kernels agree, and the report
// says what happened. The flags are restored afterwards.
func TestSelfCheck_repairStepsDownATier(t *testing.T) {
	if !hasAVX2 {
		t.Skip("AVX2 is already off here (a forced build or an old CPU): there is no tier to step down")
	}
	saved := [4]bool{hasAVX2, hasQ4KAVX2, hasAVX512VNNI, hasAVX512VNNIVL}
	t.Cleanup(func() {
		hasAVX2, hasQ4KAVX2, hasAVX512VNNI, hasAVX512VNNIVL = saved[0], saved[1], saved[2], saved[3]
		scDotI8 = dotI8
	})
	scDotI8 = func(a, b []int8) int32 {
		s := dotI8(a, b)
		if hasAVX2 && len(a) == 17 { // wrong only while the (pretend) bad AVX2 kernel is selected
			s += 3
		}
		return s
	}
	rep := SelfCheck(true)
	if rep.OK() {
		t.Fatal("the broken tier was not detected")
	}
	if !slices.Contains(rep.Disabled, "avx2") {
		t.Fatalf("expected avx2 to be disabled, got %v (first pass found %v)", rep.Disabled, rep.Mismatches)
	}
	if len(rep.Remaining) != 0 {
		t.Fatalf("after stepping down the kernels still disagree: %v", rep.Remaining)
	}
	if hasAVX2 || hasQ4KAVX2 {
		t.Errorf("AVX2 and its derived flag must be off after the repair: hasAVX2=%v hasQ4KAVX2=%v", hasAVX2, hasQ4KAVX2)
	}
	if k := ActiveKernels(); slices.Contains(k.Active, "avx2") {
		t.Errorf("ActiveKernels still reports avx2 active: %v", k.Active)
	}
}

// With VNNI on (a host that has it, or a forced build cannot: this one skips on hosts without), a bad VNNI tier is stepped off first and AVX2 is left alone.
func TestSelfCheck_repairTurnsVNNIOffBeforeAVX2(t *testing.T) {
	if !hasAVX512VNNI {
		t.Skip("no AVX-512 VNNI here: run on a VNNI host, or under SDE (-icx)")
	}
	saved := [4]bool{hasAVX2, hasQ4KAVX2, hasAVX512VNNI, hasAVX512VNNIVL}
	t.Cleanup(func() {
		hasAVX2, hasQ4KAVX2, hasAVX512VNNI, hasAVX512VNNIVL = saved[0], saved[1], saved[2], saved[3]
		scDotI8 = dotI8
	})
	scDotI8 = func(a, b []int8) int32 {
		s := dotI8(a, b)
		if hasAVX512VNNI && len(a) == 17 {
			s++
		}
		return s
	}
	rep := SelfCheck(true)
	if !slices.Equal(rep.Disabled, []string{"avx512vnni"}) || !hasAVX2 {
		t.Fatalf("want exactly avx512vnni disabled with AVX2 intact: disabled %v hasAVX2 %v", rep.Disabled, hasAVX2)
	}
}
