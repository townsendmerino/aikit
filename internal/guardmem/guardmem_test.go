package guardmem

import (
	"strings"
	"testing"
)

// sum reads x[0:n] with a plain Go loop through an index the compiler cannot bound, so
// n > len(x) reads past the slice instead of panicking on a bounds check.
//
//go:noinline
func sum(x []float32, n int) float32 {
	var s float32
	p := unsafeSlice(x, n)
	for _, v := range p {
		s += v
	}
	return s
}

func TestInBoundsReadsDoNotFault(t *testing.T) {
	a := New(t)
	for _, n := range []int{0, 1, 3, 1023, 1024, 1025, 5000} {
		e := End[float32](a, "e", n)
		s := Start[float32](a, "s", n)
		for i := range e {
			e[i], s[i] = 1, 1
		}
		if f := a.Run("sum", func() { sum(e, n); sum(s, n) }); f != nil {
			t.Fatalf("n=%d: in-bounds read faulted: %v", n, f)
		}
		if got := sum(e, n); got != float32(n) {
			t.Fatalf("n=%d: End slice sums to %v, want %v", n, got, n)
		}
	}
}

// Break-it-first: one element past the end, and one before the start, must fault and be
// named.
func TestReadPastEndIsNamed(t *testing.T) {
	a := New(t)
	for _, n := range []int{0, 1, 7, 1024, 5000} {
		x := End[float32](a, "x", n)
		f := a.Catch("sum", func() { sum(x, n+1) })
		if f == nil {
			t.Fatalf("n=%d: reading x[n] did not fault", n)
		}
		want := "sum: read outside input x: len " + itoa(n) + " float32, byte 0 past its end"
		if !strings.HasPrefix(f.Error(), want) {
			t.Fatalf("n=%d: fault = %q, want prefix %q", n, f.Error(), want)
		}
	}
}

func TestReadBeforeStartIsNamed(t *testing.T) {
	a := New(t)
	for _, n := range []int{1, 7, 1024} {
		x := Start[float32](a, "x", n)
		f := a.Catch("sumBack", func() { sum(unsafeBefore(x, 1), 1) })
		if f == nil {
			t.Fatalf("n=%d: reading x[-1] did not fault", n)
		}
		want := "sumBack: read outside input x: len " + itoa(n) + " float32, byte 4 before its start"
		if !strings.HasPrefix(f.Error(), want) {
			t.Fatalf("n=%d: fault = %q, want prefix %q", n, f.Error(), want)
		}
	}
}

func TestGuardsAreTight(t *testing.T) {
	a := New(t)
	x := End[uint8](a, "x", 3)
	if f := a.Catch("last", func() { _ = unsafeSlice(x, 3)[2] }); f != nil {
		t.Fatalf("reading the last byte faulted: %v", f)
	}
	y := Start[uint8](a, "y", 3)
	if f := a.Catch("first", func() { _ = y[0] }); f != nil {
		t.Fatalf("reading the first byte faulted: %v", f)
	}
}

func TestNonFaultPanicIsReRaised(t *testing.T) {
	a := New(t)
	defer func() {
		if r := recover(); r != "boom" {
			t.Fatalf("recovered %v, want the kernel's own panic", r)
		}
	}()
	a.Catch("panics", func() { panic("boom") })
	t.Fatal("Catch returned after a non-fault panic")
}

func TestCopyHelpers(t *testing.T) {
	a := New(t)
	src := []int8{1, -2, 3}
	for _, x := range [][]int8{EndCopy(a, "e", src), StartCopy(a, "s", src)} {
		if len(x) != 3 || x[0] != 1 || x[1] != -2 || x[2] != 3 {
			t.Fatalf("copy = %v, want %v", x, src)
		}
	}
}
