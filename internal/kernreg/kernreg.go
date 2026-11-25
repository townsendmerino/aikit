// Package kernreg is the registry of a package's assembly entry points: one Row per
// TEXT symbol in its .s files, and on the host architecture one Adapter per Row that
// takes slices. It serves two tests from one list: the over-read sweep (Sweep), which
// places every input of every kernel against a guard page at each tail and block
// boundary and holds the result equal to the call on heap slices, and the inventory
// (Complete), which fails when a TEXT symbol has no row, a row names a symbol, a
// dispatcher or an oracle test that does not exist, or a host-architecture row that
// takes slices has no adapter.
//
// Rows and adapters are plain values a package's tests declare beside each kernel's
// test file and gather into one list; nothing registers itself.
package kernreg

import (
	"bytes"
	"fmt"
	"math"
	"reflect"
	"runtime"
	"slices"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/internal/guardmem"
)

// Row is one assembly entry point, on every architecture: what the inventory knows of it.
type Row struct {
	Symbol   string // the TEXT symbol, which is also its Go stub's name
	Arch     string // "amd64" or "arm64", the .s file's suffix
	Dispatch string // a non-test function of the package that calls Symbol
	Oracle   string // a Test function that holds Symbol, or Dispatch, equal to its reference
	// NoSlices, when set, says why the entry point has no adapter (it takes no slice:
	// a CPU probe, a peak probe). A row with it set is not swept.
	NoSlices string
}

// Adapter calls one entry point on the host architecture.
type Adapter struct {
	Symbol string
	// Needs reports whether this CPU can run the kernel; nil means always. A kernel
	// whose Needs is false is skipped, and the skip names it.
	Needs func() bool
	// Lengths are the sizes swept: each reaches a tail or a block boundary of the
	// kernel (one under, at, and one over each block size), and 0 where the kernel
	// accepts it.
	Lengths []int
	// Call builds the kernel's inputs for size n, the same values on every call, places
	// each one (and each output it reads or writes) with Put, calls the kernel, and
	// returns what it wrote. The result is compared bit for bit.
	Call func(p *Placer, n int) any
}

// side is where a Placer puts inputs.
type side int

const (
	onHeap side = iota
	atEnd
	atStart
)

func (s side) String() string {
	switch s {
	case atEnd:
		return "end guard"
	case atStart:
		return "start guard"
	}
	return "heap"
}

// Placer puts an adapter's inputs on the heap or against a guard page.
type Placer struct {
	a    *guardmem.Arena
	side side
}

// Put returns a copy of src placed where p places inputs; name is the input's name in a
// fault report.
func Put[T any](p *Placer, name string, src []T) []T {
	switch p.side {
	case atEnd:
		return guardmem.EndCopy(p.a, name, src)
	case atStart:
		return guardmem.StartCopy(p.a, name, src)
	}
	return slices.Clone(src)
}

// Deadline bounds one kernel call in Sweep: a loop that does not end fails as that
// kernel and size instead of hanging the test binary until its own timeout.
const Deadline = 20 * time.Second

// Sweep runs every adapter: for each length, the call on heap slices, then with every
// input against the end guard, then against the start guard, each result equal to the
// heap call's.
func Sweep(t *testing.T, adapters []Adapter) {
	for _, ad := range adapters {
		t.Run(ad.Symbol, func(t *testing.T) {
			if ad.Needs != nil && !ad.Needs() {
				t.Skipf("%s: this CPU (or a forced-fallback tag) does not provide the kernel's instructions", ad.Symbol)
			}
			if len(ad.Lengths) == 0 {
				t.Fatalf("%s: no lengths to sweep", ad.Symbol)
			}
			a := guardmem.New(t)
			for _, n := range ad.Lengths {
				want := call(t, a, ad, onHeap, n)
				for _, s := range []side{atEnd, atStart} {
					got := call(t, a, ad, s, n)
					if !sameBits(got, want) {
						t.Errorf("%s n=%d, inputs at the %s: result differs from the heap call\n got %v\nwant %v", ad.Symbol, n, s, got, want)
					}
				}
			}
		})
	}
}

func call(t *testing.T, a *guardmem.Arena, ad Adapter, s side, n int) any {
	t.Helper()
	p := &Placer{a: a, side: s}
	var out any
	label := fmt.Sprintf("%s n=%d (inputs at the %s)", ad.Symbol, n, s)
	var f *guardmem.Fault
	if r := panicOf(func() { f = a.RunWithin(label, Deadline, func() { out = ad.Call(p, n) }) }); r != nil {
		t.Fatalf("%s: panicked: %v", label, r)
	}
	if f != nil {
		t.FailNow()
	}
	return out
}

func panicOf(fn func()) (r any) {
	defer func() { r = recover() }()
	fn()
	return nil
}

// Refused calls fn and returns "" if it returned, or the text of the panic if it refused
// its arguments with one: an explicit panic, not a runtime error. A runtime error (an
// index out of range, a nil dereference, a division by zero) is a crash, not a refusal,
// and is re-raised, so the sweep fails naming the call.
func Refused(fn func()) (msg string) {
	defer func() {
		r := recover()
		if r == nil {
			return
		}
		if _, ok := r.(runtime.Error); ok {
			panic(r)
		}
		msg = fmt.Sprint("refused: ", r)
	}()
	fn()
	return ""
}

// sameBits is reflect.DeepEqual with floats compared by their bits, so a NaN equals the
// same NaN and -0 differs from +0.
func sameBits(a, b any) bool {
	var ba, bb bytes.Buffer
	bitsOf(&ba, reflect.ValueOf(a))
	bitsOf(&bb, reflect.ValueOf(b))
	return bytes.Equal(ba.Bytes(), bb.Bytes())
}

func bitsOf(w *bytes.Buffer, v reflect.Value) {
	switch v.Kind() {
	case reflect.Invalid:
		w.WriteString("<nil>")
	case reflect.Float32:
		fmt.Fprintf(w, "f%08x,", math.Float32bits(float32(v.Float())))
	case reflect.Float64:
		fmt.Fprintf(w, "d%016x,", math.Float64bits(v.Float()))
	case reflect.Slice, reflect.Array:
		fmt.Fprintf(w, "[%d:", v.Len())
		for i := range v.Len() {
			bitsOf(w, v.Index(i))
		}
		w.WriteString("]")
	case reflect.Struct:
		w.WriteString("{")
		for i := range v.NumField() {
			bitsOf(w, v.Field(i))
		}
		w.WriteString("}")
	case reflect.Interface, reflect.Pointer:
		if v.IsNil() {
			w.WriteString("<nil>")
			return
		}
		bitsOf(w, v.Elem())
	default:
		fmt.Fprintf(w, "%v,", v.Interface())
	}
}

// HostArch is the architecture whose adapters this binary can run.
const HostArch = runtime.GOARCH
