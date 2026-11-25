// Package guardmem places test inputs against pages mapped with no access, so that an
// assembly kernel that reads one element past either end of an input faults instead of
// reading whatever the Go heap holds there. It is the harness of the over-read gate,
// and the approach is the standard library's own for its assembly
// (bytes/boundary_test.go, dangerousSlice).
//
// An Arena owns the mappings of one test. End places a slice whose last element ends
// where a guard page begins; Start places one whose first element begins where a guard
// page ends. Arena.Run calls a kernel with faults turned into panics
// (runtime/debug.SetPanicOnFault) and, on a fault, fails the test with the kernel's
// name, the input whose guard was touched, its length, and how far outside it the read
// landed.
//
// Supported reports whether the platform can map guard pages (linux and darwin); on any
// other platform New skips the test and says so.
package guardmem

import (
	"fmt"
	"runtime/debug"
	"testing"
	"unsafe"
)

// Arena holds the guarded mappings of one test; they are unmapped by t.Cleanup. An
// Arena is not safe for concurrent use.
type Arena struct {
	t      testing.TB
	inputs []*input
}

// input is one placed slice: its name and element count, its usable span [lo, hi), and
// the guard page that sits on its open side.
type input struct {
	name     string
	n        int
	elemSize uintptr
	elemType string
	lo, hi   uintptr
	guardLo  uintptr // first byte of the guard page
	guardHi  uintptr // one past its last byte
	atEnd    bool    // true: guard follows the slice; false: guard precedes it
}

// New returns an Arena for t, or skips t on a platform with no guard-page support.
func New(t testing.TB) *Arena {
	t.Helper()
	if !Supported {
		t.Skipf("guardmem: guard pages are mapped on linux and darwin only; this is %s", platform())
	}
	return &Arena{t: t}
}

// End returns n zeroed elements of T whose last element ends where a page with no
// access begins: reading x[n] (or any byte past it) faults.
func End[T any](a *Arena, name string, n int) []T {
	a.t.Helper()
	return place[T](a, name, n, true)
}

// Start returns n zeroed elements of T whose first element begins where a page with no
// access ends: reading x[-1] (or any byte before it) faults.
func Start[T any](a *Arena, name string, n int) []T {
	a.t.Helper()
	return place[T](a, name, n, false)
}

// EndCopy and StartCopy place a copy of src.
func EndCopy[T any](a *Arena, name string, src []T) []T {
	a.t.Helper()
	x := End[T](a, name, len(src))
	copy(x, src)
	return x
}

// StartCopy is EndCopy against the start guard.
func StartCopy[T any](a *Arena, name string, src []T) []T {
	a.t.Helper()
	x := Start[T](a, name, len(src))
	copy(x, src)
	return x
}

func place[T any](a *Arena, name string, n int, atEnd bool) []T {
	a.t.Helper()
	if n < 0 {
		a.t.Fatalf("guardmem: %s: negative length %d", name, n)
	}
	var zero T
	size := unsafe.Sizeof(zero)
	nbytes := uintptr(n) * size
	page := uintptr(pageSize())
	dataPages := (nbytes + page - 1) / page
	if dataPages == 0 {
		dataPages = 1 // a zero-length input still gets a real address beside its guard
	}
	// One guard page on each side; only the one on the slice's open side is checked.
	m, err := mapGuarded(int(dataPages * page))
	if err != nil {
		a.t.Fatalf("guardmem: %s: %v", name, err)
	}
	a.t.Cleanup(func() { unmap(m) })
	base := uintptr(unsafe.Pointer(unsafe.SliceData(m)))
	dataLo := base + page
	dataHi := dataLo + dataPages*page
	in := &input{name: name, n: n, elemSize: size, elemType: fmt.Sprintf("%T", zero), atEnd: atEnd}
	if atEnd {
		in.lo, in.hi = dataHi-nbytes, dataHi
		in.guardLo, in.guardHi = dataHi, dataHi+page
	} else {
		in.lo, in.hi = dataLo, dataLo+nbytes
		in.guardLo, in.guardHi = base, dataLo
	}
	a.inputs = append(a.inputs, in)
	off := in.lo - base
	return unsafe.Slice((*T)(unsafe.Pointer(&m[off])), n)
}

// Fault describes a fault Run caught.
type Fault struct {
	Kernel string
	Addr   uintptr
	Input  string // the input whose guard page holds Addr, or "" if none does
	Detail string // its length and where the read landed relative to it
}

func (f *Fault) Error() string {
	if f.Input == "" {
		return fmt.Sprintf("%s: fault at %#x, outside every guarded input", f.Kernel, f.Addr)
	}
	return fmt.Sprintf("%s: read outside input %s: %s (fault at %#x)", f.Kernel, f.Input, f.Detail, f.Addr)
}

// Run calls fn on a fresh goroutine with faults turned into panics and fails the test,
// naming kernel, if fn touched a guard page or faulted anywhere else. It returns the
// fault, or nil. A panic that is not a memory fault is re-raised on the test goroutine.
func (a *Arena) Run(kernel string, fn func()) *Fault {
	a.t.Helper()
	f, other := a.catch(kernel, fn)
	if other != nil {
		panic(other)
	}
	if f != nil {
		a.t.Error(f.Error())
	}
	return f
}

// Catch is Run without failing the test: it returns the fault for a test that expects one.
func (a *Arena) Catch(kernel string, fn func()) *Fault {
	a.t.Helper()
	f, other := a.catch(kernel, fn)
	if other != nil {
		panic(other)
	}
	return f
}

func (a *Arena) catch(kernel string, fn func()) (fault *Fault, other any) {
	done := make(chan struct{})
	go func() {
		defer close(done)
		defer func() {
			r := recover()
			if r == nil {
				return
			}
			ae, ok := r.(interface{ Addr() uintptr })
			if !ok {
				other = r
				return
			}
			fault = a.describe(kernel, ae.Addr())
		}()
		// SetPanicOnFault is per goroutine; this goroutine exists only for fn.
		debug.SetPanicOnFault(true)
		fn()
	}()
	<-done
	return fault, other
}

func (a *Arena) describe(kernel string, addr uintptr) *Fault {
	f := &Fault{Kernel: kernel, Addr: addr}
	for _, in := range a.inputs {
		if addr < in.guardLo || addr >= in.guardHi {
			continue
		}
		f.Input = in.name
		if in.atEnd {
			f.Detail = fmt.Sprintf("len %d %s, byte %d past its end", in.n, in.elemType, addr-in.hi)
		} else {
			f.Detail = fmt.Sprintf("len %d %s, byte %d before its start", in.n, in.elemType, in.lo-addr)
		}
		return f
	}
	return f
}
