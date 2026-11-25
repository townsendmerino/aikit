// Package outputcontract is the output-contract gate's harness, shared by the tests of every linalg kernel package so each
// registers its own writers. For a kernel that writes into a caller's buffer, Violations runs it with that buffer filled
// three ways (0xFF bytes, which is NaN in a float; zeros; a finite 0x5A pattern) and requires
//
//   - the covered region holds no NaN afterwards (nothing the kernel covers was left as it found it),
//   - the three results are byte-identical (the result does not depend on what the buffer held, so a caller that zeroes
//     first is doing dead work, and one that does not is correct), and
//   - the Guard elements just past the region are untouched.
//
// A scratch buffer is poisoned and guarded too, but its contents afterwards are not compared. What this does not prove:
// that the covered region is the right region (the parity gates own that), or that an arch's assembly is correct (the
// arch-tagged gates own that), so a package runs it once per build configuration.
//
// The documentation half: every exported writer's doc comment carries an "Output contract:" line in one of ContractForm's
// forms; ParseDecls reads a directory's declarations from source, so it covers every arch-tagged file on any host.
package outputcontract

import (
	"bytes"
	"fmt"
	"math"
	"testing"
	"unsafe"
)

// Guard is the number of elements of guard after every output region.
const Guard = 16

// OutBuf is one buffer a kernel writes, as Violations sees it.
type OutBuf struct {
	Name  string
	cov   []byte // the region the kernel is documented to cover
	guard []byte // the bytes just past it
	f32   bool
	f64   bool
	// scratch marks a work buffer, not an output: it is poisoned and guarded, but its own contents afterwards are not
	// compared, since a kernel may use only a prefix of it.
	scratch bool
}

// AsScratch marks o as a work buffer and returns it.
func (o *OutBuf) AsScratch() *OutBuf { o.scratch = true; return o }

// MkOut returns an n-element slice with Guard elements of guard behind it, and its OutBuf.
func MkOut[T any](name string, n int) ([]T, *OutBuf) {
	back := make([]T, n+Guard)
	var z T
	sz := int(unsafe.Sizeof(z))
	all := unsafe.Slice((*byte)(unsafe.Pointer(&back[0])), len(back)*sz)
	ob := &OutBuf{Name: name, cov: all[:n*sz], guard: all[n*sz:]}
	switch any(z).(type) {
	case float32:
		ob.f32 = true
	case float64:
		ob.f64 = true
	}
	return back[:n:n], ob
}

// Case is one kernel call: Setup builds fresh inputs and outputs and returns the call and its output buffers.
type Case struct {
	Name  string
	Setup func() (call func(), outs []*OutBuf)
}

func (o *OutBuf) hasNaN() bool {
	switch {
	case o.f32:
		for i := 0; i+4 <= len(o.cov); i += 4 {
			if math.IsNaN(float64(math.Float32frombits(uint32(o.cov[i]) | uint32(o.cov[i+1])<<8 | uint32(o.cov[i+2])<<16 | uint32(o.cov[i+3])<<24))) {
				return true
			}
		}
	case o.f64:
		for i := 0; i+8 <= len(o.cov); i += 8 {
			var u uint64
			for b := range 8 {
				u |= uint64(o.cov[i+b]) << (8 * b)
			}
			if math.IsNaN(math.Float64frombits(u)) {
				return true
			}
		}
	}
	return false
}

// Violations runs c with its outputs filled three ways and returns every way it broke the contract (nil = sound).
func Violations(c Case) []string {
	var bad []string
	var results [][]byte
	for _, fill := range []byte{0xFF, 0x00, 0x5A} {
		call, outs := c.Setup()
		for _, o := range outs {
			for i := range o.cov {
				o.cov[i] = fill
			}
			for i := range o.guard {
				o.guard[i] = 0xA7
			}
		}
		call()
		var snap []byte
		for _, o := range outs {
			for i, b := range o.guard {
				if b != 0xA7 {
					bad = append(bad, fmt.Sprintf("%s: wrote past the end of %s (guard byte %d is %#x)", c.Name, o.Name, i, b))
					break
				}
			}
			if o.scratch {
				continue
			}
			if fill == 0xFF && o.hasNaN() {
				bad = append(bad, fmt.Sprintf("%s: a NaN survives in %s after the call with it poisoned: the kernel leaves part of what it covers as it found it", c.Name, o.Name))
			}
			snap = append(snap, o.cov...)
		}
		results = append(results, snap)
	}
	if !bytes.Equal(results[0], results[1]) || !bytes.Equal(results[1], results[2]) {
		bad = append(bad, fmt.Sprintf("%s: the result depends on what the output buffers held before the call (0xFF vs 0x00 vs 0x5A fill differ)", c.Name))
	}
	return bad
}

// Run reports each of c's violations as a test error.
func Run(t testing.TB, c Case) {
	t.Helper()
	for _, m := range Violations(c) {
		t.Error(m)
	}
}
