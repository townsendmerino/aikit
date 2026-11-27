package embed

import (
	"fmt"
	"os"
	"unsafe"

	"github.com/townsendmerino/aikit/mmap"
)

// ReleaseTensors drops the resident pages backing the named tensors of a mapped file (OpenGGUFMmap, OpenGGUFSplitMmap)
// while leaving the file open and every other tensor untouched, as SafetensorsFile.ReleaseTensors does: a loader that
// converts each tensor into something smaller and never reads the source bytes again calls it, so the mapping does not
// hold the whole checkpoint resident beside what was built from it.
//
// It is advisory and safe for a caller still holding an alias into a released tensor: the mapping is read-only and
// file-backed, so a released page re-faults identical bytes. A no-op on a heap-backed file (OpenGGUF, OpenGGUFBytes), and
// a footprint win on Linux only (MADV_DONTNEED on the tensor's page-aligned interior, then POSIX_FADV_DONTNEED on the same
// bytes of the file, so a converting load neither holds the source mapped nor fills the page cache with it and pushes the
// kernel to reclaim the converted copy; see SafetensorsFile.ReleaseTensors for darwin). An unknown name is an error; madvise failures are swallowed as advisory.
func (g *GGUFFile) ReleaseTensors(names ...string) error {
	var firstErr error
	for _, name := range names {
		info, ok := g.tensors[name]
		if !ok {
			if firstErr == nil {
				firstErr = fmt.Errorf("gguf: ReleaseTensors: tensor %q not found", name)
			}
			continue
		}
		if g.mmap == nil && g.mmaps == nil {
			continue // heap-backed, or already Closed: nothing resident to drop
		}
		n := 1
		for _, d := range info.dims {
			n *= int(d)
		}
		raw, err := g.tensorBytes(info, n)
		if err != nil {
			continue // a tensor whose bytes cannot be sized holds nothing a release could name
		}
		span := mmap.PageAlignedInterior(raw)
		_ = mmap.Advise(span, false) //nolint:errcheck // madvise DONTNEED is an advisory hint on release
		if f, off := g.fileOf(span); f != nil {
			_ = mmap.DropFileCache(f, off, int64(len(span))) //nolint:errcheck // advisory, like the madvise above
		}
	}
	return firstErr
}

// TensorBytes is the named tensor's size in the file, in bytes (its quantized blocks as stored), and whether the file has it
// in a type whose size is known. A loader that releases each tensor once converted holds at most a few tensors' worth of the
// source at a time, and prices that from this.
func (g *GGUFFile) TensorBytes(name string) (int, bool) {
	info, ok := g.tensors[name]
	if !ok {
		return 0, false
	}
	n := 1
	for _, d := range info.dims {
		n *= int(d)
	}
	raw, err := g.tensorBytes(info, n)
	if err != nil {
		return 0, false
	}
	return len(raw), true
}

// fileOf is the mapped file holding span (a slice of one of g's mappings) and span's offset in it, or nil.
func (g *GGUFFile) fileOf(span []byte) (*os.File, int64) {
	if len(span) == 0 {
		return nil, 0
	}
	maps := g.mmaps
	if g.mmap != nil {
		maps = [][]byte{g.mmap}
	}
	p := uintptr(unsafe.Pointer(&span[0]))
	for i, m := range maps {
		if len(m) == 0 || i >= len(g.files) {
			continue
		}
		base := uintptr(unsafe.Pointer(&m[0]))
		if p >= base && p+uintptr(len(span)) <= base+uintptr(len(m)) {
			return g.files[i], int64(p - base)
		}
	}
	return nil, 0
}
