//go:build linux

package embed

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"unsafe"

	"github.com/townsendmerino/aikit/mmap"
)

// mappingRSSKiB is this process's resident kilobytes in the mapping that holds addr (/proc/self/smaps): the pages the
// process has mapped in, which MADV_DONTNEED drops, not the page cache's copy (mincore's view), which it keeps.
func mappingRSSKiB(t *testing.T, addr uintptr) int {
	t.Helper()
	f, err := os.Open("/proc/self/smaps")
	if err != nil {
		t.Skipf("no /proc/self/smaps: %v", err)
	}
	defer f.Close()
	in := false
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		var lo, hi uintptr
		if n, err := fmt.Sscanf(line, "%x-%x", &lo, &hi); err == nil && n == 2 && strings.Contains(line, " ") {
			in = addr >= lo && addr < hi
			continue
		}
		var kb int
		if in && strings.HasPrefix(line, "Rss:") {
			if _, err := fmt.Sscanf(strings.TrimPrefix(line, "Rss:"), "%d", &kb); err != nil {
				t.Fatalf("smaps %q: %v", line, err)
			}
			return kb
		}
	}
	t.Fatalf("no mapping holds %#x", addr)
	return 0
}

// TestGGUFReleaseTensors_dropsTheResidentPages: reading a mapped tensor makes its pages resident in this process, and
// ReleaseTensors drops them, which is what keeps a converting load from holding the whole source.
func TestGGUFReleaseTensors_dropsTheResidentPages(t *testing.T) {
	path, _ := releaseFixture(t)
	g, err := OpenGGUFMmap(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	addr := uintptr(unsafe.Pointer(&g.mmap[0]))
	if _, _, err := g.Tensor("big"); err != nil {
		t.Fatal(err)
	}
	before := mappingRSSKiB(t, addr)
	if before < 200 {
		t.Fatalf("reading the 256 KiB tensor left %d KiB of the mapping resident: the test cannot see a release", before)
	}
	if err := g.ReleaseTensors("big"); err != nil {
		t.Fatal(err)
	}
	if after := mappingRSSKiB(t, addr); after > 8 {
		t.Fatalf("%d KiB of the mapping still resident after ReleaseTensors (before: %d KiB)", after, before)
	}
}

// pageCacheResidentPages is how many pages of b the page cache holds (mincore): the file's cached copy, whether or not this
// process has them mapped.
func pageCacheResidentPages(t *testing.T, b []byte) int {
	t.Helper()
	ps := os.Getpagesize()
	vec := make([]byte, (len(b)+ps-1)/ps)
	if _, _, e := syscall.Syscall(syscall.SYS_MINCORE, uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)), uintptr(unsafe.Pointer(&vec[0]))); e != 0 {
		t.Fatalf("mincore: %v", e)
	}
	n := 0
	for _, v := range vec {
		n += int(v & 1)
	}
	return n
}

// TestGGUFReleaseTensors_dropsThePageCache: a released tensor's pages leave the page cache too, not only this process's
// mapping, so a load that converts a file larger than free memory does not push the kernel into reclaiming (and swapping)
// the converted copy while the source's cache fills. The tensor is 4 MiB: the page cache may hold a file in large folios,
// and a folio that also holds bytes still mapped (the header, the next tensor) stays, so only a tensor much larger than a
// folio shows the drop. The file is synced to a disk file system: its pages are clean, as a checkpoint's are, and tmpfs
// pages are shared memory, not file cache.
func TestGGUFReleaseTensors_dropsThePageCache(t *testing.T) {
	if runtime.GOARCH != "amd64" && runtime.GOARCH != "arm64" {
		t.Skipf("mmap.DropFileCache is a no-op on linux/%s", runtime.GOARCH)
	}
	const n = 1 << 20 // 4 MiB of f32
	big := make([]float32, n)
	for i := range big {
		big[i] = float32(i % 977)
	}
	path := filepath.Join(t.TempDir(), "release-cache.gguf")
	writeTestGGUF(t, path, []ggufKV{{"general.architecture", "llama"}}, []ggufF32{
		{"big", []uint64{1024, 1024}, big},
		{"small", []uint64{4}, []float32{1, 2, 3, 4}},
	})
	var st syscall.Statfs_t
	if err := syscall.Statfs(path, &st); err != nil || st.Type == 0x01021994 {
		t.Skipf("%s is on tmpfs (or unreadable: %v): its pages are not file cache", path, err)
	}
	f, err := os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Sync(); err != nil {
		t.Fatal(err)
	}
	f.Close()
	g, err := OpenGGUFMmap(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	if _, _, err := g.Tensor("big"); err != nil {
		t.Fatal(err)
	}
	raw, err := g.tensorBytes(g.tensors["big"], n)
	if err != nil {
		t.Fatal(err)
	}
	span := mmap.PageAlignedInterior(raw)
	pages := len(span) / os.Getpagesize()
	before := pageCacheResidentPages(t, span)
	if before*4 < pages*3 {
		t.Fatalf("%d of %d pages cached after reading the tensor: the test cannot see a release", before, pages)
	}
	if err := g.ReleaseTensors("big"); err != nil {
		t.Fatal(err)
	}
	if after := pageCacheResidentPages(t, span); after*4 > pages {
		t.Fatalf("%d of %d pages still in the page cache after ReleaseTensors (%d before)", after, pages, before)
	}
}
