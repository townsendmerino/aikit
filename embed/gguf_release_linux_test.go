//go:build linux

package embed

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"testing"
	"unsafe"
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
