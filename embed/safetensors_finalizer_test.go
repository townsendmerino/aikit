package embed

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"testing"
	"time"
	"weak"
)

// Close is the only release of an mmap'd file (see SafetensorsFile.Close): dropping the file must never unmap the
// region under a view a caller still reads. This pins that. A dropped file's F32 view is read after the file is gone
// and ten GCs have run; with SetPanicOnFault a read of unmapped memory becomes a panic this test reports, rather than
// a SIGSEGV that kills the binary. If an automatic release (a finalizer, runtime.AddCleanup) is ever added, this
// fails, and every view-holding caller needs the runtime.KeepAlive discipline item 5 declined
// (docs/internal/task-memory-campaign.md).
func TestSafetensorsMmap_droppedFileKeepsItsMapping(t *testing.T) {
	p := filepath.Join(t.TempDir(), "m.safetensors")
	payload := make([]byte, 1<<16) // whole pages, so an unmap would really fault
	for i := range payload {
		payload[i] = byte(i)
	}
	header := `{"w":{"dtype":"F32","shape":[16384],"data_offsets":[0,65536]}}`
	for (8+len(header))%8 != 0 {
		header += " " // align the payload so TensorF32 returns a view, not a copy
	}
	if err := os.WriteFile(p, minimalSafetensors(header, payload), 0o644); err != nil {
		t.Fatal(err)
	}
	view, file := openAndDropForTest(t, p)
	// Wait until the file is actually collected (it holds no back-pointer, so it is), then give a finalizer or cleanup
	// — were one attached — time to run on its goroutine.
	deadline := time.Now().Add(5 * time.Second)
	for file.Value() != nil {
		if time.Now().After(deadline) {
			t.Fatal("the dropped SafetensorsFile was never collected; the test cannot check what its collection does")
		}
		runtime.GC()
		time.Sleep(5 * time.Millisecond)
	}
	for range 10 {
		runtime.GC()
		time.Sleep(10 * time.Millisecond)
	}
	defer debug.SetPanicOnFault(debug.SetPanicOnFault(true))
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("reading a dropped file's view faulted (%v): something unmapped it without Close", r)
		}
	}()
	// Read every value and compare it with what was written: an unmapped region either faults or, once the runtime has
	// mapped something else at that address, reads back other bytes — both mean it was released without Close.
	for i, x := range view {
		if want := math.Float32frombits(binary.LittleEndian.Uint32(payload[4*i:])); math.Float32bits(x) != math.Float32bits(want) {
			t.Fatalf("value %d of a dropped file's view reads %#x, wrote %#x: the mapping was released without Close", i, math.Float32bits(x), math.Float32bits(want))
		}
	}
}

// openAndDropForTest opens p, returns a view of its tensor and a weak pointer to the file, and drops the file. Not
// inlined, so no stack slot of the caller keeps the file alive.
//
//go:noinline
func openAndDropForTest(t *testing.T, p string) ([]float32, weak.Pointer[SafetensorsFile]) {
	sf, err := OpenSafetensorsMmap(p)
	if err != nil {
		t.Fatal(err)
	}
	v, err := sf.TensorF32("w")
	if err != nil {
		t.Fatal(err)
	}
	again, err := sf.TensorF32("w")
	if err != nil {
		t.Fatal(err)
	}
	if &again[0] != &v[0] {
		t.Fatal("TensorF32 copied the payload; the test needs a view into the mapping")
	}
	return v, weak.Make(sf)
}
