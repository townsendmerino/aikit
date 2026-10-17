package embed

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"weak"
)

// The finalizeMmaps comment states that an mmap'd file holding a tensor is never finalized (Tensor.owner makes it
// a finalizer cycle), so dropping it leaks the mapping rather than unmapping it under its views. This pins that
// statement: if it stops holding — the cycle is broken, or the finalizer becomes a cleanup — this fails, and the
// comments and every view-holding caller have to be revisited (docs/internal/task-memory-campaign.md, item 5).
func TestSafetensorsMmap_droppedFileWithTensorsIsNotFinalized(t *testing.T) {
	p := filepath.Join(t.TempDir(), "m.safetensors")
	if err := os.WriteFile(p, minimalSafetensors(`{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`, make([]byte, 8)), 0o644); err != nil {
		t.Fatal(err)
	}
	var w weak.Pointer[SafetensorsFile]
	func() {
		sf, err := OpenSafetensorsMmap(p)
		if err != nil {
			t.Fatal(err)
		}
		w = weak.Make(sf)
	}()
	for range 10 {
		runtime.GC()
	}
	if w.Value() == nil {
		t.Fatal("a dropped mmap'd SafetensorsFile with a tensor was collected: its finalizer can now unmap the mapping under live views. Revisit finalizeMmaps' comment and every holder of its views before shipping this")
	}
}
