package ann

import (
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestScoreW8A8_allocatesNothing: scoring through a WeightMat costs the query no allocation once
// the Workspace is warm. The WeightMat lives in the caller's scratch; a local one escapes through
// the pointer method and allocated on every query.
func TestScoreW8A8_allocatesNothing(t *testing.T) {
	const dim, n = 64, 32
	codes, scales := linalg.QuantizeRowsInt8(make([]float32, n*dim), n, dim)
	q, dst := make([]float32, dim), make([]float32, n)
	var wm linalg.WeightMat
	var ws linalg.Workspace
	ws.SetThreshold(1 << 40) // serial: a fan-out's goroutines allocate by design
	scoreW8A8(&wm, &ws, q, codes, scales, dst, 1, dim, n)
	if a := testing.AllocsPerRun(50, func() { scoreW8A8(&wm, &ws, q, codes, scales, dst, 1, dim, n) }); a != 0 {
		t.Errorf("scoreW8A8: %v allocations a call, want 0", a)
	}
	if wm.Rows() != 0 {
		t.Error("scoreW8A8 left the scratch WeightMat holding the rows")
	}
}
