package linalg

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func (p *spanPool) idleForTest() (workers, parked int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.workers, p.parked
}

// checkOnce runs parallelSpawnCols over N columns at the given width and reports an error unless
// every column was visited exactly once, in 8-aligned chunks. Safe to call from any goroutine.
func checkOnce(t *testing.T, N, width int, inner func()) {
	t.Helper()
	hits := make([]atomic.Int32, N)
	var bad atomic.Bool
	parallelSpawnCols(N, width, func(j0, j1 int) {
		if j0%8 != 0 || j1 <= j0 || j1 > N {
			bad.Store(true)
		}
		for j := j0; j < j1; j++ {
			hits[j].Add(1)
		}
		if inner != nil {
			inner()
		}
	})
	if bad.Load() {
		t.Errorf("N=%d width=%d: a chunk was not 8-aligned or out of range", N, width)
	}
	for j := range hits {
		if got := hits[j].Load(); got != 1 {
			t.Errorf("N=%d width=%d: column %d ran %d times", N, width, j, got)
			return
		}
	}
}

// TestSpanPool_eachColumnOnce: concurrent callers and calls nested inside a chunk (a helper
// that dispatches its own parallel span) each cover their columns exactly once and return.
func TestSpanPool_eachColumnOnce(t *testing.T) {
	var wg sync.WaitGroup
	for g := range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range 50 {
				N := 1 + (g*131+i*977)%5000
				var nested func()
				if i%5 == 0 {
					nested = func() { checkOnce(t, 64, 4, nil) }
				}
				checkOnce(t, N, 1+(g+i)%9, nested)
			}
		}()
	}
	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(60 * time.Second):
		t.Fatal("parallelSpawnCols did not return: pool deadlock")
	}
}

// waitParked waits until every pool helper is parked on the cond, failing after a deadline.
func waitParked(t *testing.T) int {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		w, parked := spanWorkers.idleForTest()
		if w > 0 && w == parked && spanWorkers.spinning.Load() == 0 {
			return w
		}
		if time.Now().After(deadline) {
			t.Fatalf("helpers did not park: %d workers, %d parked, %d spinning", w, parked, spanWorkers.spinning.Load())
		}
		time.Sleep(time.Millisecond)
	}
}

// TestSpanPool_parksWhenIdle: after the last parallel call, helpers leave their spin window
// and park, and a parked pool still serves the next call.
func TestSpanPool_parksWhenIdle(t *testing.T) {
	checkOnce(t, 4096, 4, nil)
	if w := waitParked(t); w < 3 {
		t.Fatalf("pool has %d helpers after a width-4 call, want at least 3", w)
	}
	checkOnce(t, 4096, 4, nil)
	waitParked(t)
}
