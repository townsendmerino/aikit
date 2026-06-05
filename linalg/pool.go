package linalg

import (
	"runtime"
	"sync"
	"sync/atomic"
)

// pool is a persistent set of worker goroutines for a column-parallel matmul,
// driven by a SINGLE dispatcher goroutine (it lives in a Workspace, which is
// itself per-goroutine / not concurrent-safe). The point over spawning
// goroutines per matmul: the workers spin briefly before parking, so the ~4
// back-to-back dispatches per decode layer find them still hot and skip the
// futex park/wake that dominated the post-v0.5.0 decode profile.
//
// One dispatch = one generation: the dispatcher publishes (fn, N) and bumps
// gen; each worker computes its even column chunk for that gen, runs it, and
// increments done. The dispatcher runs chunk 0 itself and spin-waits for the
// workers' chunks. Results are independent per output column, so this is
// numerically identical to the serial path (parity-safe by construction).
type pool struct {
	workers int // background goroutines; the dispatcher is the +1th worker

	gen    atomic.Uint64 // bumped once per dispatch; workers spin/park on changes
	done   atomic.Int64  // background workers that finished the current gen
	parked atomic.Int64  // workers currently blocked on the cond
	stop   atomic.Bool

	curN  int              // published before gen bump, read after gen observe
	curFn func(j0, j1 int) // "

	mu   sync.Mutex // guards the park; cond wakes parked workers on dispatch
	cond *sync.Cond
}

// poolSpin is how many gen-load iterations a worker busy-checks for the next
// dispatch before parking on the cond. Sized so the sub-µs gaps between a
// layer's back-to-back matmuls are covered by spinning (worker stays hot),
// while a longer idle (between tokens) parks and stops burning a core.
const poolSpin = 1 << 12

func newPool(workers int) *pool {
	if workers < 1 {
		workers = 1
	}
	p := &pool{workers: workers - 1} // dispatcher does one chunk itself
	p.cond = sync.NewCond(&p.mu)
	for i := 1; i <= p.workers; i++ {
		go p.worker(i)
	}
	return p
}

// chunkBounds splits [0,N) into `total` even chunks and returns chunk idx.
func chunkBounds(idx, total, N int) (int, int) {
	chunk := (N + total - 1) / total
	j0 := idx * chunk
	if j0 > N {
		j0 = N
	}
	j1 := j0 + chunk
	if j1 > N {
		j1 = N
	}
	return j0, j1
}

func (p *pool) worker(idx int) {
	last := uint64(0)
	for {
		// Spin on gen, then park on the cond if no dispatch arrives. The
		// park's re-check under the lock closes the lost-wakeup window: a
		// dispatch bumps gen (atomic) before Broadcast, so a worker that
		// already observed the bump never waits on it.
		g := p.gen.Load()
		for g == last {
			if p.stop.Load() {
				return
			}
			spun := 0
			for spun < poolSpin && g == last {
				spun++
				g = p.gen.Load()
			}
			if g != last {
				break
			}
			// Park. parked++ is announced (under mu) BEFORE the final gen
			// re-check: a dispatcher does gen.Add THEN reads parked, so if it
			// bumped gen after our re-check saw the old value, it is guaranteed
			// to see our increment and Broadcast — no lost wakeup, while a
			// dispatcher with all workers spinning skips the lock entirely.
			p.mu.Lock()
			p.parked.Add(1)
			for p.gen.Load() == last && !p.stop.Load() {
				p.cond.Wait()
			}
			p.parked.Add(-1)
			g = p.gen.Load()
			p.mu.Unlock()
		}
		if p.stop.Load() {
			return
		}
		last = g
		fn, n := p.curFn, p.curN
		j0, j1 := chunkBounds(idx, p.workers+1, n)
		if fn != nil && j0 < j1 {
			fn(j0, j1)
		}
		p.done.Add(1)
	}
}

// run executes fn over [0,N) split across the dispatcher + workers, returning
// once every chunk is done. Caller is the single dispatcher (the Workspace
// owner). Falls back to a direct call when there are no background workers.
func (p *pool) run(N int, fn func(j0, j1 int)) {
	if p.workers == 0 {
		fn(0, N)
		return
	}
	p.curN = N
	p.curFn = fn
	p.done.Store(0)
	p.gen.Add(1) // publish: happens-after the curFn/curN writes (program order)
	// Only take the lock when a worker is actually parked; when they're all
	// still spinning (the back-to-back decode case) this is a lock-free bump.
	if p.parked.Load() > 0 {
		p.mu.Lock()
		p.cond.Broadcast()
		p.mu.Unlock()
	}

	// The dispatcher runs chunk 0 on its own core.
	if j0, j1 := chunkBounds(0, p.workers+1, N); j0 < j1 {
		fn(j0, j1)
	}
	// Spin (don't park) waiting for the workers — the work is µs and we need
	// the result immediately; parking the dispatcher would reintroduce the
	// very wakeup cost we're avoiding.
	for p.done.Load() < int64(p.workers) {
		runtime.Gosched()
	}
	p.curFn = nil // release the closure (no concurrent reader past the barrier)
}

// close stops the workers. Idempotent; the Workspace owner calls it when done.
func (p *pool) close() {
	p.stop.Store(true)
	p.mu.Lock()
	p.cond.Broadcast()
	p.mu.Unlock()
}
