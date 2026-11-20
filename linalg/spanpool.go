package linalg

import (
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// spanJob is one parallelSpawnCols call: [0,n) handed out grain columns at a time from next.
// A job is recycled through spanJobs once its caller has retracted it from the queue and every
// helper that joined it has left, so nothing outside run holds a pointer to it after run returns.
type spanJob struct {
	fn     func(j0, j1 int)
	n      int64
	grain  int64
	next   atomic.Int64
	want   int          // helpers still wanted; guarded by spanPool.mu
	active atomic.Int32 // helpers that joined and have not finished draining
}

var spanJobs = sync.Pool{New: func() any { return new(spanJob) }}

// drain runs chunks until none is left. Each column is claimed by exactly one drainer.
func (j *spanJob) drain() {
	for {
		j0 := j.next.Add(j.grain) - j.grain
		if j0 >= j.n {
			return
		}
		j.fn(int(j0), int(min(j0+j.grain, j.n)))
	}
}

// spanPool is the process-wide set of helper goroutines behind parallelSpawnCols. It is safe for
// concurrent and nested use: the calling goroutine drains its own job too, so a call completes
// even when every helper is busy elsewhere; helpers only make it finish sooner. A helper that
// finds no work spins for spanSpin (so a decode layer's back-to-back matmuls find it awake), then
// parks on cond and uses no CPU until the next call. Helpers are started on demand up to the
// largest helper count any call has asked for, and are never stopped.
type spanPool struct {
	mu      sync.Mutex
	cond    sync.Cond
	queue   []*spanJob // jobs with want > 0
	workers int        // helpers started
	parked  int        // helpers waiting on cond

	wanted   atomic.Int64 // sum of want over queue, polled lock-free by spinning helpers
	spinning atomic.Int32 // helpers inside their spin window
}

// spanSpin is how long an idle helper polls for new work before parking.
var spanSpin = 50 * time.Microsecond

var spanWorkers = newSpanPool()

func newSpanPool() *spanPool {
	p := &spanPool{}
	p.cond.L = &p.mu
	return p
}

// run executes fn over [0,n) in grain-sized chunks on the caller plus up to helpers pool
// goroutines, and returns once every chunk has finished.
func (p *spanPool) run(n, grain, helpers int, fn func(j0, j1 int)) {
	j := spanJobs.Get().(*spanJob)
	j.fn, j.n, j.grain = fn, int64(n), int64(grain)
	j.next.Store(0)

	p.mu.Lock()
	for p.workers < helpers {
		p.workers++
		go p.worker()
	}
	j.want = helpers
	p.queue = append(p.queue, j)
	p.wanted.Add(int64(helpers))
	for i := 0; i < min(p.parked, helpers); i++ {
		p.cond.Signal()
	}
	p.mu.Unlock()

	j.drain()

	// Retract the job so no helper joins it late, then wait for those that joined. Their
	// in-flight chunks are short, so the caller spins rather than parks.
	p.mu.Lock()
	if j.want > 0 {
		p.wanted.Add(-int64(j.want))
		j.want = 0
		p.removeLocked(j)
	}
	p.mu.Unlock()
	for j.active.Load() != 0 {
		runtime.Gosched()
	}
	j.fn = nil
	spanJobs.Put(j)
}

func (p *spanPool) removeLocked(j *spanJob) {
	for i, q := range p.queue {
		if q == j {
			copy(p.queue[i:], p.queue[i+1:])
			p.queue[len(p.queue)-1] = nil
			p.queue = p.queue[:len(p.queue)-1]
			return
		}
	}
}

// popLocked joins the oldest queued job, or returns nil when the queue is empty.
func (p *spanPool) popLocked() *spanJob {
	if len(p.queue) == 0 {
		return nil
	}
	j := p.queue[0]
	j.active.Add(1)
	j.want--
	p.wanted.Add(-1)
	if j.want == 0 {
		p.removeLocked(j)
	}
	return j
}

func (p *spanPool) worker() {
	for {
		j := p.take()
		j.drain()
		j.active.Add(-1) // j may be recycled from here on
	}
}

// take returns the next job to help with: polled for spanSpin, then waited for on cond.
func (p *spanPool) take() *spanJob {
	p.spinning.Add(1)
	var deadline time.Time
	for {
		for range 64 {
			if p.wanted.Load() > 0 {
				p.mu.Lock()
				j := p.popLocked()
				p.mu.Unlock()
				if j != nil {
					p.spinning.Add(-1)
					return j
				}
			}
		}
		now := time.Now()
		if deadline.IsZero() {
			deadline = now.Add(spanSpin)
		} else if now.After(deadline) {
			break
		}
		runtime.Gosched()
	}
	p.spinning.Add(-1)

	p.mu.Lock()
	defer p.mu.Unlock()
	for {
		if j := p.popLocked(); j != nil {
			return j
		}
		p.parked++
		p.cond.Wait()
		p.parked--
	}
}
