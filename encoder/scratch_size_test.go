package encoder

import (
	"sync"
	"testing"
)

const mib = 1 << 20

// scratchBytes sums the pooled footprint of s — every buffer the forward can size,
// workerScratch included — as lenB (what the forward asked for) and capB (what is
// actually pinned, including ensureF32's 25% growth headroom). Both are reported:
// cap is the real memory, len is what a layout change moves.
func scratchBytes(s *scratch) (lenB, capB int) {
	f32 := [][]float32{
		s.qkv, s.Q, s.K, s.V, s.ctx, s.out,
		s.val, s.gate, s.mid, s.upGate,
		s.qH, s.kH, s.vH, s.ctxHead, s.scores,
		s.moeAllScores, s.moeWeight, s.moeGathered, s.moeAcc,
		s.deqW,
	}
	for i := range s.workerScratch {
		hs := &s.workerScratch[i]
		f32 = append(f32, hs.qH, hs.kH, hs.vH, hs.ctxHead, hs.scores)
	}
	for _, b := range f32 {
		lenB += 4 * len(b)
		capB += 4 * cap(b)
	}
	for _, b := range [][]int32{s.moeExpert, s.moeTokens} {
		lenB += 4 * len(b)
		capB += 4 * cap(b)
	}
	return lenB, capB
}

// scratchRecorder watches every scratch returned to the pool while installed. Batched
// forwards return one scratch per worker concurrently, so it keeps both the largest single
// scratch and the pinned total across all of them.
type scratchRecorder struct {
	mu         sync.Mutex
	lenB, capB int
	workers    int              // largest workerScratch set seen (0 = head-parallel never engaged)
	byScratch  map[*scratch]int // latest cap per distinct scratch, for the pinned total
}

// trackScratch installs a recorder for the rest of b. Call it before the timed loop and
// report after it; the hook is removed on cleanup so no other benchmark inherits it.
//
// It also empties the pool first. Pooled buffers only grow, so a scratch that an earlier
// benchmark in the same process sized for a bigger model would otherwise be reported as
// this one's footprint.
func trackScratch(b *testing.B) *scratchRecorder {
	b.Helper()
	scratchPool = sync.Pool{New: func() any { return &scratch{} }}
	r := &scratchRecorder{byScratch: map[*scratch]int{}}
	scratchPutHook = func(s *scratch) {
		l, c := scratchBytes(s)
		r.mu.Lock()
		r.lenB, r.capB = max(r.lenB, l), max(r.capB, c)
		r.workers = max(r.workers, len(s.workerScratch))
		r.byScratch[s] = c
		r.mu.Unlock()
	}
	b.Cleanup(func() { scratchPutHook = nil })
	return r
}

// report emits scratch-MiB (largest single scratch, pinned cap), scratch-len-MiB (its
// requested len), scratch-total-MiB (pinned cap summed over every distinct scratch the
// run used — the batched path's real footprint) and head-workers (the head-parallel
// fan-out, 0 if it never engaged).
func (r *scratchRecorder) report(b *testing.B) {
	b.Helper()
	r.mu.Lock()
	defer r.mu.Unlock()
	b.ReportMetric(float64(r.capB)/mib, "scratch-MiB")
	b.ReportMetric(float64(r.lenB)/mib, "scratch-len-MiB")
	total := 0
	for _, c := range r.byScratch {
		total += c
	}
	b.ReportMetric(float64(total)/mib, "scratch-total-MiB")
	b.ReportMetric(float64(r.workers), "head-workers")
}
