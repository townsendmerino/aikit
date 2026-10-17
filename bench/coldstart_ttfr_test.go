package bench

import (
	"os"
	"slices"
	"testing"
	"time"
)

// TestW3_timeToFirstResult_heapVsMapped is the measurement behind v1.71.0's Model2Vec default
// (docs/internal/task-memory-campaign.md item 8): the embedded-corpus example's whole cold start, model load to
// first fused result (coldStart), with the model heap-read (LoadFromFSHeap, the old default) and mapped
// (LoadFromFS, the new one), interleaved in one process so drift hits both arms alike. The page cache is warm
// (the file was just read), so it measures what a mapped load defers to the first Encode: the minor faults of a
// fresh mapping. A measurement, not a gate: set AIKIT_W3_TTFR=1 to run it.
func TestW3_timeToFirstResult_heapVsMapped(t *testing.T) {
	if os.Getenv("AIKIT_W3_TTFR") == "" {
		t.Skip("a measurement; set AIKIT_W3_TTFR=1 to run it")
	}
	modelDir, indexBlob, corpusJSON := coldStartAssets(t)
	const warmup, rounds = 2, 15
	var heap, mapped []time.Duration
	for r := range warmup + rounds {
		for _, useMmap := range []bool{false, true} {
			start := time.Now()
			coldStart(t, modelDir, indexBlob, corpusJSON, useMmap)
			d := time.Since(start)
			if r < warmup {
				continue
			}
			if useMmap {
				mapped = append(mapped, d)
			} else {
				heap = append(heap, d)
			}
		}
	}
	med := func(ds []time.Duration) time.Duration { s := slices.Clone(ds); slices.Sort(s); return s[len(s)/2] }
	rng := func(ds []time.Duration) (time.Duration, time.Duration) { return slices.Min(ds), slices.Max(ds) }
	hl, hh := rng(heap)
	ml, mh := rng(mapped)
	t.Logf("time to first result, %d interleaved rounds: heap median %v (range %v–%v); mapped median %v (range %v–%v); mapped/heap %.2f×",
		rounds, med(heap).Round(10*time.Microsecond), hl.Round(10*time.Microsecond), hh.Round(10*time.Microsecond),
		med(mapped).Round(10*time.Microsecond), ml.Round(10*time.Microsecond), mh.Round(10*time.Microsecond),
		float64(med(mapped))/float64(med(heap)))
}
