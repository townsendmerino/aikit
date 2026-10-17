package bench

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/ann"
	"github.com/townsendmerino/aikit/bm25"
	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/fuse"
)

// W3 — cold start, and the peak-heap instrument the footprint findings need.
// Lens doc §5.6 N9: "cold-start warm-up is real and nothing measures it."
//
// WHY THIS IS SEPARATE FROM W1/W2. Those measure steady-state throughput, which
// is the wrong instrument for every remaining item in the lens doc: §4.3
// (MarshalBinary doubles peak RSS), §4.5 (LoadWeightsQ8 holds the f32 model and
// its int8 copy at once), §4.7 (QueryBatch allocates the whole M×N score
// matrix), §4.8 (chunker buffer sizing) and N4 (bm25 has no serialization
// surface, 67% of this example's cold start) are all peak-memory or
// time-to-first-result claims. `B/op` does not arbitrate any of them — it counts
// bytes ALLOCATED over a run, and a doubling of peak RSS can happen with no
// change in that total, while a large pool can change it with no change in peak.
//
// So this file reports a `peakMiB` metric alongside the usual ones. It is a
// sampled maximum of runtime.MemStats.HeapInuse, which is an approximation and
// is documented as one below — but it is an approximation of the right quantity,
// where B/op is an exact measure of the wrong one.
//
// The transcription is faithful to examples/embedded-corpus/main.go: load the
// model, load the int8 index blob, unmarshal the corpus JSON, rebuild the BM25
// index (it has no on-disk form), then run the first query.

const coldAssets = "../examples/embedded-corpus/assets"

// coldChunk mirrors the example's Chunk. Only Text is read here.
type coldChunk struct {
	Source string `json:"source"`
	Ref    string `json:"ref"`
	Text   string `json:"text"`
}

func coldStartAssets(tb testing.TB) (modelDir string, indexBlob, corpusJSON []byte) {
	tb.Helper()
	modelDir = filepath.Join(coldAssets, "model")
	if _, err := os.Stat(filepath.Join(modelDir, "model.safetensors")); err != nil {
		tb.Skipf("no embedded-corpus assets at %s — run its `go generate`", coldAssets)
	}
	var err error
	if indexBlob, err = os.ReadFile(filepath.Join(coldAssets, "index.bin")); err != nil {
		tb.Skipf("no index.bin: %v", err)
	}
	if corpusJSON, err = os.ReadFile(filepath.Join(coldAssets, "corpus.json")); err != nil {
		tb.Skipf("no corpus.json: %v", err)
	}
	return modelDir, indexBlob, corpusJSON
}

// peakHeapMiB runs fn and returns the largest HeapInuse observed while it ran,
// in MiB.
//
// HOW GOOD IS THIS NUMBER. It samples rather than instruments, so it can miss a
// spike shorter than the sampling interval, and HeapInuse counts spans the
// allocator holds rather than RSS the kernel reports — the two differ by
// whatever the allocator has not returned. It is therefore a LOWER BOUND on peak
// heap and a loose proxy for peak RSS.
//
// That is still the right tool for the findings it exists to arbitrate: they
// predict doublings, not percentages. A 2× claim survives a sampler that might
// miss a microsecond-scale spike; it would not survive being measured with B/op,
// which answers a different question entirely. Where a finding turns out to hinge
// on a few percent, this instrument is not enough and the commit should say so.
func peakHeapMiB(fn func()) float64 {
	runtime.GC()
	var stop atomic.Bool
	var peak atomic.Uint64
	done := make(chan struct{})
	go func() {
		defer close(done)
		var ms runtime.MemStats
		for !stop.Load() {
			runtime.ReadMemStats(&ms)
			for {
				cur := peak.Load()
				if ms.HeapInuse <= cur || peak.CompareAndSwap(cur, ms.HeapInuse) {
					break
				}
			}
			time.Sleep(200 * time.Microsecond)
		}
	}()
	fn()
	stop.Store(true)
	<-done
	return float64(peak.Load()) / (1 << 20)
}

// BenchmarkW3 is the cold-start path of examples/embedded-corpus, stage by
// stage, with `sum` running the whole thing so stage-sum is a completeness check.
//
// Every sub-benchmark reports peakMiB as well as time, because for this workload
// the memory number is the one several open findings are about.
func BenchmarkW3(b *testing.B) {
	modelDir, indexBlob, corpusJSON := coldStartAssets(b)

	var corpus []coldChunk
	if err := json.Unmarshal(corpusJSON, &corpus); err != nil {
		b.Fatal(err)
	}
	docs := make([][]string, len(corpus))
	for i, c := range corpus {
		docs[i] = bm25.TokenizePlain(c.Text)
	}
	b.Logf("assets: model %s, index.bin %d KB, corpus.json %d KB, %d chunks",
		modelDir, len(indexBlob)/1024, len(corpusJSON)/1024, len(corpus))

	withPeak := func(b *testing.B, fn func()) {
		b.ReportAllocs()
		for b.Loop() {
			fn()
		}
		b.StopTimer()
		b.ReportMetric(peakHeapMiB(fn), "peakMiB")
		b.StartTimer()
	}

	b.Run("loadModel", func(b *testing.B) {
		withPeak(b, func() {
			m, err := embed.LoadFromFSHeap(os.DirFS(modelDir), ".")
			if err != nil {
				b.Fatal(err)
			}
			sinkColdModel = m
		})
	})
	b.Run("loadModelMmap", func(b *testing.B) {
		withPeak(b, func() {
			m, err := embed.LoadMmap(modelDir)
			if err != nil {
				b.Fatal(err)
			}
			sinkColdModel = m
		})
	})
	b.Run("loadIndex", func(b *testing.B) {
		withPeak(b, func() {
			ix, err := ann.LoadFlatI8(indexBlob)
			if err != nil {
				b.Fatal(err)
			}
			sinkFlatI8 = ix
		})
	})
	b.Run("unmarshalCorpus", func(b *testing.B) {
		withPeak(b, func() {
			var c []coldChunk
			if err := json.Unmarshal(corpusJSON, &c); err != nil {
				b.Fatal(err)
			}
			sinkColdCorpus = c
		})
	})
	b.Run("tokenizePlain", func(b *testing.B) {
		withPeak(b, func() {
			d := make([][]string, len(corpus))
			for i, c := range corpus {
				d[i] = bm25.TokenizePlain(c.Text)
			}
			sinkColdDocs = d
		})
	})
	b.Run("bm25Build", func(b *testing.B) {
		withPeak(b, func() { sinkIndex = bm25.Build(docs) })
	})
	b.Run("sum", func(b *testing.B) {
		withPeak(b, func() { coldStart(b, modelDir, indexBlob, corpusJSON, false) })
	})
	b.Run("sumMmap", func(b *testing.B) {
		withPeak(b, func() { coldStart(b, modelDir, indexBlob, corpusJSON, true) })
	})
}

// coldStart is the example's startup path, start to first result.
func coldStart(tb testing.TB, modelDir string, indexBlob, corpusJSON []byte, useMmap bool) {
	// LoadFromFS maps an os.DirFS model since v1.71.0, so the heap arm names its loader explicitly.
	load := func() (*embed.StaticModel, error) { return embed.LoadFromFSHeap(os.DirFS(modelDir), ".") }
	if useMmap {
		load = func() (*embed.StaticModel, error) { return embed.LoadFromFS(os.DirFS(modelDir), ".") }
	}
	model, err := load()
	if err != nil {
		tb.Fatal(err)
	}
	defer model.Close() // a mapped model keeps its mapping until Close
	index, err := ann.LoadFlatI8(indexBlob)
	if err != nil {
		tb.Fatal(err)
	}
	var corpus []coldChunk
	if err := json.Unmarshal(corpusJSON, &corpus); err != nil {
		tb.Fatal(err)
	}
	docs := make([][]string, len(corpus))
	for i, c := range corpus {
		docs[i] = bm25.TokenizePlain(c.Text)
	}
	lex := bm25.Build(docs)

	// First query — cold caches, cold branch predictors, nothing warmed.
	const q = "how do i build an index over my own documents"
	qv := model.Encode(q)
	dense := index.Query(qv, 50)
	lexHits := lex.TopK(bm25.TokenizePlain(q), 50)
	sinkFused = fuse.RRF(fuse.DefaultK,
		fuse.Keys(dense, func(h ann.Hit) int { return h.Index }),
		fuse.Keys(lexHits, func(r bm25.Result) int { return r.Doc }))
}

// BenchmarkW3FirstQueryIsCold measures the thing N9 says nothing measures: how
// much more the FIRST query costs than a warm one, on the same process.
//
// bench/harness.go gained a warm-up pass for exactly this reason (Step 0b) — it
// was reporting a p99 that was mostly the cold first query. That fixed the
// harness's own reporting; it did not produce a number for the effect.
//
// MEASURED, AND THE ANSWER IS "NOT IN THIS PROCESS": 78.7 µs on a freshly loaded
// index against 82.7 µs warm — the cold arm is 5% FASTER, i.e. nothing. That is
// a limit of the instrument, not a refutation of the effect: reloading the index
// leaves the code, the allocator and the model warm, so the only thing made cold
// is the data, and 443 KB of it does not miss enough to show. A real first-query
// penalty needs a genuinely cold process, which a Go benchmark cannot be.
//
// It is kept as a recorded negative. The claim was floating unmeasured; now the
// limit of what can be measured in-process is written down, and BenchmarkW3
// covers what actually dominates cold start — loading, not the first query,
// which is 0.2% of it.
func BenchmarkW3FirstQueryIsCold(b *testing.B) {
	modelDir, indexBlob, corpusJSON := coldStartAssets(b)
	model, err := embed.LoadFromFS(os.DirFS(modelDir), ".")
	if err != nil {
		b.Fatal(err)
	}
	index, err := ann.LoadFlatI8(indexBlob)
	if err != nil {
		b.Fatal(err)
	}
	var corpus []coldChunk
	if err := json.Unmarshal(corpusJSON, &corpus); err != nil {
		b.Fatal(err)
	}
	docs := make([][]string, len(corpus))
	for i, c := range corpus {
		docs[i] = bm25.TokenizePlain(c.Text)
	}
	lex := bm25.Build(docs)
	queries := []string{
		"how do i build an index over my own documents",
		"quantize vectors to int8 for a smaller index",
		"fuse lexical and dense rankings",
		"tokenize source code into terms",
	}
	run := func(q string) {
		qv := model.Encode(q)
		dense := index.Query(qv, 50)
		lexHits := lex.TopK(bm25.TokenizePlain(q), 50)
		sinkFused = fuse.RRF(fuse.DefaultK,
			fuse.Keys(dense, func(h ann.Hit) int { return h.Index }),
			fuse.Keys(lexHits, func(r bm25.Result) int { return r.Doc }))
	}

	// Warm: the steady state every other benchmark in this repo reports. It runs
	// THE SAME query as the cold arm below — an earlier version cycled four and
	// the comparison was meaningless, since the arms then differed in workload as
	// well as in warmth.
	b.Run("warm", func(b *testing.B) {
		for range 32 {
			run(queries[0])
		}
		for b.Loop() {
			run(queries[0])
		}
	})
	// Cold-ish: a freshly loaded dense index per iteration, so the query runs
	// against structures nothing has touched.
	//
	// Only the ANN index is rebuilt, not the BM25 one. That is a measurement
	// constraint, not a modelling choice: rebuilding BM25 costs 8.8 ms of
	// untimed setup against 0.09 ms of timed work, so Go — which raises the
	// iteration count until the TIMED total reaches -benchtime — spends minutes
	// of wall clock per second of measurement. The first version of this
	// benchmark did exactly that and had to be killed. Run this one with a fixed
	// small count (`-benchtime 300x`); a time-based benchtime is still wasteful
	// even at 0.24 ms of setup.
	b.Run("firstOnFreshIndex", func(b *testing.B) {
		for b.Loop() {
			b.StopTimer()
			ix, err := ann.LoadFlatI8(indexBlob)
			if err != nil {
				b.Fatal(err)
			}
			b.StartTimer()
			qv := model.Encode(queries[0])
			dense := ix.Query(qv, 50)
			lexHits := lex.TopK(bm25.TokenizePlain(queries[0]), 50)
			sinkFused = fuse.RRF(fuse.DefaultK,
				fuse.Keys(dense, func(h ann.Hit) int { return h.Index }),
				fuse.Keys(lexHits, func(r bm25.Result) int { return r.Doc }))
		}
	})
}

var (
	sinkColdModel  *embed.StaticModel
	sinkColdCorpus []coldChunk
	sinkColdDocs   [][]string
)
