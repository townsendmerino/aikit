package ann

import (
	"fmt"
	"github.com/townsendmerino/aikit/linalg"
	"runtime"
	"sync"
	"testing"
)

// Roofline probes for the Flat f32 cosine scan (docs/task-archsimd-eval.md).
//
// The question these answer: is ann.Flat.Query compute-bound (so a wider SIMD
// kernel — AVX-512 FMA, or a simd-package rewrite — could move p50) or
// memory-bandwidth-bound (so it cannot)?
//
// Arithmetic intensity settles the shape of the answer before any measurement:
// one dim-D f32 candidate is D*4 bytes and D multiply-accumulates, i.e. a fixed
// 0.25 MAC per byte at EVERY D. A dot product reads each operand once and
// reuses nothing, so no kernel — scalar, NEON, AVX2 or AVX-512 — can change
// that ratio. It is a property of the operation, not of the implementation.
// The only lever a kernel has is where on the roofline the scan actually sits,
// which is what these probes measure.
//
// benchReadBW is the ceiling anchor. Per the SIMD audit's own rule ("a
// bandwidth probe without a ceiling assert is not a probe"), it uses a
// data-dependent fill and a package-level sink so the compiler cannot fold the
// reduction, and the result is asserted against the box's theoretical DRAM
// ceiling.

var bwSink uint64

// benchReadBW measures pure streaming read bandwidth over a buffer far larger
// than any cache, at the given goroutine count. This is the roofline's slanted
// ceiling for a read-only streaming kernel like the flat scan.
func benchReadBW(b *testing.B, threads int) {
	const bytes = 512 << 20 // 512 MB — past any L2/SLC on this class of box
	const n = bytes / 8
	buf := make([]uint64, n)
	// Data-dependent fill: a constant-filled array lets the compiler (or the
	// hardware) collapse the reduction and report a fictional bandwidth.
	for i := range buf {
		buf[i] = uint64(i)*2654435761 + 1
	}
	b.SetBytes(bytes)
	b.ResetTimer()
	for range b.N {
		var wg sync.WaitGroup
		chunk := n / threads
		for t := range threads {
			wg.Add(1)
			go func(lo, hi int) {
				defer wg.Done()
				// Eight independent accumulators: a single `s += v` chain
				// is limited by add latency (~1/cycle/core), which on this
				// box caps out near 19 GB/s and would be mistaken for a
				// memory ceiling. Unrolling breaks the dependency so the
				// loop is limited by load/DRAM, which is what we want.
				var s0, s1, s2, s3, s4, s5, s6, s7 uint64
				c := buf[lo:hi]
				j := 0
				for ; j+8 <= len(c); j += 8 {
					s0 += c[j]
					s1 += c[j+1]
					s2 += c[j+2]
					s3 += c[j+3]
					s4 += c[j+4]
					s5 += c[j+5]
					s6 += c[j+6]
					s7 += c[j+7]
				}
				for ; j < len(c); j++ {
					s0 += c[j]
				}
				bwSink += s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
			}(t*chunk, min((t+1)*chunk, n))
		}
		wg.Wait()
	}
}

func BenchmarkReadBW(b *testing.B) {
	for _, t := range []int{1, 2, 4, 6, 8} {
		if t > runtime.NumCPU() {
			continue
		}
		b.Run(fmt.Sprintf("threads%d", t), func(b *testing.B) { benchReadBW(b, t) })
	}
}

// BenchmarkFlatScanSerial measures the SINGLE-THREADED scan kernel (scanFlat,
// i.e. linalg.Dot8x4 plus the emit closure) with no sharding and no top-k, so
// the number is the kernel's own streaming rate, uncontaminated by fan-out and
// heap pushes. Sweeping N at fixed dim walks the working set from L1 through
// L2 and out to DRAM, which is the whole point: the cache-resident sizes and
// the DRAM-resident sizes give different verdicts, and only one of them is
// what ken actually runs.
func BenchmarkFlatScanSerial(b *testing.B) {
	for _, d := range []int{256, 768} {
		for _, n := range []int{1_000, 8_000, 50_000, 200_000} {
			wsMB := float64(n*d*4) / (1 << 20)
			b.Run(fmt.Sprintf("d%d/N%d_%.1fMB", d, n, wsMB), func(b *testing.B) {
				corpus := makeUnitVectors(n, d, 0xfeed)
				query := makeUnitVectors(1, d, 0xdade)[0]
				var sink float64
				b.SetBytes(int64(n) * int64(d) * 4)
				b.ResetTimer()
				for range b.N {
					scanFlat(query, corpus, func(_ int, s float64) { sink += s })
				}
				b.StopTimer()
				bwSink += uint64(sink)
			})
		}
	}
}

// makeUnitVectorsContig is makeUnitVectors with one flat backing array: row i is
// backing[i*d:(i+1)*d], so rows are exactly adjacent with no allocator span
// boundaries between them. Same values, same seed — only the layout differs.
func makeUnitVectorsContig(n, d int, seed uint64) [][]float32 {
	rows := makeUnitVectors(n, d, seed)
	backing := make([]float32, n*d)
	out := make([][]float32, n)
	for i, r := range rows {
		copy(backing[i*d:(i+1)*d], r)
		out[i] = backing[i*d : (i+1)*d : (i+1)*d]
	}
	return out
}

// BenchmarkFlatScanLayout isolates memory LAYOUT from kernel compute. Same
// scanFlat, same dims, same N — the only difference is whether the corpus rows
// are separately allocated ([][]float32 from the allocator, which is what
// ann.New is handed today) or carved from one contiguous backing array.
//
// If the scan were compute-bound the two must measure the same. A gap is
// memory-side, and says the lever is layout, not a wider multiply.
func BenchmarkFlatScanLayout(b *testing.B) {
	for _, d := range []int{256, 768} {
		for _, n := range []int{50_000, 200_000} {
			for _, mode := range []string{"scattered", "contig"} {
				b.Run(fmt.Sprintf("d%d/N%d/%s", d, n, mode), func(b *testing.B) {
					var corpus [][]float32
					if mode == "contig" {
						corpus = makeUnitVectorsContig(n, d, 0xfeed)
					} else {
						corpus = makeUnitVectors(n, d, 0xfeed)
					}
					query := makeUnitVectors(1, d, 0xdade)[0]
					var sink float64
					b.SetBytes(int64(n) * int64(d) * 4)
					b.ResetTimer()
					for range b.N {
						scanFlat(query, corpus, func(_ int, s float64) { sink += s })
					}
					b.StopTimer()
					bwSink += uint64(sink)
				})
			}
		}
	}
}

// scanFlatTiledK is scanFlat with the K (dimension) axis tiled into strips, per
// the explicit warning on linalg.Dot8x4: its 8 live accumulators plus 8 streamed
// b-rows outgrow the register/cache budget at large K, and a caller handing it a
// single large-K row "should prefer Dot4x4" or feed it strips. ann.scanFlat
// hands it the whole row. This variant measures what that costs.
//
// NOTE: tiling K re-associates the summation, so this is a MEASUREMENT probe,
// not a drop-in replacement — see the findings doc on numerics.
func scanFlatTiledK(q []float32, vecs [][]float32, strip int, emit func(i int, score float64)) {
	d := len(q)
	n4 := d / 4
	tailStart := n4 * 4
	var sums [32]float32
	var acc [8]float32
	i := 0
	for ; d > 0 && i+8 <= len(vecs); i += 8 {
		v0, v1, v2, v3 := vecs[i], vecs[i+1], vecs[i+2], vecs[i+3]
		v4, v5, v6, v7 := vecs[i+4], vecs[i+5], vecs[i+6], vecs[i+7]
		acc = [8]float32{}
		for k0 := 0; k0 < tailStart; k0 += strip {
			k1 := min(k0+strip, tailStart)
			linalg.Dot8x4(&q[k0], &v0[k0], &v1[k0], &v2[k0], &v3[k0],
				&v4[k0], &v5[k0], &v6[k0], &v7[k0], (k1-k0)/4, &sums)
			for j := range 8 {
				b := j * 4
				acc[j] += sums[b] + sums[b+1] + sums[b+2] + sums[b+3]
			}
		}
		group := [8][]float32{v0, v1, v2, v3, v4, v5, v6, v7}
		for j := range 8 {
			s := acc[j]
			for kk := tailStart; kk < d; kk++ {
				s += q[kk] * group[j][kk]
			}
			emit(i+j, float64(s))
		}
	}
	for ; i < len(vecs); i++ {
		if v := vecs[i]; len(v) == d {
			emit(i, float64(linalg.Dot(q, v)))
		}
	}
}

// BenchmarkFlatScanTileK sweeps the K-strip width at d=768 (CodeRankEmbed's
// dimension) against the untiled scanFlat baseline (strip=0).
func BenchmarkFlatScanTileK(b *testing.B) {
	const d, n = 768, 200_000
	corpus := makeUnitVectors(n, d, 0xfeed)
	query := makeUnitVectors(1, d, 0xdade)[0]
	for _, strip := range []int{0, 64, 128, 256, 384} {
		name := "untiled"
		if strip > 0 {
			name = fmt.Sprintf("strip%d", strip)
		}
		b.Run(name, func(b *testing.B) {
			var sink float64
			b.SetBytes(int64(n) * int64(d) * 4)
			b.ResetTimer()
			for range b.N {
				if strip == 0 {
					scanFlat(query, corpus, func(_ int, s float64) { sink += s })
				} else {
					scanFlatTiledK(query, corpus, strip, func(_ int, s float64) { sink += s })
				}
			}
			b.StopTimer()
			bwSink += uint64(sink)
		})
	}
}

// BenchmarkFlatScanScaling is the decision gate. It runs the SAME scanFlat
// kernel over the same DRAM-resident corpus at W concurrent workers, each
// scanning a disjoint slice — which is what Flat.queryShards does in
// production.
//
// Reading it: if the single-core scan is compute/issue-bound, aggregate
// throughput scales roughly linearly with W until the cores run out, and a
// faster kernel (wider SIMD) would raise every point on that line. If the scan
// is memory-bandwidth-bound, throughput plateaus well below linear and a faster
// kernel changes nothing — the cores are already waiting on DRAM.
func BenchmarkFlatScanScaling(b *testing.B) {
	for _, d := range []int{256, 768} {
		const n = 200_000
		corpus := makeUnitVectors(n, d, 0xfeed)
		query := makeUnitVectors(1, d, 0xdade)[0]
		for _, w := range []int{1, 2, 4, 6, 8} {
			if w > runtime.NumCPU() {
				continue
			}
			b.Run(fmt.Sprintf("d%d/workers%d", d, w), func(b *testing.B) {
				chunk := n / w
				sinks := make([]float64, w*16) // padded to avoid false sharing
				b.SetBytes(int64(n) * int64(d) * 4)
				b.ResetTimer()
				for range b.N {
					var wg sync.WaitGroup
					for t := range w {
						wg.Add(1)
						go func(t, lo, hi int) {
							defer wg.Done()
							s := 0.0
							scanFlat(query, corpus[lo:hi], func(_ int, v float64) { s += v })
							sinks[t*16] = s
						}(t, t*chunk, min((t+1)*chunk, n))
					}
					wg.Wait()
				}
				b.StopTimer()
				for _, s := range sinks {
					bwSink += uint64(s)
				}
			})
		}
	}
}

// BenchmarkFlatPrecisionLadder is the roofline's actual recommendation, measured.
//
// The flat scan sits ~10x on the memory side of the machine's ridge point, so
// the only lever that moves it is BYTES PER CANDIDATE, not multiply width:
// f32 (4 B/dim) -> int8 (1 B/dim) -> binary (1 bit/dim). All three already
// exist in this package. Same N, same dim, full end-to-end Query (sharding and
// top-k included) so the numbers are what a caller actually gets.
func BenchmarkFlatPrecisionLadder(b *testing.B) {
	for _, d := range []int{256, 768} {
		const n = 200_000
		corpus := makeUnitVectors(n, d, 0xfeed)
		query := makeUnitVectors(1, d, 0xdade)[0]
		f32 := New(corpus)
		i8 := NewFlatI8(corpus)
		bin := NewFlatBinary(corpus)
		for _, tc := range []struct {
			name string
			run  func()
		}{
			{"f32", func() { _ = f32.Query(query, 10) }},
			{"int8", func() { _ = i8.Query(query, 10) }},
			{"binary", func() { _ = bin.Query(query, 10) }},
		} {
			b.Run(fmt.Sprintf("d%d/%s", d, tc.name), func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for range b.N {
					tc.run()
				}
			})
		}
	}
}
