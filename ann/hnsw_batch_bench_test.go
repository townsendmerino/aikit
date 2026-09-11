package ann

import (
	"math"
	"math/rand"
	"testing"
)

// BenchmarkHNSWQueryBatched is the arbiter for item 15. The doc's prototype
// measured 1.40× at ef=64 and 1.36× at ef=200 on n=50k, d=256.
func BenchmarkHNSWQueryBatched(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	const n, d = 50_000, 256
	vecs := make([][]float32, n)
	for i := range vecs {
		v := make([]float32, d)
		var norm float64
		for j := range v {
			v[j] = float32(rng.NormFloat64())
			norm += float64(v[j]) * float64(v[j])
		}
		inv := float32(1 / math.Sqrt(norm))
		for j := range v {
			v[j] *= inv
		}
		vecs[i] = v
	}
	h := BuildHNSW(vecs, Config{})
	q := vecs[0]
	for _, ef := range []int{64, 200} {
		b.Run("ef"+itoaAnn(ef), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				sinkHits = h.QueryEf(q, 10, ef)
			}
		})
	}
}

// BenchmarkHNSWQueryBatchedInt8 is BenchmarkHNSWQueryBatched's int8 twin
// (audit M-21): scoreInto's int8 branch went from a scalar DotI8-per-candidate
// loop to DotI8x8, the same 8-row batching item 15 gave the f32 path. Same
// shape (n, d, ef) as the f32 benchmark, so the two are directly comparable.
func BenchmarkHNSWQueryBatchedInt8(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	const n, d = 50_000, 256
	vecs := make([][]float32, n)
	for i := range vecs {
		v := make([]float32, d)
		var norm float64
		for j := range v {
			v[j] = float32(rng.NormFloat64())
			norm += float64(v[j]) * float64(v[j])
		}
		inv := float32(1 / math.Sqrt(norm))
		for j := range v {
			v[j] *= inv
		}
		vecs[i] = v
	}
	h := NewHNSW(Config{Int8: true})
	for _, v := range vecs {
		h.Add(v)
	}
	q := vecs[0]
	for _, ef := range []int{64, 200} {
		b.Run("ef"+itoaAnn(ef), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				sinkHits = h.QueryEf(q, 10, ef)
			}
		})
	}
}

// BenchmarkHNSWQueryInt8_batchedVsUnbatched isolates the DotI8x8 win from
// graph-traversal noise: same index, same query, h.scoreUnbatched toggled —
// the only thing that changes is whether scoreInto's int8 branch batches.
func BenchmarkHNSWQueryInt8_batchedVsUnbatched(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	const n, d = 50_000, 256
	vecs := make([][]float32, n)
	for i := range vecs {
		v := make([]float32, d)
		var norm float64
		for j := range v {
			v[j] = float32(rng.NormFloat64())
			norm += float64(v[j]) * float64(v[j])
		}
		inv := float32(1 / math.Sqrt(norm))
		for j := range v {
			v[j] *= inv
		}
		vecs[i] = v
	}
	h := NewHNSW(Config{Int8: true})
	for _, v := range vecs {
		h.Add(v)
	}
	q := vecs[0]
	for _, ef := range []int{64, 200} {
		for _, unbatched := range []bool{true, false} {
			name := "ef" + itoaAnn(ef) + "/batched"
			if unbatched {
				name = "ef" + itoaAnn(ef) + "/unbatched"
			}
			b.Run(name, func(b *testing.B) {
				h.scoreUnbatched = unbatched
				b.ReportAllocs()
				for b.Loop() {
					sinkHits = h.QueryEf(q, 10, ef)
				}
				h.scoreUnbatched = false
			})
		}
	}
}
