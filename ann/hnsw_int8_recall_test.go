package ann_test

import (
	"testing"

	"github.com/townsendmerino/aikit/ann"
	"github.com/townsendmerino/aikit/linalg"
)

// dequantize round-trips vectors through int8: quantize to int8 + per-vector
// scales, then dequantize back to float32. The result holds exactly the values an
// int8 index sees, because dequant(a)·dequant(b) = scale[a]·scale[b]·(int8 dot) —
// same ranking as a real int8 dot. So building/searching an f32 HNSW on these
// faithfully measures an int8 HNSW's recall, with no core changes or int8 kernels.
func dequantize(vecs [][]float32) [][]float32 {
	n := len(vecs)
	if n == 0 {
		return nil
	}
	dim := len(vecs[0])
	flat := make([]float32, n*dim)
	for i, v := range vecs {
		copy(flat[i*dim:i*dim+dim], v)
	}
	bq, scales := linalg.QuantizeRowsInt8(flat, n, dim)
	out := make([][]float32, n)
	for i := range out {
		row := make([]float32, dim)
		linalg.DequantizeRowInt8(bq[i*dim:i*dim+dim], scales[i], row)
		out[i] = row
	}
	return out
}

func recallAtK(got []ann.Hit, truth map[int]bool, k int) float64 {
	hit := 0
	for _, h := range got {
		if truth[h.Index] {
			hit++
		}
	}
	return float64(hit) / float64(k)
}

// TestHNSW_int8RecallGate is the §3.3 gate the roadmap demands: does building AND
// searching an HNSW at int8 precision hold recall? It compares an f32 HNSW to one
// built on int8-quantized vectors (queries quantized too, W8A8-faithful), both vs
// the exact float32 Flat top-k, on real Model2Vec embeddings. Only if int8 stays
// within tolerance of f32 here is productionizing an int8 HNSW (¼-memory graph)
// worth the engineering.
func TestHNSW_int8RecallGate(t *testing.T) {
	m, vecs, queries := realCorpus(t)
	cfg := ann.Config{M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1}

	f32h := ann.BuildHNSW(vecs, cfg)
	i8h := ann.BuildHNSW(dequantize(vecs), cfg)
	truth := ann.New(vecs) // exact float32 reference

	const k = 10
	var f32sum, i8sum float64
	for _, qt := range queries {
		q := m.Encode(qt)
		set := make(map[int]bool, k)
		for _, h := range truth.Query(q, k) {
			set[h.Index] = true
		}
		f32sum += recallAtK(f32h.Query(q, k), set, k)
		qd := dequantize([][]float32{q})[0] // quantize the query too
		i8sum += recallAtK(i8h.Query(qd, k), set, k)
	}
	f32r := f32sum / float64(len(queries))
	i8r := i8sum / float64(len(queries))
	t.Logf("HNSW recall@%d on %d real embeddings: f32 %.4f, int8 %.4f (Δ %.4f)", k, len(vecs), f32r, i8r, f32r-i8r)

	if f32r-i8r > 0.03 {
		t.Errorf("int8 HNSW recall %.4f is >0.03 below f32 %.4f — gate fails, int8 HNSW not worth productionizing as-is", i8r, f32r)
	}
	if i8r < 0.90 {
		t.Errorf("int8 HNSW recall %.4f below 0.90 floor", i8r)
	}
}
