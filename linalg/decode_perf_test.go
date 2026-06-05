package linalg

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// Qwen2.5-0.5B per-layer projection shapes (hidden 896, GQA 14:2 heads × 64,
// intermediate 4864), the M=1 single-token decode path goinfer's profile is
// dominated by. q/k/v share one activation (the input-norm output); gate/up
// share another (the post-attn-norm output); o and down have their own.
const qHidden = 896

var qProj = []struct {
	name string
	K, N int
}{
	{"q_proj", 896, 896}, {"k_proj", 896, 128}, {"v_proj", 896, 128},
	{"o_proj", 896, 896}, {"gate", 896, 4864}, {"up", 896, 4864}, {"down", 4864, 896},
}

func randI8(n int) []int8 {
	r := rand.New(rand.NewSource(int64(n) + 1))
	v := make([]int8, n)
	for i := range v {
		v[i] = int8(r.Intn(255) - 127)
	}
	return v
}

func randF(n int) []float32 {
	r := rand.New(rand.NewSource(int64(n) + 7))
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(r.NormFloat64())
	}
	return v
}

type proj struct {
	bq   []int8
	bs   []float32
	dst  []float32
	K, N int
}

func buildLayer(M int) (acts map[string][]float32, p map[string]*proj) {
	acts = map[string][]float32{
		"attn": randF(M * qHidden), // q/k/v + o input
		"mlp":  randF(M * qHidden), // gate/up input
		"down": randF(M * 4864),    // down input
	}
	p = map[string]*proj{}
	for _, l := range qProj {
		p[l.name] = &proj{bq: randI8(l.N * l.K), bs: randF(l.N), dst: make([]float32, M*l.N), K: l.K, N: l.N}
	}
	return acts, p
}

// runLayerInto runs the 7 projections one-per-call (MatmulBTW8A8Into).
func runLayerInto(ws *Workspace, acts map[string][]float32, p map[string]*proj, M int) {
	in := map[string]string{"q_proj": "attn", "k_proj": "attn", "v_proj": "attn", "o_proj": "attn", "gate": "mlp", "up": "mlp", "down": "down"}
	for name, src := range in {
		o := p[name]
		MatmulBTW8A8Into(ws, acts[src], o.bq, o.bs, o.dst, M, o.K, o.N)
	}
}

// runLayerBatch is the goinfer-style fused dispatch: qkv in one parallel
// region, gate+up in one, o and down on their own — 4 dispatches, not 7.
func runLayerBatch(ws *Workspace, acts map[string][]float32, p map[string]*proj, M int) {
	opOf := func(name string) W8A8Op { o := p[name]; return W8A8Op{BQ: o.bq, Scales: o.bs, Dst: o.dst, N: o.N} }
	MatmulBTW8A8Batch(ws, acts["attn"], M, qHidden, []W8A8Op{opOf("q_proj"), opOf("k_proj"), opOf("v_proj")})
	MatmulBTW8A8Batch(ws, acts["mlp"], M, qHidden, []W8A8Op{opOf("gate"), opOf("up")})
	o := p["o_proj"]
	MatmulBTW8A8Into(ws, acts["attn"], o.bq, o.bs, o.dst, M, o.K, o.N)
	d := p["down"]
	MatmulBTW8A8Into(ws, acts["down"], d.bq, d.bs, d.dst, M, d.K, d.N)
}

// withThreshold runs fn with parThreshold set to force serial or parallel.
func withThreshold(serial bool, fn func()) {
	old := parThreshold
	if serial {
		parThreshold = math.MaxInt
	} else {
		parThreshold = 0
	}
	defer func() { parThreshold = old }()
	fn()
}

// BenchmarkDecodePool compares the spawn-per-call fan-out vs the persistent
// spin-then-park pool on the goinfer-style batched decode layer (M=1, forced
// parallel — the regime goinfer runs decode in). The pool keeps workers hot
// across the layer's ~4 dispatches instead of spawning + parking each. NOTE:
// this back-to-back microbench can't reproduce the between-token cooling of a
// real decode loop, so goinfer's end-to-end sweep is the arbiter (it under-
// shows here, per the task doc).
func BenchmarkDecodePool(b *testing.B) {
	acts, p := buildLayer(1)
	for _, workers := range []int{0, 2, 6, 8} { // 0 = spawn-per-call (no pool)
		name := fmt.Sprintf("workers=%d", workers)
		if workers == 0 {
			name = "spawn"
		}
		b.Run(name, func(b *testing.B) {
			ws := &Workspace{}
			if workers > 0 {
				ws.SetWorkers(workers)
				defer ws.Close()
			}
			withThreshold(false, func() { // force parallel
				runLayerBatch(ws, acts, p, 1)
				b.ReportAllocs()
				b.ResetTimer()
				for range b.N {
					runLayerBatch(ws, acts, p, 1)
				}
			})
		})
	}
}

// BenchmarkLayer covers the matrix: {Into, Batch} × {serial, parallel} × {M=1
// decode, M=64 prefill}. Decode should favor serial (fork/join > work);
// prefill should favor parallel. All paths share one Workspace → ~0 allocs.
func BenchmarkLayer(b *testing.B) {
	for _, M := range []int{1, 64} {
		acts, p := buildLayer(M)
		ws := &Workspace{}
		for _, path := range []struct {
			name string
			run  func()
		}{
			{"Into", func() { runLayerInto(ws, acts, p, M) }},
			{"Batch", func() { runLayerBatch(ws, acts, p, M) }},
		} {
			for _, serial := range []bool{true, false} {
				mode := "parallel"
				if serial {
					mode = "serial"
				}
				b.Run(fmt.Sprintf("M=%d/%s/%s", M, path.name, mode), func(b *testing.B) {
					withThreshold(serial, func() {
						path.run() // warm the workspace
						b.ReportAllocs()
						b.ResetTimer()
						for range b.N {
							path.run()
						}
					})
				})
			}
		}
	}
}
