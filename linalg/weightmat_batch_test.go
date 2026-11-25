package linalg

import (
	"fmt"
	"math/rand"
	"slices"
	"testing"
)

func batchWeights(t *testing.T, kind string, rows []int, K int) []WeightMat {
	t.Helper()
	rng := rand.New(rand.NewSource(int64(len(kind)*1000 + K)))
	out := make([]WeightMat, len(rows))
	for i, N := range rows {
		w := make([]float32, N*K)
		for j := range w {
			w[j] = float32(rng.NormFloat64())
		}
		switch kind {
		case "w8a8":
			out[i] = QuantizeInt8(w, N, K, true)
		case "q8":
			out[i] = QuantizeInt8(w, N, K, false)
		case "int4":
			out[i] = QuantizeInt4(w, N, K, 32)
		case "int4+repacked":
			out[i] = QuantizeInt4(w, N, K, 32)
			out[i].RepackInt4Row4()
			out[i].RepackInt4SplitHalf()
		case "int4 repacked-only":
			c := QuantizeInt4(w, N, K, 32)
			q, s16, _, _ := c.Int4F16()
			var ok bool
			if out[i], ok = RepackInt4InPlace(slices.Clone(q), slices.Clone(s16), N, K, 32, PreferredInt4Layout(N, K, 32)); !ok {
				t.Fatalf("RepackInt4InPlace declined the preferred layout")
			}
		case "f32":
			out[i] = WrapF32(w, N, K)
		default:
			t.Fatalf("unknown kind %q", kind)
		}
	}
	return out
}

func batchRun(ws *Workspace, a []float32, M int, wms []*WeightMat) (batched, perOp [][]float32) {
	ops := make([]BatchOp, len(wms))
	for i, w := range wms {
		batched = append(batched, make([]float32, M*w.Rows()))
		perOp = append(perOp, make([]float32, M*w.Rows()))
		ops[i] = BatchOp{W: w, Dst: batched[i]}
	}
	MatmulBTBatchInto(ws, a, M, ops)
	var ws2 Workspace
	for i, w := range wms {
		w.MatmulBTInto(&ws2, a, perOp[i], M)
	}
	return batched, perOp
}

func ptrs(wms []WeightMat) []*WeightMat {
	out := make([]*WeightMat, len(wms))
	for i := range wms {
		out[i] = &wms[i]
	}
	return out
}

// TestMatmulBTBatchInto_matchesPerOp holds every batch, fused or not, bit-equal to MatmulBTInto once per op, serial and
// fanned out, at M = 1 and M > 1.
func TestMatmulBTBatchInto_matchesPerOp(t *testing.T) {
	const K = 256
	rows := []int{64, 32, 32} // q‖k‖v-shaped
	mixed := append(batchWeights(t, "w8a8", rows[:1], K), batchWeights(t, "int4", rows[1:], K)...)
	cases := map[string][]WeightMat{
		"w8a8":               batchWeights(t, "w8a8", rows, K),
		"int4":               batchWeights(t, "int4", rows, K),
		"int4+repacked":      batchWeights(t, "int4+repacked", rows, K),
		"int4 repacked-only": batchWeights(t, "int4 repacked-only", rows, K),
		"q8 (per op)":        batchWeights(t, "q8", rows, K),
		"f32 (per op)":       batchWeights(t, "f32", rows, K),
		"mixed (per op)":     mixed,
	}
	rng := rand.New(rand.NewSource(5))
	for name, wms := range cases {
		for _, M := range []int{1, 3, 5} {
			for _, thr := range []int{1 << 40, 0} { // serial, then always parallel
				a := make([]float32, M*K)
				for i := range a {
					a[i] = float32(rng.NormFloat64())
				}
				var ws Workspace
				ws.SetThreshold(thr)
				got, want := batchRun(&ws, a, M, ptrs(wms))
				for i := range got {
					if !slices.Equal(got[i], want[i]) {
						t.Errorf("%s M=%d thr=%d: op %d differs from MatmulBTInto", name, M, thr, i)
					}
				}
			}
		}
	}
}

// TestMatmulBTBatchInto_honoursActGroups: a weight-stamped activation group is used when the workspace sets none, and
// a batch whose weights carry different stamps runs per op, each with its own.
func TestMatmulBTBatchInto_honoursActGroups(t *testing.T) {
	const K, M = 256, 2
	rng := rand.New(rand.NewSource(9))
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	for _, kind := range []string{"w8a8", "int4"} {
		for _, stamps := range [][]int{{64, 64}, {64, 0}, {32, 128}} {
			wms := batchWeights(t, kind, []int{32, 32}, K)
			for i, g := range stamps {
				wms[i].SetActQuantGroup(g)
			}
			var ws Workspace
			got, want := batchRun(&ws, a, M, ptrs(wms))
			for i := range got {
				if !slices.Equal(got[i], want[i]) {
					t.Errorf("%s stamps %v: op %d differs from MatmulBTInto", kind, stamps, i)
				}
			}
		}
	}
}

func TestMatmulBTBatchInto_fusesOnlyWhatShares(t *testing.T) {
	const K = 256
	w8 := batchWeights(t, "w8a8", []int{16, 16}, K)
	w4 := batchWeights(t, "int4", []int{16, 16}, K)
	w4ro := batchWeights(t, "int4 repacked-only", []int{16, 16}, K)
	w4odd := batchWeights(t, "int4 repacked-only", []int{16}, K)
	if PreferredInt4Layout(6, K, 32) == LayoutSplitHalf { // split-half has no row constraint
		w4odd = append(w4odd, batchWeights(t, "int4 repacked-only", []int{6}, K)...)
	} else {
		w4odd = append(w4odd, batchWeights(t, "int4", []int{6}, K)...)
	}
	otherK := batchWeights(t, "w8a8", []int{16}, 2*K)
	g64 := batchWeights(t, "int4", []int{16}, K)
	g64[0] = QuantizeInt4(make([]float32, 16*K), 16, K, 64)
	stamped := batchWeights(t, "w8a8", []int{16}, K)
	stamped[0].SetActQuantGroup(64)
	// Without a repacked layout on this CPU (no DotProd, no AVX2, or a VNNI host) the "repacked-only" weights are
	// canonical, and canonical int4 fuses at any M.
	repackedWant := batchPerOp
	if PreferredInt4Layout(16, K, 32) == LayoutCanonical {
		repackedWant = batchW4A8
	}
	for _, c := range []struct {
		name string
		ops  []*WeightMat
		M    int
		want batchKindT
	}{
		{"all w8a8", ptrs(w8), 1, batchW8A8},
		{"all int4", ptrs(w4), 4, batchW4A8},
		{"repacked-only at M=1", ptrs(w4ro), 1, batchW4A8},
		{"repacked-only at M=2", ptrs(w4ro), 2, repackedWant},
		{"repacked-only beside rows%4 != 0", ptrs(w4odd), 1, repackedWant},
		{"w8a8 and int4", []*WeightMat{&w8[0], &w4[0]}, 1, batchPerOp},
		{"two K", []*WeightMat{&w8[0], &otherK[0]}, 1, batchPerOp},
		{"two int4 groups", []*WeightMat{&w4[0], &g64[0]}, 1, batchPerOp},
		{"two act-group stamps", []*WeightMat{&w8[0], &stamped[0]}, 1, batchPerOp},
	} {
		ops := make([]BatchOp, len(c.ops))
		for i, w := range c.ops {
			ops[i] = BatchOp{W: w}
		}
		if got := batchKind(ops, c.M); got != c.want {
			t.Errorf("%s: batchKind = %d, want %d", c.name, got, c.want)
		}
	}
}

func TestMatmulBTBatchInto_steadyStateAllocatesNothing(t *testing.T) {
	const K, M = 256, 1
	for _, kind := range []string{"w8a8", "int4"} {
		wms := batchWeights(t, kind, []int{64, 32, 32}, K)
		a := make([]float32, M*K)
		ops := make([]BatchOp, len(wms))
		for i := range wms {
			ops[i] = BatchOp{W: &wms[i], Dst: make([]float32, M*wms[i].Rows())}
		}
		var ws Workspace
		ws.SetThreshold(1 << 40) // serial: a fan-out's goroutines allocate by design
		MatmulBTBatchInto(&ws, a, M, ops)
		if n := testing.AllocsPerRun(20, func() { MatmulBTBatchInto(&ws, a, M, ops) }); n != 0 {
			t.Errorf("%s: %v allocations a call after warm-up, want 0", kind, n)
		}
		for i, op := range ws.batchW8 {
			if op.BQ != nil || op.Dst != nil {
				t.Errorf("%s: batchW8[%d] still holds a weight after the call", kind, i)
			}
		}
		for i, op := range ws.batchW4 {
			if op.W4 != nil || op.Row4 != nil || op.SplitHalf != nil || op.Dst != nil {
				t.Errorf("%s: batchW4[%d] still holds a weight after the call", kind, i)
			}
		}
	}
}

func TestMatmulBTBatchInto_emptyAndNil(t *testing.T) {
	MatmulBTBatchInto(nil, nil, 1, nil) // nothing to do
	defer func() {
		if r := recover(); r == nil || fmt.Sprint(r) != "linalg: MatmulBTBatchInto: ops[0].W is nil" {
			t.Errorf("a nil weight: recover() = %v", r)
		}
	}()
	MatmulBTBatchInto(nil, make([]float32, 4), 1, []BatchOp{{Dst: make([]float32, 4)}})
}
