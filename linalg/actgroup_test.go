package linalg

import (
	"math"
	"math/rand/v2"
	"testing"
)

func withActGroup(t *testing.T, g int) {
	t.Helper()
	prev := ActQuantGroup()
	SetActQuantGroup(g)
	t.Cleanup(func() { SetActQuantGroup(prev) })
}

func agRandMat(r *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(r.NormFloat64())
	}
	return v
}

// relErr is ||got-want|| / ||want||.
func agRelErr(got, want []float32) float64 {
	var num, den float64
	for i := range want {
		d := float64(got[i]) - float64(want[i])
		num += d * d
		den += float64(want[i]) * float64(want[i])
	}
	return math.Sqrt(num / den)
}

// TestActGroup_wholeRowGroupIsThePerRowKernel: one activation group spanning the whole row is the
// per-row scheme, so the reference must reproduce the production kernels. W8A8 is bit-identical
// (same int32 dot, same (dot·aS)·bS order); W4A8 moves the activation scale inside the per-group
// sum, so it agrees to float rounding only.
func TestActGroup_wholeRowGroupIsThePerRowKernel(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	const M, K, N = 3, 256, 40
	a := agRandMat(r, M*K)
	w := agRandMat(r, N*K)

	w8 := QuantizeInt8(w, N, K, true)
	q8, s8, _, _ := w8.Int8()
	want8 := make([]float32, M*N)
	MatmulBTW8A8Into(new(Workspace), a, q8, s8, want8, M, K, N)
	w4 := QuantizeInt4(w, N, K, 32)
	q4, s4, _, _ := w4.Int4()
	want4 := make([]float32, M*N)
	MatmulBTW4A8Into(new(Workspace), a, q4, s4, want4, M, K, N, 32)

	withActGroup(t, K)
	got8 := make([]float32, M*N)
	MatmulBTW8A8Into(new(Workspace), a, q8, s8, got8, M, K, N)
	for i := range want8 {
		if got8[i] != want8[i] {
			t.Fatalf("W8A8 [%d]: reference %v != kernel %v (want bit-identical)", i, got8[i], want8[i])
		}
	}
	got4 := make([]float32, M*N)
	MatmulBTW4A8Into(new(Workspace), a, q4, s4, got4, M, K, N, 32)
	if e := agRelErr(got4, want4); e > 1e-6 {
		t.Errorf("W4A8 whole-row reference vs kernel: rel err %.3g, want float-rounding level", e)
	}
}

// TestActGroup_layoutsAgree: every W4A8 entry point under per-32 activations — the canonical free
// function, the Batch entry over the split-half layout, and the WeightMat method — agrees with the Go
// reference to accumulation order, at M=3 (the per-row loop over the M=1 kernels, and the canonical
// kernel's M>1 path).
func TestActGroup_layoutsAgree(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	const M, K, N = 3, 128, 24
	a := agRandMat(r, M*K)
	a[40] = 300
	wm := QuantizeInt4(agRandMat(r, N*K), N, K, 32)
	q4, s4, _, _ := wm.Int4()
	withActGroup(t, 32)

	ref := make([]float32, M*N)
	matmulW4A8GroupedRef(new(Workspace), 32, a, int4Layout{w4: q4, wS: s4, group: 32, K: K}, ref, M, N)
	canon := make([]float32, M*N)
	MatmulBTW4A8Into(new(Workspace), a, q4, s4, canon, M, K, N, 32)
	sh := RepackW4A8SplitHalf(q4, N, K, 32)
	batch := make([]float32, M*N)
	MatmulBTW4A8Batch(new(Workspace), a, M, K, 32, []W4A8Op{{SplitHalf: sh, Scales: s4, Dst: batch, N: N}})
	method := make([]float32, M*N)
	wm.MatmulBTW4A8Into(new(Workspace), a, method, M)
	for name, got := range map[string][]float32{"canonical": canon, "batch split-half": batch, "method": method} {
		if e := agRelErr(got, ref); e > 1e-6 {
			t.Errorf("%s vs reference: rel err %.3g", name, e)
		}
	}
}

// TestActGroup_outlierStaysInItsGroup is the reason the change exists (goinfer H2): one massive
// outlier in an activation row. Under a per-row scale it rounds every other element to zero; under
// per-32 scales only its own group loses precision. Measured against the f32 product of the same
// dequantized weights, so only activation quantization differs.
func TestActGroup_outlierStaysInItsGroup(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	const M, K, N = 1, 3072, 64
	a := agRandMat(r, M*K)
	a[700] = 500 // ~max/rms 500, the Phi-3 regime
	wm := QuantizeInt8(agRandMat(r, N*K), N, K, true)
	q8, s8, _, _ := wm.Int8()
	ref := make([]float32, M*N) // f32 activations × the same int8 weights
	MatmulBTQ8(a, q8, s8, ref, M, K, N)
	// The outlier's own contribution is exact under any activation scale (it sets the scale), so
	// compare on the product WITHOUT it — what the other 3071 inputs contribute.
	for n := range N {
		ref[n] -= a[700] * float32(q8[n*K+700]) * s8[n]
	}
	sub := func(out []float32) []float32 {
		o := append([]float32(nil), out...)
		for n := range N {
			o[n] -= a[700] * float32(q8[n*K+700]) * s8[n]
		}
		return o
	}
	perRow := make([]float32, M*N)
	MatmulBTW8A8Into(new(Workspace), a, q8, s8, perRow, M, K, N)
	withActGroup(t, 32)
	grouped := make([]float32, M*N)
	MatmulBTW8A8Into(new(Workspace), a, q8, s8, grouped, M, K, N)
	eRow, eGrp := agRelErr(sub(perRow), ref), agRelErr(sub(grouped), ref)
	t.Logf("rel err without the outlier's own term: per-row %.3g, per-32 %.3g", eRow, eGrp)
	if eRow < 0.5 {
		t.Errorf("premise: per-row error %.3g should be large (the outlier flushes the row)", eRow)
	}
	// The outlier's own group still loses its other 31 elements to rounding: ~sqrt(31/3071) ≈ 0.10 of
	// the remaining signal. Everything outside that group survives, so the bound is that group's cost
	// plus int8 noise, and at least 8x below per-row.
	if eGrp > 0.15 || eGrp*8 > eRow {
		t.Errorf("per-32 error %.3g, want < 0.15 and at least 8x below per-row (%.3g)", eGrp, eRow)
	}
}

// TestActGroup_offIsInert: with the group at 0 the entry points are the production kernels,
// bit-for-bit.
func TestActGroup_offIsInert(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	const M, K, N = 2, 96, 16
	a := agRandMat(r, M*K)
	wm := QuantizeInt8(agRandMat(r, N*K), N, K, true)
	q8, s8, _, _ := wm.Int8()
	before := make([]float32, M*N)
	MatmulBTW8A8Into(new(Workspace), a, q8, s8, before, M, K, N)
	SetActQuantGroup(32)
	SetActQuantGroup(0)
	after := make([]float32, M*N)
	MatmulBTW8A8Into(new(Workspace), a, q8, s8, after, M, K, N)
	for i := range before {
		if before[i] != after[i] {
			t.Fatalf("[%d] %v != %v after toggling the group off", i, after[i], before[i])
		}
	}
}

// TestActGroup_perWorkspace: a workspace's own group applies to matmuls run through it and only to
// them, with the process-wide setting untouched: goinfer runs two models in one process with
// different choices by giving each its own workspace setting.
func TestActGroup_perWorkspace(t *testing.T) {
	r := rand.New(rand.NewPCG(25, 26))
	const M, K, N = 1, 256, 32
	a := agRandMat(r, M*K)
	a[10] = 400
	wm := QuantizeInt8(agRandMat(r, N*K), N, K, true)
	q8, s8, _, _ := wm.Int8()
	if ActQuantGroup() != 0 {
		t.Fatal("process-wide group must start at 0")
	}
	perRow := make([]float32, N)
	MatmulBTW8A8Into(new(Workspace), a, q8, s8, perRow, M, K, N)
	var grouped Workspace
	grouped.SetActQuantGroup(32)
	viaWS := make([]float32, N)
	MatmulBTW8A8Into(&grouped, a, q8, s8, viaWS, M, K, N)
	ref := make([]float32, N)
	matmulW8A8GroupedRef(new(Workspace), 32, a, q8, s8, ref, M, K, N)
	if e := agRelErr(viaWS, ref); e > 1e-6 {
		t.Errorf("workspace group 32 vs reference: rel err %.3g", e)
	}
	again := make([]float32, N)
	MatmulBTW8A8Into(new(Workspace), a, q8, s8, again, M, K, N)
	for i := range perRow {
		if again[i] != perRow[i] {
			t.Fatalf("[%d] a default workspace changed after another used group 32: %v != %v", i, again[i], perRow[i])
		}
	}
	// Premise: the default workspace really did run per-row — its output is not the per-32 result.
	// (A small relative gap: the outlier's own, exactly-quantized term dominates every output.)
	if agRelErr(perRow, ref) < 1e-4 {
		t.Errorf("premise: the default workspace's output matches the per-32 reference (rel err %.3g)", agRelErr(perRow, ref))
	}
}
