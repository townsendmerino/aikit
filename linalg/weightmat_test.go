package linalg

import (
	"math/rand/v2"
	"testing"
)

// TestWeightMat_bitIdenticalToKernels: WeightMat is a wrapper, so each precision's
// MatmulBT/Row must be EXACTLY the raw linalg kernel call a consumer makes today —
// asserted bit-for-bit (==), since the dedup must not change any consumer's output.
func TestWeightMat_bitIdenticalToKernels(t *testing.T) {
	rng := rand.New(rand.NewPCG(17, 23))
	rv := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = rng.Float32()*2 - 1
		}
		return v
	}
	const M, K, N, group = 6, 64, 16, 32
	a := rv(M * K)
	wf := rv(N * K)
	eq := func(name string, got, want []float32) {
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s: idx %d got %v want %v (must be bit-identical)", name, i, got[i], want[i])
				return
			}
		}
	}

	// f32
	{
		wm := WrapF32(wf, N, K)
		got := make([]float32, M*N)
		wm.MatmulBT(a, got, M)
		want := make([]float32, M*N)
		MatmulBT(a, wf, want, M, K, N)
		eq("f32", got, want)
	}
	// int8 weight-only (Q8) — the decoder/encoder-storage path
	{
		wm := QuantizeInt8(wf, N, K, false)
		q, s := QuantizeRowsInt8(wf, N, K)
		got := make([]float32, M*N)
		wm.MatmulBT(a, got, M)
		want := make([]float32, M*N)
		MatmulBTQ8(a, q, s, want, M, K, N)
		eq("int8_q8", got, want)
		// accessors expose the same arrays for GPU export / a consumer's own matmul
		gq, gs, gw8a8, ok := wm.Int8()
		if !ok || gw8a8 || len(gq) != N*K || len(gs) != N {
			t.Fatalf("Int8() accessor: ok=%v w8a8=%v len(q)=%d len(s)=%d", ok, gw8a8, len(gq), len(gs))
		}
		eq("int8_scales", gs, s)
	}
	// int8 W8A8 — the vision path + decoder quantInt8I8
	{
		wm := QuantizeInt8(wf, N, K, true)
		q, s := QuantizeRowsInt8(wf, N, K)
		got := make([]float32, M*N)
		wm.MatmulBT(a, got, M)
		want := make([]float32, M*N)
		MatmulBTW8A8(a, q, s, want, M, K, N)
		eq("w8a8", got, want)
		if _, _, w8a8, _ := wm.Int8(); !w8a8 {
			t.Fatal("Int8() should report w8a8=true")
		}
	}
	// int4 group-wise (W4A8) — the decoder int4 path
	{
		wm := QuantizeInt4(wf, N, K, group)
		q4, q4s := QuantizeGroupsInt4(wf, N, K, group)
		got := make([]float32, M*N)
		wm.MatmulBT(a, got, M)
		want := make([]float32, M*N)
		MatmulBTW4A8(a, q4, q4s, want, M, K, N, group)
		eq("int4", got, want)
	}
	// Row dequant (embedRow) — int8 + int4 + f32
	{
		wm8 := QuantizeInt8(wf, N, K, false)
		q, s := QuantizeRowsInt8(wf, N, K)
		got := make([]float32, K)
		wm8.Row(3, got)
		want := make([]float32, K)
		DequantizeRowInt8(q[3*K:4*K], s[3], want)
		eq("row_int8", got, want)

		wmf := WrapF32(wf, N, K)
		wmf.Row(3, got)
		eq("row_f32", got, wf[3*K:4*K])
	}
}

// TestWeightMat_wrapConstructors: WrapInt8/WrapInt4 must be the exact inverse of
// Int8()/Int4() — they wrap caller-owned pre-quantized arrays WITHOUT copying or
// re-quantizing (the goinfer .giw zero-copy mmap-alias path). Asserted three ways:
// (1) a wrapped weight matmuls/dequantizes bit-identically to the same arrays run
// through the Quantize* constructor; (2) the accessors hand back the SAME backing
// slice (aliased, not copied); (3) shape mismatches panic.
func TestWeightMat_wrapConstructors(t *testing.T) {
	rng := rand.New(rand.NewPCG(41, 7))
	rv := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = rng.Float32()*2 - 1
		}
		return v
	}
	const M, K, N, group = 5, 64, 16, 32
	a := rv(M * K)
	wf := rv(N * K)
	eq := func(name string, got, want []float32) {
		t.Helper()
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s: idx %d got %v want %v (wrap must be bit-identical)", name, i, got[i], want[i])
			}
		}
	}

	// int8 weight-only: WrapInt8 of the quantizer's output == QuantizeInt8.
	{
		q, s := QuantizeRowsInt8(wf, N, K)
		wm := WrapInt8(q, s, N, K, false)
		ref := QuantizeInt8(wf, N, K, false)
		got, want := make([]float32, M*N), make([]float32, M*N)
		wm.MatmulBT(a, got, M)
		ref.MatmulBT(a, want, M)
		eq("wrapint8_matmul", got, want)
		// accessor aliases the wrapped slices (zero-copy), and round-trips w8a8.
		gq, gs, w8a8, ok := wm.Int8()
		if !ok || w8a8 || &gq[0] != &q[0] || &gs[0] != &s[0] {
			t.Fatalf("WrapInt8 accessor: ok=%v w8a8=%v aliased(q)=%v aliased(s)=%v", ok, w8a8, &gq[0] == &q[0], &gs[0] == &s[0])
		}
		wm8a8 := WrapInt8(q, s, N, K, true)
		if _, _, w8a8q, _ := wm8a8.Int8(); !w8a8q {
			t.Fatal("WrapInt8 w8a8=true not reported")
		}
	}
	// int4 group-wise: WrapInt4 of the quantizer's output == QuantizeInt4.
	{
		q4, q4s := QuantizeGroupsInt4(wf, N, K, group)
		wm := WrapInt4(q4, q4s, N, K, group)
		ref := QuantizeInt4(wf, N, K, group)
		got, want := make([]float32, M*N), make([]float32, M*N)
		wm.MatmulBT(a, got, M)
		ref.MatmulBT(a, want, M)
		eq("wrapint4_matmul", got, want)
		// Row dequant matches, and the accessor aliases (zero-copy).
		rGot, rWant := make([]float32, K), make([]float32, K)
		wm.Row(2, rGot)
		ref.Row(2, rWant)
		eq("wrapint4_row", rGot, rWant)
		gq4, gq4s, ggroup, ok := wm.Int4()
		if !ok || ggroup != group || &gq4[0] != &q4[0] || &gq4s[0] != &q4s[0] {
			t.Fatalf("WrapInt4 accessor: ok=%v group=%d aliased(q4)=%v aliased(q4s)=%v", ok, ggroup, &gq4[0] == &q4[0], &gq4s[0] == &q4s[0])
		}
	}
	// shape guards panic.
	mustPanic := func(name string, f func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Fatalf("%s: expected panic", name)
			}
		}()
		f()
	}
	q, s := QuantizeRowsInt8(wf, N, K)
	mustPanic("int8_bad_q", func() { WrapInt8(q[:N*K-1], s, N, K, false) })
	mustPanic("int8_bad_scales", func() { WrapInt8(q, s[:N-1], N, K, false) })
	q4, q4s := QuantizeGroupsInt4(wf, N, K, group)
	mustPanic("int4_bad_group", func() { WrapInt4(q4, q4s, N, K, 0) })
	mustPanic("int4_bad_q4", func() { WrapInt4(q4[:len(q4)-1], q4s, N, K, group) })
}
