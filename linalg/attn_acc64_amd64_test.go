//go:build amd64

package linalg

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestAccAcc64AVX_isLoadBearing is the first thing to check about a new kernel:
// that it RUNS. Both dispatchers return 0 on several decline paths (no AVX2,
// no keys, K%4 != 0, too few blocks), and a kernel that silently declined
// would leave every acc64 test passing against the pure-Go path while proving
// nothing about the assembly.
func TestAccAcc64AVX_isLoadBearing(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2 on this CPU — the kernels decline by design")
	}
	const nKeys, hd = 6, 64
	rng := rand.New(rand.NewPCG(41, 43))
	rf := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	srow := rf(nKeys)
	vals := rf(nKeys * hd)
	drow := make([]float32, hd)
	if got := avAcc64Blocks(srow, vals, drow, nKeys, hd, 0, hd); got != 64 {
		t.Errorf("avAcc64Blocks covered %d dims, want 64 — the AV kernel did not engage", got)
	}

	const K, N = 8, 12
	q := rf(K)
	b := rf(N * K)
	out := make([]float32, N)
	if got := qkAcc64Keys(q, b, out, K, N, 0, K); got != 12 {
		t.Errorf("qkAcc64Keys covered %d keys, want 12 — the QK kernel did not engage", got)
	}
}

// TestAvAcc64AVX32_matchesGo pins the AV kernel against the Go block it
// replaces, EXACTLY. The products are exact in f64 (both operands come from
// float32, so the 48-bit product fits float64's 53), which is what lets the
// kernel use FMA where the Go reference does a separate multiply and add — so
// equality, not a tolerance, is the right assertion.
func TestAvAcc64AVX32_matchesGo(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2 on this CPU")
	}
	rng := rand.New(rand.NewPCG(7, 11))
	for _, nKeys := range []int{1, 2, 7, 33} {
		const hd = 32
		srow := make([]float32, nKeys)
		for i := range srow {
			srow[i] = float32(rng.NormFloat64())
		}
		vals := make([]float32, nKeys*hd)
		for i := range vals {
			vals[i] = float32(rng.NormFloat64())
		}
		got := make([]float32, hd)
		avAcc64AVX32(&srow[0], &vals[0], nKeys, hd*4, &got[0])

		want := make([]float32, hd)
		for d := range hd {
			var acc float64
			for s := range nKeys { // key-ascending, one accumulator per output dim
				acc += float64(srow[s]) * float64(vals[s*hd+d])
			}
			want[d] = float32(acc)
		}
		for d := range hd {
			if math.Float32bits(got[d]) != math.Float32bits(want[d]) {
				t.Fatalf("nKeys=%d d=%d: kernel %v (%08x) != Go %v (%08x)",
					nKeys, d, got[d], math.Float32bits(got[d]), want[d], math.Float32bits(want[d]))
			}
		}
	}
}

// TestQkAcc64AVX4_matchesGo is the QK sibling: one key per f64 lane, dims
// ascending, compared bit-for-bit against the scalar f64 dot.
func TestQkAcc64AVX4_matchesGo(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2 on this CPU")
	}
	rng := rand.New(rand.NewPCG(13, 17))
	for _, K := range []int{4, 8, 64, 128} {
		for _, nBlocks := range []int{1, 3} {
			N := nBlocks * 4
			q := make([]float32, K)
			for i := range q {
				q[i] = float32(rng.NormFloat64())
			}
			rows := make([]float32, N*K)
			for i := range rows {
				rows[i] = float32(rng.NormFloat64())
			}
			got := make([]float32, N)
			qkAcc64AVX4(&q[0], &rows[0], K*4, K/4, nBlocks, &got[0])

			for n := range N {
				var acc float64
				for d := range K { // dim-ascending, one accumulator per key
					acc += float64(q[d]) * float64(rows[n*K+d])
				}
				want := float32(acc)
				if math.Float32bits(got[n]) != math.Float32bits(want) {
					t.Fatalf("K=%d N=%d key=%d: kernel %v (%08x) != Go %v (%08x)",
						K, N, n, got[n], math.Float32bits(got[n]), want, math.Float32bits(want))
				}
			}
		}
	}
}

// TestQkAcc64AVX16_matchesAVX4AndGo pins the 16-key kernel EXACTLY against the 4-key kernel it takes over from and against the Go
// reference: four 4-key blocks interleaved as four independent accumulators, each key still one f64 lane with its K dims folded in
// ascending d, so every key's sum is today's sum bit for bit. Shapes cover K at the production head dims and a key stride wider than
// K (a head slice of a kvDim-wide row).
func TestQkAcc64AVX16_matchesAVX4AndGo(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2 on this CPU")
	}
	rng := rand.New(rand.NewPCG(19, 23))
	for _, K := range []int{4, 64, 96, 128, 256} {
		for _, n16 := range []int{1, 2, 5} {
			for _, stride := range []int{K, 3*K + 4} {
				N := n16 * 16
				q := make([]float32, K)
				for i := range q {
					q[i] = float32(rng.NormFloat64())
				}
				rows := make([]float32, (N-1)*stride+K)
				for i := range rows {
					rows[i] = float32(rng.NormFloat64())
				}
				got, ref := make([]float32, N), make([]float32, N)
				qkAcc64AVX16(&q[0], &rows[0], stride*4, K/4, n16, &got[0])
				qkAcc64AVX4(&q[0], &rows[0], stride*4, K/4, n16*4, &ref[0])
				for n := range N {
					var acc float64
					for d := range K {
						acc += float64(q[d]) * float64(rows[n*stride+d])
					}
					want := float32(acc)
					if math.Float32bits(got[n]) != math.Float32bits(want) || math.Float32bits(got[n]) != math.Float32bits(ref[n]) {
						t.Fatalf("K=%d N=%d stride=%d key=%d: AVX16 %08x, AVX4 %08x, Go %08x", K, N, stride, n,
							math.Float32bits(got[n]), math.Float32bits(ref[n]), math.Float32bits(want))
					}
				}
			}
		}
	}
}

// TestQkAcc64Keys_takesAVX16 checks through the dispatcher itself (qkAcc64KeysSplit, whose two counts qkAcc64Keys sums) that the
// 16-key kernel takes the whole 16-key groups and the 4-key kernel the rest: N = 16·3 + 4·2 + 3 keys come back as 48 + 8 written, at a
// non-zero head offset, each score equal to the Go loop's, and the N%4 tail left as it was.
func TestQkAcc64Keys_takesAVX16(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2 on this CPU")
	}
	rng := rand.New(rand.NewPCG(29, 31))
	const K, N, stride, bOff = 96, 16*3 + 4*2 + 3, 96*4 + 16, 96 + 8
	q := make([]float32, K)
	for i := range q {
		q[i] = float32(rng.NormFloat64())
	}
	rows := make([]float32, bOff+(N-1)*stride+K)
	for i := range rows {
		rows[i] = float32(rng.NormFloat64())
	}
	out := make([]float32, N)
	sentinel := float32(math.NaN())
	for i := range out {
		out[i] = sentinel
	}
	n16, n4 := qkAcc64KeysSplit(q, rows, out, K, N, bOff, stride)
	if n16 != 48 || n4 != 8 {
		t.Fatalf("the dispatch wrote %d keys by the 16-key kernel and %d by the 4-key one, want 48 and 8", n16, n4)
	}
	for n := range N - 3 {
		var acc float64
		for d := range K {
			acc += float64(q[d]) * float64(rows[bOff+n*stride+d])
		}
		if math.Float32bits(out[n]) != math.Float32bits(float32(acc)) {
			t.Fatalf("key %d: %08x != Go %08x", n, math.Float32bits(out[n]), math.Float32bits(float32(acc)))
		}
	}
	for n := N - 3; n < N; n++ {
		if !math.IsNaN(float64(out[n])) {
			t.Errorf("tail key %d was written (%v): the kernels must leave the N%%4 tail to the Go loop", n, out[n])
		}
	}
}

// TestAcc64F16AVX_isLoadBearing checks the f16 kernels RUN on an AVX2+F16C host: a declining dispatcher would leave the f16 tests
// passing against the Go loops while proving nothing about the assembly.
func TestAcc64F16AVX_isLoadBearing(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("no AVX2+F16C on this CPU: the kernels decline by design")
	}
	rng := rand.New(rand.NewPCG(53, 59))
	const K, N, stride = 96, 16*3 + 5, 96 * 2
	h, w := f16Cache(rng, (N-1)*stride+K)
	q, scores := make([]float32, K), make([]float32, N)
	for i := range q {
		q[i] = float32(rng.NormFloat64())
	}
	for i := range scores {
		scores[i] = rng.Float32() / N
	}
	got, want := make([]float32, N), make([]float32, N)
	if n16, n4 := qkAcc64KeysSplitF16(q, h, got, K, N, 0, stride); n16 != 48 || n4 != 4 {
		t.Errorf("the f16 QK dispatch wrote %d keys by the 16-key kernel and %d by the 4-key one, want 48 and 4", n16, n4)
	}
	qkAcc64KeysSplit(q, w, want, K, N, 0, stride)
	for n := range 52 {
		if got[n] == 0 || math.Float32bits(got[n]) != math.Float32bits(want[n]) {
			t.Fatalf("QK key %d: f16 kernel %v, f32 kernel on the widened cache %v (want equal and non-zero)", n, got[n], want[n])
		}
	}
	gotV, wantV := make([]float32, K), make([]float32, K)
	if d := avAcc64BlocksF16(scores, h, gotV, N, K, 0, stride); d != 96 {
		t.Errorf("avAcc64BlocksF16 covered %d dims, want 96: the f16 AV kernel did not engage", d)
	}
	avAcc64Blocks(scores, w, wantV, N, K, 0, stride)
	for d := range K {
		if gotV[d] == 0 || math.Float32bits(gotV[d]) != math.Float32bits(wantV[d]) {
			t.Fatalf("AV dim %d: f16 kernel %v, f32 kernel on the widened cache %v (want equal and non-zero)", d, gotV[d], wantV[d])
		}
	}
}

// TestQkAcc64Keys16_declinesAtAnAliasingStride checks the 16-key kernel's stride guard: at a key stride that is a multiple of 4 KiB
// (3,072 floats, 12 KiB) its sixteen live rows share L1 sets and it declines, leaving the 4-key kernel to take the call; one 64 B line
// off (3,088 floats) it engages. Either way the scores equal the Go loop's.
func TestQkAcc64Keys16_declinesAtAnAliasingStride(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2 on this CPU")
	}
	rng := rand.New(rand.NewPCG(61, 67))
	const K, N = 96, 48
	for _, c := range []struct {
		stride, want16 int
	}{{3072, 0}, {3088, 48}} {
		q := make([]float32, K)
		for i := range q {
			q[i] = float32(rng.NormFloat64())
		}
		rows := make([]float32, (N-1)*c.stride+K)
		for i := range rows {
			rows[i] = float32(rng.NormFloat64())
		}
		if got := qkAcc64Keys16(q, rows, make([]float32, N), K, N, 0, c.stride); got != c.want16 {
			t.Errorf("stride %d: the 16-key kernel wrote %d keys, want %d", c.stride, got, c.want16)
		}
		out := make([]float32, N)
		if got := qkAcc64Keys(q, rows, out, K, N, 0, c.stride); got != N {
			t.Errorf("stride %d: the dispatch wrote %d keys, want %d", c.stride, got, N)
		}
		for n := range N {
			var acc float64
			for d := range K {
				acc += float64(q[d]) * float64(rows[n*c.stride+d])
			}
			if math.Float32bits(out[n]) != math.Float32bits(float32(acc)) {
				t.Fatalf("stride %d key %d: %08x != Go %08x", c.stride, n, math.Float32bits(out[n]), math.Float32bits(float32(acc)))
			}
		}
	}
}

// TestQkAcc64KeysF16_strideGuard checks the f16 QK dispatch: at a key stride that is a multiple of 4 KiB (2,048 halves) the 16-key
// kernel declines and the 4-key f16 kernel takes the call; one 64 B line off (2,080 halves) the 16-key kernel engages. The scores equal
// the f32 kernel's on the widened cache either way.
func TestQkAcc64KeysF16_strideGuard(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("no AVX2+F16C on this CPU")
	}
	rng := rand.New(rand.NewPCG(71, 73))
	const K, N = 96, 51
	for _, c := range []struct {
		stride, want16, wantAll int
	}{{2048, 0, 48}, {2080, 48, 48}} {
		h, w := f16Cache(rng, (N-1)*c.stride+K)
		q := make([]float32, K)
		for i := range q {
			q[i] = float32(rng.NormFloat64())
		}
		if n16, n4 := qkAcc64KeysSplitF16(q, h, make([]float32, N), K, N, 0, c.stride); n16 != c.want16 || n16+n4 != c.wantAll {
			t.Errorf("stride %d: the f16 dispatch wrote %d keys by the 16-key kernel and %d in all, want %d and %d", c.stride, n16, n16+n4, c.want16, c.wantAll)
		}
		got, want := make([]float32, N), make([]float32, N)
		MatmulQKAcc64F16(q, h, got, 1, K, N, 0, c.stride)
		MatmulQKAcc64(q, w, want, 1, K, N, 0, c.stride)
		for n := range N {
			if math.Float32bits(got[n]) != math.Float32bits(want[n]) {
				t.Fatalf("stride %d key %d: %08x != f32 kernel on the widened cache %08x", c.stride, n, math.Float32bits(got[n]), math.Float32bits(want[n]))
			}
		}
	}
}

// TestQkAcc64AVX4F16_matchesAVX4OnWidened pins the 4-key f16 kernel to the 4-key f32 kernel on the widened cache, bit for bit.
func TestQkAcc64AVX4F16_matchesAVX4OnWidened(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("no AVX2+F16C on this CPU")
	}
	rng := rand.New(rand.NewPCG(79, 83))
	for _, K := range []int{4, 64, 96, 128} {
		for _, stride := range []int{K, 3*K + 4, 2048} {
			const nBlocks = 5
			N := 4 * nBlocks
			h, w := f16Cache(rng, (N-1)*stride+K)
			q := make([]float32, K)
			for i := range q {
				q[i] = float32(rng.NormFloat64())
			}
			got, want := make([]float32, N), make([]float32, N)
			qkAcc64AVX4F16(&q[0], &h[0], stride*2, K/4, nBlocks, &got[0])
			qkAcc64AVX4(&q[0], &w[0], stride*4, K/4, nBlocks, &want[0])
			for n := range N {
				if math.Float32bits(got[n]) != math.Float32bits(want[n]) {
					t.Fatalf("K=%d stride=%d key %d: AVX4F16 %08x, AVX4 on the widened cache %08x", K, stride, n, math.Float32bits(got[n]), math.Float32bits(want[n]))
				}
			}
		}
	}
}
