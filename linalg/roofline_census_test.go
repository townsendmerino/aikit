package linalg

import (
	"bytes"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg/attn"
	"github.com/townsendmerino/aikit/linalg/bitvec"
	"github.com/townsendmerino/aikit/linalg/elem"
)

// The roofline census: each kernel perfgate or goinfer's kernelprint names, at one cache-resident
// and one streamed shape, single-threaded. tools/roofline runs these two benchmarks, reads the
// metrics they report and computes each row's fraction of this box's roofs; it owns the
// arithmetic, so these report counts, not rates.
//
// Sub-benchmark names are <class>/<label>/<regime>. class picks the compute roof: f32 and f64
// (the FMA peak, at 8 and 4 lanes a YMM register or 4 and 2 a NEON one), int8 (the int8 dot
// peak), stream (no MACs: the streaming-read roof only). Each row reports rdB/op, the bytes of
// its operands counted once (not the traffic: a two-pass kernel reads more), and MAC/op.

// censusResidentBytes and censusStreamedBytes size each kernel's largest operand: well inside a
// 512 KiB L2, and many times any last-level cache on the boxes this runs on.
const (
	censusResidentBytes = 128 << 10
	censusStreamedBytes = 128 << 20
)

// censusCase is one row: the call, its operand bytes and MACs, and a shape label.
type censusCase struct {
	run         func()
	bytes, macs int64
	shape       string
}

// censusEntry builds a kernel's case for a regime's operand budget; nil for a regime the kernel
// has no meaningful shape in (the reason goes in skip).
type censusEntry struct {
	class, label string
	resident     func(r *rand.Rand, budget int) censusCase
	streamed     func(r *rand.Rand, budget int) censusCase
	skip         string // why streamed is nil
}

func censusF32(r *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(r.NormFloat64())
	}
	return v
}

func censusBytes(r *rand.Rand, n int) []byte {
	v := make([]byte, n)
	for i := 0; i < n; i += 8 {
		u := r.Uint64()
		for j := i; j < min(i+8, n); j++ {
			v[j] = byte(u)
			u >>= 8
		}
	}
	return v
}

func censusI8(r *rand.Rand, n int) []int8 {
	b := censusBytes(r, n)
	v := make([]int8, n)
	for i, x := range b {
		v[i] = int8(x)
		if v[i] == -128 {
			v[i] = -127
		}
	}
	return v
}

func censusF16(r *rand.Rand, n int) []uint16 {
	v := make([]uint16, n)
	F32ToF16Slice(v, censusF32(r, n))
	return v
}

// censusMatShape returns K and N for an M-row matmul whose weight takes about budget bytes at
// bpw bytes a weight: K is 512 for a resident budget and 4096 for a streamed one, N a multiple of 64.
func censusMatShape(budget int, bpw float64) (K, N int) {
	K = 512
	if budget > censusResidentBytes {
		K = 4096
	}
	N = max(64, int(float64(budget)/(float64(K)*bpw))/64*64)
	return K, N
}

func serialWS() *Workspace {
	var ws Workspace
	ws.SetThreshold(math.MaxInt)
	return &ws
}

func censusMatmulF32(M int) func(*rand.Rand, int) censusCase {
	return func(r *rand.Rand, budget int) censusCase {
		K, N := censusMatShape(budget, 4)
		a, w, dst, ws := censusF32(r, M*K), censusF32(r, N*K), make([]float32, M*N), serialWS()
		return censusCase{func() { ws.MatmulBT(a, w, dst, M, K, N) }, int64(N*K*4 + M*K*4), int64(M * K * N), fmt.Sprintf("M%d_K%d_N%d", M, K, N)}
	}
}

func censusInt8(M int, w8a8 bool) func(*rand.Rand, int) censusCase {
	return func(r *rand.Rand, budget int) censusCase {
		K, N := censusMatShape(budget, 1)
		a, q, dst, ws := censusF32(r, M*K), censusI8(r, N*K), make([]float32, M*N), serialWS()
		s := make([]float32, N)
		for i := range s {
			s[i] = 0.01
		}
		run := func() { MatmulBTQ8Into(ws, a, q, s, dst, M, K, N) }
		if w8a8 {
			run = func() { MatmulBTW8A8Into(ws, a, q, s, dst, M, K, N) }
		}
		return censusCase{run, int64(N*K + 4*N + 4*M*K), int64(M * K * N), fmt.Sprintf("M%d_K%d_N%d", M, K, N)}
	}
}

// censusInt4 is the W4A8 matmul at group 32 with binary16 scales: canonical through the raw entry
// point, or this arch's repacked layout through a WeightMat (the production path).
func censusInt4(M int, repacked bool) func(*rand.Rand, int) censusCase {
	return func(r *rand.Rand, budget int) censusCase {
		const g = 32
		K, N := censusMatShape(budget, 0.5+2.0/g)
		a, dst, ws := censusF32(r, M*K), make([]float32, M*N), serialWS()
		q4, s16 := censusBytes(r, N*K/2), make([]uint16, N*K/g)
		for i := range s16 {
			s16[i] = F32ToF16(0.01)
		}
		c := censusCase{bytes: int64(N*K/2 + 2*N*K/g + 4*M*K), macs: int64(M * K * N), shape: fmt.Sprintf("M%d_K%d_N%d_g%d", M, K, N, g)}
		if !repacked {
			c.run = func() { MatmulBTW4A8F16Into(ws, a, q4, s16, dst, M, K, N, g) }
			return c
		}
		layout := "row4"
		wm, ok := RepackInt4Row4InPlaceF16(q4, s16, N, K, g)
		if !ok {
			layout = "splithalf"
			wm, ok = RepackInt4SplitHalfInPlaceF16(q4, s16, N, K, g)
		}
		if !ok {
			layout = "canonical"
			wm = WrapInt4F16(q4, s16, N, K, g)
		}
		c.run = func() { wm.MatmulBTW4A8Into(ws, a, dst, M) }
		c.shape += "_" + layout
		return c
	}
}

func censusKQuant(kind string) func(*rand.Rand, int) censusCase {
	return func(r *rand.Rand, budget int) censusCase {
		var rowBytes func(int) int
		switch kind {
		case "q4k":
			rowBytes = Q4KRowBytes
		case "q5k":
			rowBytes = Q5KRowBytes
		default:
			rowBytes = Q6KRowBytes
		}
		K, N := censusMatShape(budget, float64(rowBytes(4096))/4096)
		var raw []byte
		var wm WeightMat
		var err error
		switch kind {
		case "q4k":
			raw = q4kTestRows(r, N, K, false)
			wm, err = WrapQ4K(raw, N, K)
		case "q5k":
			raw = q5kTestRows(r, N, K, false)
			wm, err = WrapQ5K(raw, N, K)
		default:
			raw = q6kTestRows(r, N, K, 0)
			wm, err = WrapQ6K(raw, N, K)
		}
		if err != nil {
			panic(err)
		}
		a, dst, ws := censusF32(r, K), make([]float32, N), serialWS()
		return censusCase{func() { wm.MatmulBTInto(ws, a, dst, 1) }, int64(len(raw) + 4*K), int64(K * N), fmt.Sprintf("M1_K%d_N%d", K, N)}
	}
}

// censusAttn is one head's acc64 QK or AV at head dim 128 over G queries, the key count sized
// from the cache budget at four (f32) or two (f16) bytes an element.
func censusAttn(qk, f16 bool, G int) func(*rand.Rand, int) censusCase {
	return func(r *rand.Rand, budget int) censusCase {
		const hd = 128
		el := 4
		if f16 {
			el = 2
		}
		depth := budget / (hd * el)
		q, scores, acc := censusF32(r, G*hd), censusF32(r, G*depth), make([]float64, G*hd)
		c := censusCase{bytes: int64(depth*hd*el + 4*G*hd), macs: int64(G * depth * hd), shape: fmt.Sprintf("G%d_hd%d_depth%d", G, hd, depth)}
		var cache32 []float32
		var cache16 []uint16
		if f16 {
			cache16 = censusF16(r, depth*hd)
		} else {
			cache32 = censusF32(r, depth*hd)
		}
		switch {
		case qk && f16:
			dst := make([]float32, G*depth)
			c.run = func() { attn.MatmulQKAcc64F16(q, cache16, dst, G, hd, depth, 0, hd) }
		case !qk && f16:
			dst := make([]float32, G*hd)
			c.run = func() { attn.MatmulAVAcc64F16(scores, cache16, dst, acc, G, depth, hd, 0, hd) }
		case qk && G > 1:
			dst := make([]float32, G*depth)
			c.run = func() { attn.MatmulQKAcc64Group(q, cache32, dst, G, hd, depth, 0, hd) }
		case !qk && G > 1:
			dst := make([]float32, G*hd)
			c.run = func() { attn.MatmulAVAcc64Group(scores, cache32, dst, acc, G, depth, hd, 0, hd) }
		case qk:
			dst := make([]float32, depth)
			c.run = func() { attn.MatmulQKAcc64(q, cache32, dst, 1, hd, depth, 0, hd) }
		default:
			dst := make([]float32, hd)
			c.run = func() { attn.MatmulAVAcc64(scores, cache32, dst, acc, 1, depth, hd, 0, hd) }
		}
		if G > 1 {
			c.bytes += int64(4 * G * depth)
		}
		return c
	}
}

func censusFused(contract bool) func(*rand.Rand, int) censusCase {
	return func(r *rand.Rand, budget int) censusCase {
		const hd = 80
		np := max(16, budget/(3*hd*4))
		qh, kh, vBlk, ch := censusF32(r, np*hd), censusF32(r, np*hd), censusF32(r, np*hd), make([]float32, np*hd)
		sc := attn.NewFusedAttnScratch(np, hd)
		lo, hi := make([]int, np), make([]int, np)
		for i := range np {
			hi[i] = np - 1
		}
		fn := attn.AttendTileFused
		if contract {
			fn = attn.AttendTileFusedContractExp
		}
		return censusCase{func() { fn(MatmulBT, qh, kh, vBlk, ch, sc, np, hd, np, 1/8.944, lo, hi) }, int64(3 * np * hd * 4), int64(2 * np * np * hd), fmt.Sprintf("np%d_hd%d", np, hd)}
	}
}

// censusElem is an elementwise kernel over n elements of el input bytes each.
func censusElem(el int, mk func(r *rand.Rand, n int) func()) func(*rand.Rand, int) censusCase {
	return func(r *rand.Rand, budget int) censusCase {
		n := budget / el
		return censusCase{mk(r, n), int64(n * el), 0, fmt.Sprintf("n%d", n)}
	}
}

func f32In(f func(dst, src []float32)) func(*rand.Rand, int) func() {
	return func(r *rand.Rand, n int) func() {
		src, dst := censusF32(r, n), make([]float32, n)
		return func() { f(dst, src) }
	}
}

func both(f func(*rand.Rand, int) censusCase) (func(*rand.Rand, int) censusCase, func(*rand.Rand, int) censusCase) {
	return f, f
}

func censusEntries() []censusEntry {
	e := func(class, label string, f func(*rand.Rand, int) censusCase) censusEntry {
		res, str := both(f)
		return censusEntry{class: class, label: label, resident: res, streamed: str}
	}
	fusedSkip := "O(np²) work on O(np) data: no shape streams"
	return []censusEntry{
		e("f32", "MatmulBT_M1", censusMatmulF32(1)),
		e("f32", "MatmulBT_M32", censusMatmulF32(32)),
		e("f32", "MatmulBTQ8Into_M1", censusInt8(1, false)),
		e("int8", "MatmulBTW8A8Into_M1", censusInt8(1, true)),
		e("int8", "MatmulBTW8A8Into_M32", censusInt8(32, true)),
		e("int8", "MatmulBTW4A8F16Into_M1", censusInt4(1, false)),
		e("int8", "WeightMat.int4_M1", censusInt4(1, true)),
		e("int8", "WeightMat.int4_M32", censusInt4(32, true)),
		e("int8", "WeightMat.q4k_M1", censusKQuant("q4k")),
		e("int8", "WeightMat.q5k_M1", censusKQuant("q5k")),
		e("int8", "WeightMat.q6k_M1", censusKQuant("q6k")),
		e("f64", "MatmulQKAcc64", censusAttn(true, false, 1)),
		e("f64", "MatmulAVAcc64", censusAttn(false, false, 1)),
		e("f64", "MatmulQKAcc64F16", censusAttn(true, true, 1)),
		e("f64", "MatmulAVAcc64F16", censusAttn(false, true, 1)),
		e("f64", "MatmulQKAcc64Group_G4", censusAttn(true, false, 4)),
		e("f64", "MatmulAVAcc64Group_G4", censusAttn(false, false, 4)),
		{class: "f32", label: "AttendTileFused", resident: censusFused(false), skip: fusedSkip},
		{class: "f32", label: "AttendTileFusedContractExp", resident: censusFused(true), skip: fusedSkip},
		e("stream", "SoftmaxRowContractInto", censusElem(4, f32In(elem.SoftmaxRowContractInto))),
		e("stream", "SiLUContractInto", censusElem(4, f32In(elem.SiLUContractInto))),
		e("stream", "GELUContractInto", censusElem(4, f32In(elem.GELUContractInto))),
		e("stream", "QuantizeRowInt8", censusElem(4, func(r *rand.Rand, n int) func() {
			src, q := censusF32(r, n), make([]int8, n)
			return func() { QuantizeRowInt8(src, q) }
		})),
		e("stream", "F16ToF32Slice", censusElem(2, func(r *rand.Rand, n int) func() {
			src, dst := censusF16(r, n), make([]float32, n)
			return func() { F16ToF32Slice(dst, src) }
		})),
		e("stream", "F32ToF16Slice", censusElem(4, func(r *rand.Rand, n int) func() {
			src, dst := censusF32(r, n), make([]uint16, n)
			return func() { F32ToF16Slice(dst, src) }
		})),
		e("stream", "DequantizeRowsInt8Into", censusElem(1, func(r *rand.Rand, n int) func() {
			const cols = 4096
			rows := max(1, n/cols)
			q, s, dst := censusI8(r, rows*cols), make([]float32, rows), make([]float32, rows*cols)
			for i := range s {
				s[i] = 0.01
			}
			return func() { DequantizeRowsInt8Into(dst, q, s, rows, cols) }
		})),
		e("stream", "HammingRows", censusElem(1, func(r *rand.Rand, n int) func() {
			const words = 4 // 256-bit codes
			rows := max(1, n/(8*words))
			b := censusBytes(r, 8*words*(rows+1))
			u := make([]uint64, words*(rows+1))
			for i := range u {
				for j := range 8 {
					u[i] |= uint64(b[8*i+j]) << (8 * j)
				}
			}
			dst := make([]uint16, rows)
			return func() { bitvec.HammingRows(u[:words], u[words:], words, rows, dst) }
		})),
	}
}

// BenchmarkRooflineCensus is the census's rows; tools/roofline runs it.
func BenchmarkRooflineCensus(b *testing.B) {
	r := rand.New(rand.NewPCG(2026, 11))
	for _, e := range censusEntries() {
		for _, rg := range []struct {
			name   string
			budget int
			mk     func(*rand.Rand, int) censusCase
		}{{"resident", censusResidentBytes, e.resident}, {"streamed", censusStreamedBytes, e.streamed}} {
			if rg.mk == nil {
				continue
			}
			b.Run(e.class+"/"+e.label+"/"+rg.name, func(b *testing.B) {
				c := rg.mk(r, rg.budget)
				for b.Loop() {
					c.run()
				}
				b.ReportMetric(float64(c.bytes), "rdB/op")
				b.ReportMetric(float64(c.macs), "MAC/op")
				b.Log("shape " + c.shape)
			})
		}
	}
}

// streamReadGBs is a streaming-read probe: bytes.IndexByte, the standard library's vector scan,
// over a zero buffer for a byte it does not hold, so every byte is read once and nothing is
// written. It returns the best of passes runs in GB/s: a ceiling is a maximum.
func streamReadGBs(buf []byte, passes, repeat int) float64 {
	best := 0.0
	for range passes {
		t0 := time.Now()
		for range repeat {
			if bytes.IndexByte(buf, 1) >= 0 {
				panic("probe buffer not zero")
			}
		}
		if gbs := float64(len(buf)) * float64(repeat) / float64(time.Since(t0).Nanoseconds()); gbs > best {
			best = gbs
		}
	}
	return best
}

// BenchmarkRooflineRoofs measures this core's roofs. Run it at -test.benchtime=1x: each sub-
// benchmark does its own repeats and reports its ceiling as a metric; ns/op means nothing here.
//   - fma: the f32 FMA peak probe, GFLOP/s.
//   - int8: the int8 dot peak probe where this arch has one, GMAC/s.
//   - stream: streamReadGBs over 512 MiB, and over 32 KiB for the cross-check that the probe is
//     not instruction-bound (the resident rate must be well above the streamed one).
func BenchmarkRooflineRoofs(b *testing.B) {
	b.Run("fma", func(b *testing.B) {
		gflops, ok := measureFMAPeak()
		if !ok {
			b.Skip("no FMA peak probe on this arch or CPU")
		}
		b.ReportMetric(gflops, "GFLOP/s")
	})
	b.Run("int8", func(b *testing.B) {
		gmacs, ok := int8PeakGMACs()
		if !ok {
			b.Skip("no int8 dot peak probe on this arch or CPU")
		}
		b.ReportMetric(gmacs, "GMAC/s")
	})
	b.Run("stream", func(b *testing.B) {
		big := make([]byte, 512<<20)
		for i := range big { // touch every page so the first pass does not time page faults
			big[i] = 0
		}
		small := make([]byte, 32<<10)
		b.ReportMetric(streamReadGBs(big, 5, 1), "GB/s")
		b.ReportMetric(streamReadGBs(small, 5, (512<<20)/len(small)), "residentGB/s")
	})
}
