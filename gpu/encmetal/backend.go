//go:build darwin

// Package encmetal is the native-Metal encoder.Backend — text-encoder matmuls on Apple
// GPUs, cgo-free (docs/task-native-gpu.md, Phase 4). It is the Apple mirror of
// gpu/enccuda.
//
// Importing it registers "metal" with encoder.RegisterBackend, so
// encoder.NewBackend("metal") resolves; aikit's core encoder never imports gpu, and the
// default build stays pure-Go CPU.
//
// # Two things about encoder.Backend shape this backend (identical to enccuda's)
//
//  1. **The interface takes HOST slices and has no residency hook**, so both operands are
//     copied into resident buffers on EVERY call. On Metal that copy is a memcpy into a
//     shared (UMA) MTLBuffer, NOT a PCIe upload — cheap — but it is still a copy, and a
//     12-layer forward re-copies the same weights ~72 times.
//
//     A pointer-keyed weight cache is the obvious idea and it is WRONG here. In attention,
//     `b` is `kH`/`vHT` — scratch slices from a pool whose backing array is stable across
//     calls while their CONTENTS change every call. Keying residency on the pointer would
//     serve stale data, silently, and only for attention. A size threshold does not rescue
//     it either: at long sequences the per-head QKᵀ grows past any threshold you would pick
//     for perf. So: no cache. TestEncMetal_noStaleOperands is the regression gate. Fixing
//     this properly means widening encoder.Backend with a "this operand is a resident
//     weight" concept — a Hard-tier change.
//
//  2. **A backend sees EVERY f32 matmul**, including the tiny per-head QKᵀ and scores·V,
//     far below where a device round-trip pays. So this backend declines small work to the
//     CPU path. That threshold is the product of the benchmark (docs/BENCH-gpu.md: "the
//     curve IS the dispatch threshold") — see minGPUFlops.
//
// The one Metal-vs-CUDA difference that matters here: UMA means the "upload" is a
// `copy()` into the buffer's `Floats()` view and the "download" is a `copy()` back out,
// with no explicit device transfer and no data race of the kind the CUDA upload path had
// to synchronize against (an MTLBuffer is a live Go slice). Each device dispatch runs
// under one runtime.LockOSThread, because Run2D allocs+drains an NSAutoreleasePool that
// objc requires on a single thread.
package encmetal

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/townsendmerino/aikit/encoder"
	gpu "github.com/townsendmerino/aikit/gpu"
)

func init() {
	encoder.RegisterBackend("metal", func() (encoder.Backend, error) { return New() })
}

// minGPUFlops is the dispatch threshold: below this many FLOPs (2*M*K*N) a matmul runs on
// the CPU path instead of the device. Per docs/BENCH-gpu.md the crossover IS the
// deliverable of BenchmarkEncoderMatmulBT, so this is DERIVED from the sweep, not picked —
// and deliberately NOT copied from enccuda's 1 MFLOP: the CPU, the transfer cost (UMA vs
// PCIe) and the crossover are all different here.
//
// This backend runs the PRODUCTION f32 GEMM via ViT.GEMMF32Plan — the aligned no-barrier
// gemm_f32_sg_big (direct device loads, ~1080 GFLOP/s, up to 4.25× the CPU) for shapes with
// M,N%32==0 & K%8==0, and the general gemm_f32_sg otherwise — NOT the correctness-first
// gemm_f32_tiled. Successive kernels dropped the crossover ~16× from where the tiled one put
// it: tiled ~2.1 GFLOP → gemm_f32_sg ~0.5 GFLOP → gemm_f32_sg_big ~0.13 GFLOP.
//
// Measurement (Apple M1 Pro, GPU vs 10-core pure-Go CPU, same box, ns/op, both operands copied
// into UMA buffers and the result copied back every call — the honest cost of this interface).
// GFLOP/s in parentheses (kernel-only throughput is higher; these include the copies):
//
//	   1 MFLOP  M=80   K=64   N=80    cpu 224us(3.7)   gpu 252us(3.3)   0.89x   CPU
//	   2 MFLOP  M=128  K=64   N=128   cpu 576us(3.6)   gpu 238us(8.8)   2.42x   (CPU-artifact)
//	  19 MFLOP  M=64   K=384  N=384   cpu 265us(71)    gpu 283us(67)    0.93x   CPU
//	  24 MFLOP  M=80   K=384  N=384   cpu 332us(71)    gpu 398us(59)    0.83x   CPU
//	 151 MFLOP  M=128  K=768  N=768   cpu 592us(255)   gpu 444us(340)   1.33x   GPU
//	 604 MFLOP  M=128  K=768  N=3072  cpu 2.24ms(270)  gpu 1.74ms(347)  1.28x   GPU
//	2416 MFLOP  M=512  K=768  N=3072  cpu 7.91ms(305)  gpu 3.56ms(678)  2.22x   GPU
//	8590 MFLOP  M=1024 K=1024 N=4096  cpu 37.3ms(230)  gpu 8.78ms(978)  4.25x   GPU
//
// The crossover is now between 24 MFLOP (GPU loses, 0.83×) and 151 MFLOP (GPU wins, 1.33×),
// so the threshold sits below the proven win at 151. The M1-Pro CPU is strong (~300 GFLOP/s
// in its blocked path), but gemm_f32_sg_big reaches ~978 GFLOP/s effective (incl. copies),
// enough to pay once the size amortizes the ~250us fixed dispatch+sync floor. The lone
// small-shape "win" (2 MFLOP, 2.42×) is a CPU artifact — the pure-Go path switches
// scalar→blocked at ~4 MFLOP — and is pure GPU floor; routing a forward's many small matmuls
// each to that floor would be slower in aggregate, so the threshold keeps them on the CPU.
//
// HONESTY NOTE (BENCH-gpu.md: microbenchmarks tune, end-to-end publishes):
// TestEncMetal_endToEnd times a REAL MiniLM Encode. Even at this lower crossover a single
// short-sequence forward (small L, hidden 384) has no matmul above the threshold, so the
// backend correctly stays entirely on the CPU and the numerics are identical; the device
// pays for batched or larger-model encode.
const minGPUFlops = 1 << 27 // ~134 MFLOP — derived from the sweep above (sg_big crossover)

// Backend is the Metal encoder backend.
type Backend struct {
	dev *gpu.Device
	q   gpu.Queue
	k   gpu.ViT
	cpu encoder.Backend // the pure-Go path, for shapes below the threshold

	// Device buffers are reused across calls (grown on demand) — safe, since they are
	// re-copied every call. Caching CONTENTS is what is NOT safe; see the package note.
	mu               sync.Mutex
	aBuf, bBuf, cBuf gpu.Buffer
	aCap, bCap, cCap int
	mS, nS, kS       gpu.Buffer // reusable scalar buffers (Metal has no by-value arg)
	// q8 holds DEQUANTIZED int8 weights resident on the device. Sound here (a model weight
	// is write-once) where an f32 cache would not be; see MatmulBTQ8.
	q8 map[uintptr]q8w
}

// New creates the backend. It fails (rather than degrading silently) when there is no
// Metal device, so encoder.NewBackend("metal") reports the real reason.
func New() (*Backend, error) {
	dev, err := gpu.CreateSystemDefaultDevice()
	if err != nil {
		return nil, fmt.Errorf("encmetal: %w", err)
	}
	k, err := dev.NewViT()
	if err != nil {
		dev.ReleaseObjects()
		return nil, fmt.Errorf("encmetal: load kernels: %w", err)
	}
	cpu, err := encoder.NewBackend("cpu")
	if err != nil {
		dev.ReleaseObjects()
		return nil, err
	}
	b := &Backend{dev: dev, q: dev.NewCommandQueue(), k: k, cpu: cpu}
	b.mS, b.nS, b.kS = gpu.NewBufferOf(dev, []uint32{0}), gpu.NewBufferOf(dev, []uint32{0}), gpu.NewBufferOf(dev, []uint32{0})
	return b, nil
}

func (b *Backend) Name() string { return "metal" }

// grow ensures buf holds at least n float32s, reallocating when it does not.
func (b *Backend) grow(buf *gpu.Buffer, cap *int, n int) {
	if n <= *cap {
		return
	}
	if *cap > 0 {
		b.dev.ReleaseBuf(*buf)
	}
	*buf, *cap = b.dev.NewBufferLen(n), n
}

// MatmulBT computes dst[M,N] = a[M,K] · b[N,K]ᵀ.
//
// Small shapes go to the CPU path — see minGPUFlops. Errors are not in the signature
// (encoder.Backend predates this backend), so a device failure falls back to the CPU path
// rather than producing wrong numbers. Silent wrongness is the one outcome not on the table.
//
// Output contract: overwrites dst[:M*N]; do not pre-zero (the device result is read back over it, and the CPU fallback zeroes internally).
func (b *Backend) MatmulBT(a, w, dst []float32, M, K, N int) {
	if 2*int64(M)*int64(K)*int64(N) < minGPUFlops || M <= 0 || K <= 0 || N <= 0 {
		b.cpu.MatmulBT(a, w, dst, M, K, N)
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.gpuMatmul(a, w, dst, M, K, N); err != nil {
		b.cpu.MatmulBT(a, w, dst, M, K, N)
	}
}

func (b *Backend) gpuMatmul(a, w, dst []float32, M, K, N int) (err error) {
	defer func() {
		if r := recover(); r != nil { // MustBuf panics on OOM; degrade, don't die
			err = fmt.Errorf("encmetal: device allocation failed: %v", r)
		}
	}()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	b.grow(&b.aBuf, &b.aCap, M*K)
	b.grow(&b.bBuf, &b.bCap, N*K)
	b.grow(&b.cBuf, &b.cCap, M*N)
	copy(b.aBuf.Floats()[:M*K], a[:M*K]) // UMA "upload": memcpy into a shared buffer
	copy(b.bBuf.Floats()[:N*K], w[:N*K])
	b.mS.SetU32(uint32(int32(M)))
	b.nS.SetU32(uint32(int32(N)))
	b.kS.SetU32(uint32(int32(K)))
	p, gx, gy, tgx, tgy := b.k.GEMMF32Plan(M, N, K) // aligned → gemm_f32_sg_big, else gemm_f32_sg
	b.q.Run2D(p, gx, gy, tgx, tgy, b.aBuf, b.bBuf, b.cBuf, b.mS, b.nS, b.kS)
	copy(dst[:M*N], b.cBuf.Floats()[:M*N]) // UMA "download" (Run2D already waited)
	return nil
}

// --- int8 (weight-only) path: encoder.Q8Backend ---
//
// The encoder's int8 is WEIGHT-ONLY — int8 weights widened to f32, multiplied against f32
// activations. It is NOT W8A8, and reaching for the W8A8 GEMV here would be a quality
// regression, not an optimization: quantizing the activations was measured and rejected for
// falling below the 0.97 reranker bar (see encoder.Q8Backend's contract). UMA makes the
// "just bind the int8 codes into gemm_w8a8" shortcut especially tempting — don't; the
// activation must stay f32.
//
// So this dequantizes the weight ONCE, keeps the f32 result resident, and runs the SAME
// GEMMF32Plan kernel (gemm_f32_sg_big) the f32 path uses — identical arithmetic to the CPU's
// matmulBTQ8Into, which is what makes the tight parity bound available. It is also the CHEAPER
// shape: the CPU redoes the N*K widen on every call (the documented reason LoadQ8 ran ~5×
// slower than Load), whereas here a 12-layer forward pays the widen once per matrix.
//
// Caching is SOUND here where it was not for f32. The f32 backend has no cache because its
// `b` operand is attention's pooled kH/vHT scratch (stable pointer, changing contents). A `wq`
// is a model weight: loaded once, never written. The negative control (the ACTIVATION must
// never be cached) is still tested.

// q8w is one dequantized weight resident on the device.
type q8w struct {
	buf gpu.Buffer
	n   int // element count, so a recycled address at a different length cannot hit
}

// residentQ8 returns the UMA buffer holding dequant(wq), dequantizing+uploading on first use.
// Dequant runs on the host — one pass per weight for the process lifetime, so a device kernel
// would buy nothing and cost a launch.
func (b *Backend) residentQ8(wq []int8, wscales []float32, K, N int) gpu.Buffer {
	k := q8key(wq)
	if c, ok := b.q8[k]; ok && c.n == len(wq) {
		return c.buf
	}
	// Dequantize STRAIGHT into the UMA buffer's backing (lens §4.4). NewBufferOf
	// would first stage a make([]float32, N*K) on the Go heap only to memcpy it into
	// this same shared memory — 37.8 MB × 12 layers = 453 MB allocated, written, and
	// immediately garbage on the first forward. On UMA the CPU can write the device
	// buffer directly, so there is no host copy to stage.
	buf := b.dev.NewBufferLen(N * K)
	deq := buf.Floats()[:N*K]
	for n := range N {
		sc := wscales[n]
		row, src := deq[n*K:(n+1)*K], wq[n*K:(n+1)*K]
		for i := range row {
			row[i] = float32(src[i]) * sc
		}
	}
	if b.q8 == nil {
		b.q8 = map[uintptr]q8w{}
	}
	b.q8[k] = q8w{buf: buf, n: len(wq)}
	return buf
}

func q8key(w []int8) uintptr {
	if len(w) == 0 {
		return 0
	}
	return uintptr(unsafe.Pointer(&w[0]))
}

// MatmulBTQ8 implements encoder.Q8Backend: dst[M,N] = a[M,K] · dequant(wq)[N,K]ᵀ, activations
// untouched. Returns false to leave the call on the CPU path.
//
// Output contract: overwrites dst[:M*N]; do not pre-zero (when it returns true; on false dst is untouched).
func (b *Backend) MatmulBTQ8(dst, a []float32, wq []int8, wscales []float32, M, K, N int) bool {
	if 2*int64(M)*int64(K)*int64(N) < minGPUFlops || M <= 0 || K <= 0 || N <= 0 {
		return false
	}
	if len(a) < M*K || len(wq) < N*K || len(wscales) < N || len(dst) < M*N {
		return false
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.gpuMatmulQ8(dst, a, wq, wscales, M, K, N) == nil
}

func (b *Backend) gpuMatmulQ8(dst, a []float32, wq []int8, wscales []float32, M, K, N int) (err error) {
	defer func() {
		if r := recover(); r != nil { // MustBuf panics on OOM; degrade, don't die
			err = fmt.Errorf("encmetal: device allocation failed: %v", r)
		}
	}()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	wb := b.residentQ8(wq, wscales, K, N)
	b.grow(&b.aBuf, &b.aCap, M*K)
	b.grow(&b.cBuf, &b.cCap, M*N)
	// The ACTIVATION is copied every call — it changes every call. Only the weight is resident.
	copy(b.aBuf.Floats()[:M*K], a[:M*K])
	b.mS.SetU32(uint32(int32(M)))
	b.nS.SetU32(uint32(int32(N)))
	b.kS.SetU32(uint32(int32(K)))
	p, gx, gy, tgx, tgy := b.k.GEMMF32Plan(M, N, K)
	b.q.Run2D(p, gx, gy, tgx, tgy, b.aBuf, wb, b.cBuf, b.mS, b.nS, b.kS)
	copy(dst[:M*N], b.cBuf.Floats()[:M*N])
	return nil
}

// Close releases the device and every reused buffer (buffers first, so no in-flight work
// references freed memory; the last Run2D already waited).
func (b *Backend) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.dev != nil {
		b.dev.ReleaseAll() // frees the reused buffers AND the resident dequantized weights
		b.dev.ReleaseObjects()
		b.dev = nil
		b.q8 = nil
	}
	return nil
}
