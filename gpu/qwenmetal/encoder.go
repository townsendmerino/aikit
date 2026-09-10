//go:build darwin

// Package qwenmetal is the native-Metal vision.QwenResidentEncoder — the Qwen2.5-VL
// vision tower running on Apple GPUs, cgo-free (docs/task-native-gpu.md, Phase 3). It is
// the Apple counterpart of gpu/qwencuda, and mirrors it op for op; the device-level
// differences are all here.
//
// Importing this package is the opt-in; aikit's core `vision` never imports `gpu`, so the
// default build stays pure-Go CPU. This package's init() plugs the factory into
// vision.RegisterQwenResident, and QwenVisionEncoder.EnableResident then routes here.
//
// WHAT RUNS WHERE, AND WHY (identical contract to gpu/qwencuda)
// ------------------------------------------------------------
// The transformer blocks run on the device. Two things stay on the host, both on purpose:
//
//   - The WINDOW PERMUTATION. Qwen reorders patches into window groups before the blocks
//     and back afterwards. Rather than a gather kernel, this permutes the PIXEL ROWS
//     before upload: the patch embed is row-wise, so permuting its input permutes its
//     output identically, and the de-window is a permutation of the result on the way
//     back. The index arithmetic stays in vision.BuildWindowPlan where the CPU path's
//     version already lives — reimplementing it is exactly where a ViT goes silently
//     wrong.
//   - The PATCH MERGER, per the seam's contract: three small ops over n_patches/merge²
//     groups against many blocks of tower, left on the CPU (vision.MergeHidden).
//
// Metal ↔ CUDA divergences (inverses of gpu/qwencuda's): dispatchThreads launches exactly
// n threads (no bounds checks); Apple UMA makes a buffer's Floats()/U32s() a zero-copy
// live view (no explicit upload/download); scalars bind as 1-element buffers; and the
// whole forward runs under ONE runtime.LockOSThread, because each dispatch's Run1D
// allocates and drains an NSAutoreleasePool that objc requires on a single thread.
package qwenmetal

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/aikit/vision"
)

func init() {
	vision.RegisterQwenResident(func(e *vision.QwenVisionEncoder) (vision.QwenResidentEncoder, error) {
		return newEncoder(e)
	})
}

// mat is one resident weight. Qwen towers load fp32 (the parity configuration) or int8;
// both are supported, and the GEMM used per call follows this flag.
type mat struct {
	quant      bool
	a, b       gpu.Buffer // int8 codes + scales when quant; a is the f32 matrix otherwise
	rows, cols int
}

type block struct {
	norm1w, norm2w gpu.Buffer
	qkvw           mat
	qkvb           gpu.Buffer
	projw          mat
	projb          gpu.Buffer
	gatew, upw     mat
	gateb, upb     gpu.Buffer
	downw          mat
	downb          gpu.Buffer
}

type encoder struct {
	dev *gpu.Device
	q   gpu.Queue
	k   gpu.ViT
	src *vision.QwenVisionEncoder
	w   vision.QwenGPUWeights

	patchW gpu.Buffer
	blocks []block

	// Reusable scalar buffers (Metal has no by-value kernel arg): three uint32 slots and
	// two float slots, rewritten before each dispatch. Safe to reuse because each Run1D
	// commits and waits, so a scalar is fully consumed before it is overwritten.
	s0, s1, s2       gpu.Buffer
	epsBuf, scaleBuf gpu.Buffer

	mu sync.Mutex
	// Scratch is sized to the patch count of the first forward and grown on demand:
	// Qwen is dynamic-resolution, so unlike SigLIP there is no fixed np to preallocate.
	cap               int
	pix, h, n1, n2    gpu.Buffer
	qkv, att, projOut gpu.Buffer
	gate, up          gpu.Buffer
	qi8, qs           gpu.Buffer
	cosB, sinB        gpu.Buffer
	segS, segE        gpu.Buffer
}

func newEncoder(src *vision.QwenVisionEncoder) (enc *encoder, err error) {
	w := src.GPUWeights()
	dev, err := gpu.CreateSystemDefaultDevice()
	if err != nil {
		return nil, err
	}
	defer func() {
		if r := recover(); r != nil {
			dev.ReleaseAll()
			dev.ReleaseObjects()
			enc, err = nil, fmt.Errorf("qwenmetal: device upload failed: %v", r)
		}
	}()
	kern, err := dev.NewViT()
	if err != nil {
		dev.ReleaseObjects()
		return nil, err
	}
	e := &encoder{dev: dev, q: dev.NewCommandQueue(), k: kern, src: src, w: w}
	upF := func(x []float32) gpu.Buffer { return gpu.NewBufferOf(dev, x) }
	upM := func(m vision.QwenGPUMat) mat {
		if m.Quantized {
			return mat{quant: true, a: gpu.NewBufferOf(dev, m.Q), b: gpu.NewBufferOf(dev, m.Scales), rows: m.Rows, cols: m.Cols}
		}
		return mat{a: gpu.NewBufferOf(dev, m.F32), rows: m.Rows, cols: m.Cols}
	}
	e.patchW = upF(w.PatchW)
	e.blocks = make([]block, len(w.Blocks))
	for i, B := range w.Blocks {
		e.blocks[i] = block{
			norm1w: upF(B.Norm1w), norm2w: upF(B.Norm2w),
			qkvw: upM(B.QKVw), qkvb: upF(B.QKVb),
			projw: upM(B.Projw), projb: upF(B.Projb),
			gatew: upM(B.Gatew), upw: upM(B.Upw),
			gateb: upF(B.Gateb), upb: upF(B.Upb),
			downw: upM(B.Downw), downb: upF(B.Downb),
		}
	}
	e.s0, e.s1, e.s2 = gpu.NewBufferOf(dev, []uint32{0}), gpu.NewBufferOf(dev, []uint32{0}), gpu.NewBufferOf(dev, []uint32{0})
	e.epsBuf, e.scaleBuf = gpu.NewBufferOf(dev, []float32{0}), gpu.NewBufferOf(dev, []float32{0})
	return e, nil
}

// ensure grows the per-call scratch to hold n patches. Buffers are released and
// reallocated rather than kept at a high-water mark for every shape, because a
// dynamic-resolution tower can see a huge image once and small ones thereafter.
func (e *encoder) ensure(n int) {
	if n <= e.cap {
		return
	}
	if e.cap > 0 {
		for _, b := range []gpu.Buffer{e.pix, e.h, e.n1, e.n2, e.qkv, e.att, e.projOut, e.gate, e.up, e.qi8, e.qs, e.cosB, e.sinB, e.segS, e.segE} {
			e.dev.ReleaseBuf(b)
		}
	}
	H, I, hd := e.w.Hidden, e.w.Inter, e.w.HeadDim
	f32 := func(c int) gpu.Buffer { return e.dev.NewBufferLen(c) }
	e.pix = f32(n * e.w.PatchDim)
	e.h, e.n1, e.n2 = f32(n*H), f32(n*H), f32(n*H)
	e.qkv = f32(n * 3 * H)
	e.att, e.projOut = f32(n*H), f32(n*H)
	e.gate, e.up = f32(n*I), f32(n*I)
	wide := max(I, H)
	e.qi8 = gpu.NewBufferOf(e.dev, make([]int8, n*wide))
	e.qs = f32(n)
	e.cosB, e.sinB = f32(n*hd), f32(n*hd)
	e.segS = gpu.NewBufferOf(e.dev, make([]uint32, n))
	e.segE = gpu.NewBufferOf(e.dev, make([]uint32, n))
	e.cap = n
}

func (e *encoder) setI(b gpu.Buffer, v int) { b.SetU32(uint32(int32(v))) }

func vitTG(n int) int {
	if n < gpu.ViTBlock {
		return n
	}
	return gpu.ViTBlock
}

// uploadSeg writes int32 per-patch bounds into a resident uint32 buffer in place (UMA).
// Bounds are non-negative, so the uint32 bit pattern equals the int32 the kernel reads.
func uploadSeg(b gpu.Buffer, s []int32) {
	v := b.U32s()
	for i, x := range s {
		v[i] = uint32(x)
	}
}

// proj runs one projection: dst[M,N] = src[M,K] · w[N,K]ᵀ + bias, quantizing the
// activation first when the weight is int8 — the same op sequence as gpu/qwencuda's.
func (e *encoder) proj(src gpu.Buffer, m mat, bias, dst gpu.Buffer, M int) {
	K, N := m.cols, m.rows
	if m.quant {
		// W8A8 stays on the tiled kernel — simdgroup_matrix has no int8 form.
		tgxg, tgyg, ttx, tty := gpu.TileDims(M, N)
		e.setI(e.s0, M)
		e.setI(e.s1, K)
		e.q.Run1D(e.k.QuantRows, M*gpu.ViTBlock, gpu.ViTBlock, src, e.qi8, e.qs, e.s0, e.s1)
		e.setI(e.s0, M)
		e.setI(e.s1, N)
		e.setI(e.s2, K)
		e.q.Run2D(e.k.GEMMW8A8Tiled, tgxg, tgyg, ttx, tty, e.qi8, e.qs, m.a, m.b, dst, e.s0, e.s1, e.s2)
	} else {
		p, gx, gy, tgx, tgy := e.k.GEMMF32Plan(M, N, K) // aligned → sg_big, else sg
		e.setI(e.s0, M)
		e.setI(e.s1, N)
		e.setI(e.s2, K)
		e.q.Run2D(p, gx, gy, tgx, tgy, src, m.a, dst, e.s0, e.s1, e.s2)
	}
	if bias.Len() == 0 {
		return
	}
	e.setI(e.s0, M)
	e.setI(e.s1, N)
	e.q.Run1D(e.k.AddBias, M*N, vitTG(M*N), dst, bias, e.s0, e.s1)
}

func (e *encoder) rms(src, w, dst gpu.Buffer, rows, dim int) {
	e.setI(e.s0, rows)
	e.setI(e.s1, dim)
	e.epsBuf.Floats()[0] = 1e-6
	e.q.Run1D(e.k.RMSNorm, rows*gpu.ViTBlock, gpu.ViTBlock, src, w, dst, e.s0, e.s1, e.epsBuf)
}

// ForwardViT runs the ViT blocks on the device and returns the pre-merge hidden state in
// ORIGINAL patch order, matching the CPU contract. One OS-thread pin spans the whole
// forward (every Run1D allocs+drains an NSAutoreleasePool, which objc requires on one
// thread).
func (e *encoder) ForwardViT(pixelValues []float32, gridTHW [][3]int) ([]float32, error) {
	plan, err := e.src.BuildWindowPlan(gridTHW)
	if err != nil {
		return nil, err
	}
	n := plan.NPatches

	// Threadgroup-memory budget check (audit C-02). AttentionSeg stages a
	// per-query score row in DYNAMIC threadgroup memory of maxSeg*4 bytes, on top
	// of the kernel's two static threadgroup arrays (smax/ssum, ViTBlock floats
	// each). A dispatch whose total exceeds the device maximum — ~32 KiB on Apple
	// GPUs — aborts the command buffer, and waitUntilCompleted returns cleanly
	// from an abort: e.att simply keeps the previous layer's contents and the
	// forward returns a plausible, wrong hidden state. mustCmdBufOK now turns
	// that into a loud panic rather than silence, but declining here is better
	// still: it is the documented obligation on MaxThreadgroupMemoryLength, it
	// costs nothing, and it names the real limit instead of reporting a GPU
	// fault. CUDA's analogue already fails the launch with an error.
	//
	// Both branches of the per-layer dispatch are covered by taking the larger of
	// the window and full-attention segment bounds.
	maxSegAny := max(plan.MaxWinSeg, plan.MaxFullSeg)
	if need, have := attnThreadgroupBytes(maxSegAny), e.dev.MaxThreadgroupMemoryLength(); need > have {
		return nil, fmt.Errorf("gpu: qwen ViT attention needs %d B of threadgroup memory for %d patches (max segment %d) but the device allows %d — reduce max_pixels", need, n, maxSegAny, have)
	}
	H, I, nH, hd := e.w.Hidden, e.w.Inter, e.w.NumHeads, e.w.HeadDim
	pd := e.w.PatchDim
	if len(pixelValues) != n*pd {
		return nil, fmt.Errorf("qwenmetal: pixel_values len %d, want %d", len(pixelValues), n*pd)
	}
	merge := e.w.SpatialMergeSize
	mergeUnit := merge * merge
	groups := n / mergeUnit

	e.mu.Lock()
	defer e.mu.Unlock()
	e.ensure(n)

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Permute pixel rows into WINDOW order on the host, then UMA-copy — the patch embed is
	// row-wise, so this makes the device's hidden state window-ordered without a gather.
	pix := e.pix.Floats()
	for g := range groups {
		src := plan.WinIdx[g]
		for u := range mergeUnit {
			dp, sp := (g*mergeUnit+u)*pd, (src*mergeUnit+u)*pd
			copy(pix[dp:dp+pd], pixelValues[sp:sp+pd])
		}
	}
	copy(e.cosB.Floats(), plan.Cos)
	copy(e.sinB.Floats(), plan.Sin)

	// patch embed: h = pixels[n,pd] · patchW[H,pd]ᵀ (no bias).
	e.setI(e.s0, n)
	e.setI(e.s1, H)
	e.setI(e.s2, pd)
	pp, pgx, pgy, ptgx, ptgy := e.k.GEMMF32Plan(n, H, pd) // aligned → sg_big, else sg
	e.q.Run2D(pp, pgx, pgy, ptgx, ptgy, e.pix, e.patchW, e.h, e.s0, e.s1, e.s2)

	scale := float32(1.0 / math.Sqrt(float64(hd)))
	curFull := -1 // which segment bounds are currently uploaded: 1 full, 0 windowed
	for li := range e.blocks {
		B := &e.blocks[li]
		full := e.src.IsFullAtt(li)
		want := 0
		if full {
			want = 1
		}
		if want != curFull {
			ss, se := plan.WinStart, plan.WinEnd
			if full {
				ss, se = plan.FullStart, plan.FullEnd
			}
			uploadSeg(e.segS, ss)
			uploadSeg(e.segE, se)
			curFull = want
		}
		maxSeg := plan.MaxWinSeg
		if full {
			maxSeg = plan.MaxFullSeg
		}

		// --- attention block ---
		e.rms(e.h, B.norm1w, e.n1, n, H)
		e.proj(e.n1, B.qkvw, B.qkvb, e.qkv, n)
		e.setI(e.s0, n)
		e.setI(e.s1, nH)
		e.setI(e.s2, hd)
		e.q.Run1D(e.k.RopeQK, n*nH*(hd/2), vitTG(n*nH*(hd/2)), e.qkv, e.cosB, e.sinB, e.s0, e.s1, e.s2)
		e.setI(e.s0, n)
		e.setI(e.s1, nH)
		e.setI(e.s2, hd)
		e.scaleBuf.Floats()[0] = scale
		e.q.Run1DTG(e.k.AttentionSeg, n*nH*gpu.ViTBlock, gpu.ViTBlock, maxSeg*4,
			e.qkv, e.att, e.segS, e.segE, e.s0, e.s1, e.s2, e.scaleBuf)
		e.proj(e.att, B.projw, B.projb, e.projOut, n)
		e.setI(e.s0, n*H)
		e.q.Run1D(e.k.AddVec, n*H, vitTG(n*H), e.h, e.projOut, e.s0)
		// --- gated SiLU MLP ---
		e.rms(e.h, B.norm2w, e.n2, n, H)
		e.proj(e.n2, B.gatew, B.gateb, e.gate, n)
		e.proj(e.n2, B.upw, B.upb, e.up, n)
		e.setI(e.s0, n*I)
		e.q.Run1D(e.k.SiLUMul, n*I, vitTG(n*I), e.gate, e.up, e.s0)
		e.proj(e.gate, B.downw, B.downb, e.projOut, n)
		e.setI(e.s0, n*H)
		e.q.Run1D(e.k.AddVec, n*H, vitTG(n*H), e.h, e.projOut, e.s0)
	}

	// de-window back to original patch order (the last Run1D already waited; UMA view).
	hWin := e.h.Floats()
	out := make([]float32, n*H)
	for g := range groups {
		dst := plan.WinIdx[g]
		for u := range mergeUnit {
			dp, sp := (dst*mergeUnit+u)*H, (g*mergeUnit+u)*H
			copy(out[dp:dp+H], hWin[sp:sp+H])
		}
	}
	return out, nil
}

// Close releases the device buffers and objects (buffers first, so no in-flight work
// references freed memory; the forward's last Run1D already waited).
func (e *encoder) Close() {
	e.dev.ReleaseAll()
	e.dev.ReleaseObjects()
}

// attnThreadgroupBytes is the threadgroup memory one AttentionSeg dispatch needs
// when the largest per-patch segment in the plan is maxSeg: the dynamic score row
// the caller binds at index 0 (maxSeg floats), plus the kernel's two STATIC
// threadgroup arrays, smax[LNBLOCK] and ssum[LNBLOCK], which are ViTBlock floats
// each (gpu/metal_vit.go, attention_seg).
//
// Counting the static pair matters: the audit put the abort threshold at
// n > 8192 patches from the dynamic term alone, but against a 32 KiB device
// budget the extra 2 KiB brings the real limit down to a segment of 7680.
// Factored out of ForwardViT so it can be tested without a checkpoint — the
// qwenmetal suite skips entirely when testdata/qwen25vl-vision-tiny is absent.
func attnThreadgroupBytes(maxSeg int) int { return maxSeg*4 + 2*gpu.ViTBlock*4 }
