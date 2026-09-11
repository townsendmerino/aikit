//go:build linux

package gpu

import (
	_ "embed"
	"fmt"
)

// cuda_vit.go publishes the transformer-encoder kernel set (docs/task-native-gpu.md,
// Phase 3) — the compute a ViT forward needs on top of the cuda.go device layer.
//
// These live in gpu/ rather than beside a vision backend because none of them is
// vision-specific: a quantized GEMM, a LayerNorm, a GELU, a bidirectional attention
// and a per-row int8 quantizer are the same ops a text encoder needs (Phase 4). The
// vision-specific part — how a SigLIP tower is wired out of these — lives in
// gpu/visioncuda, which is also where the aikit/vision import lives, so this package
// stays aikit-import-free like the rest of the device layer.
//
// Every kernel mirrors vision/encoder.go's arithmetic exactly (double-accumulated
// reductions, tanh-GELU, max-subtract softmax); see vit.cu for why that matters and
// where it is deliberately not the "standard" formulation.

// ViTPTX is the compiled vit.cu. Exported so a consumer can load it into a module it
// already owns. Regenerate with ./build_ptx.sh — never hand-edit.
//
//go:embed testdata/vit.ptx
var ViTPTX []byte

// Kernel entry points in ViTPTX.
const (
	// KernelQuantRows quantizes [rows,dim] f32 to int8 with a per-row scale, exactly
	// as linalg's quantizeRowInt8 does. Launch ONE BLOCK PER ROW.
	//   (x f32*, q int8*, s f32*, rows i32, dim i32)
	KernelQuantRows = "quant_rows"

	// KernelGEMMW8A8 is C[M,N] = (Aq[M,K]·Bq[N,K]) * aScale[M] * bScale[N]; B is
	// row-major [N,K], the B-transposed layout linalg.MatmulBT uses.
	//   (A int8*, aScale f32*, B int8*, bScale f32*, C f32*, M, N, K i32)
	KernelGEMMW8A8 = "gemm_w8a8"

	// KernelGEMMF32 is C[M,N] = A[M,K]·B[N,K] in f32, for unquantized weights.
	//   (A f32*, B f32*, C f32*, M, N, K i32)
	KernelGEMMF32 = "gemm_f32"

	// KernelAddBias does x[r,d] += bias[d] over [rows,dim].
	//   (x f32*, bias f32*, rows i32, dim i32)
	KernelAddBias = "add_bias"

	// KernelAddVec does x[i] += v[i] — positional embeddings and residual adds.
	//   (x f32*, v f32*, n i32)
	KernelAddVec = "add_vec"

	// KernelLayerNorm is mean/variance LayerNorm (not RMS) with weight and bias,
	// double-accumulated. Launch ONE BLOCK PER ROW.
	//   (x f32*, w f32*, b f32*, out f32*, rows i32, dim i32, eps f32)
	KernelLayerNorm = "layernorm"

	// KernelGELUTanh applies the tanh-approximation GELU in place.
	//   (x f32*, n i32)
	KernelGELUTanh = "gelu_tanh"

	// KernelAttention is bidirectional multi-head self-attention over np patches.
	// Launch ONE BLOCK PER (head, query) — i.e. nH*np blocks — with np*4 bytes of
	// dynamic shared memory for the score row.
	//   (q f32*, k f32*, v f32*, out f32*, np, nH, hd i32, scale f32)
	KernelAttention = "attention"

	// --- Qwen2.5-VL additions ---

	// KernelRMSNorm is weight-only RMS normalization (no mean subtraction, no bias).
	// Launch ONE BLOCK PER ROW.
	//   (x f32*, w f32*, out f32*, rows i32, dim i32, eps f32)
	KernelRMSNorm = "rmsnorm"

	// KernelRopeQK applies NeoX rotate_half 2D rotary IN PLACE to the q and k thirds
	// of a FUSED qkv buffer [seq, 3*hidden] (row layout [3, nH, hd]); v is untouched.
	// One thread per (patch, head, d<hd/2).
	//   (qkv f32*, cos f32*, sin f32*, seq, nH, hd i32)
	KernelRopeQK = "rope_qk"

	// KernelAttentionSeg is bidirectional MHA restricted to each patch's segment —
	// a window, or a whole image for the fullatt blocks. Reads q/k/v from the fused
	// qkv buffer. Launch ONE BLOCK PER (head, query) with maxSegment*4 bytes of
	// dynamic shared memory. segStart/segEnd are PER-PATCH bounds, not cu_seqlens.
	//   (qkv f32*, out f32*, segStart i32*, segEnd i32*, seq, nH, hd i32, scale f32)
	KernelAttentionSeg = "attention_seg"

	// KernelSiLUMul computes gate = silu(gate) * up — the gated-MLP activation.
	//   (gate f32*, up f32*, n i32)
	KernelSiLUMul = "silu_mul"

	// KernelGELUErf is the EXACT (erf) GELU that nn.GELU() defaults to, used by
	// Qwen's patch merger. NOT interchangeable with KernelGELUTanh — they differ by
	// ~5e-4, far above any parity bar. Pick deliberately.
	//   (x f32*, n i32)
	KernelGELUErf = "gelu_erf"

	// --- tiled GEMMs (throughput) ---

	// KernelGEMMW8A8Tiled is gemm_w8a8 staged through shared memory. Same signature,
	// and BIT-IDENTICAL output: the accumulator is int32 and integer addition is
	// associative, so re-chunking the K loop cannot change the result.
	// Launch with TileGrid(M, N).
	KernelGEMMW8A8Tiled = "gemm_w8a8_tiled"

	// KernelGEMMW8A8Reg is the register-blocked int8 GEMM: a 64x64 output tile per
	// block, 4x4 outputs per thread, packed-word shared staging and __dp4a. The
	// int8 twin of gemm_f32_reg (audit M-12) — gemm_w8a8_tiled computes ONE output
	// per thread from byte-granular shared memory, which is 32 ld.shared.u8 per 4
	// dp4a and LSU-bound, the shape the roofline campaign measured at 7% of roof
	// and retired from the ANN path. ALIGNED ONLY — M%64==0, N%64==0, K%16==0.
	// BIT-IDENTICAL to gemm_w8a8_tiled: an exact int32 accumulator scaled by
	// aScale[m]*bScale[n] in the same order. Use GEMMW8A8Plan rather than picking
	// it directly; same signature and binding order as gemm_w8a8_tiled.
	KernelGEMMW8A8Reg = "gemm_w8a8_reg"

	// KernelGEMMF32Reg is the register-blocked f32 GEMM: a 64x64 output tile per block,
	// 4x4 outputs per thread. ALIGNED ONLY — M%64==0, N%64==0, K%16==0 — so its inner
	// loops carry no bounds checks. Use GEMMF32Plan rather than picking it directly.
	// Same signature and binding order as gemm_f32_tiled.
	KernelGEMMF32Reg = "gemm_f32_reg"

	// KernelGEMMF32Tiled is gemm_f32 staged through shared memory. Same signature.
	// NOT bit-identical to the untiled kernel — f32 addition reassociates — so it is
	// gated on a tight relative bound instead. Launch with TileGrid(M, N).
	KernelGEMMF32Tiled = "gemm_f32_tiled"
)

// GEMMTile is the tiled GEMMs' tile width; it must match vit.cu's TILE, which sizes
// their shared-memory staging arrays statically.
const GEMMTile = 16

// ViTBlock is the block width the per-row/attention kernels reduce at; it must match
// vit.cu's LNBLOCK (the static __shared__ reduction arrays are sized to it).
//
// It is PART OF THE BIT-IDENTITY CONTRACT, not just an array size. The layernorm,
// rmsnorm and softmax kernels sum across blockDim.x (== this width) threads, and
// addition is not associative — in f64 as in f32 — so the summation order, and
// therefore the exact bits, is fixed by this value. Note what does NOT protect it:
// vit.cu carries a BIT-IDENTITY-EXEMPT declaration and the FMA lint deliberately
// skips it, and even a contracted file's explicit intrinsics would only remove the
// compiler's discretion over CONTRACTION — the reduction width is chosen here, in
// host code, where no kernel-level rule reaches.
//
// Do not sweep it for a speed win without re-baselining the ViT parity gate. That
// gate (gpu/visioncuda/encoder_test.go) is a cosine bound, deliberately — the CPU
// tower accumulates in float64 while these kernels work in f32/int32, so bit
// equality is impossible by construction — and a small consistent shift passes a
// tolerance while the numbers have moved. Change ViTBlock and LNBLOCK together; the
// five cross-thread `+=` reductions in vit.cu are each tagged at the edit point.
// (The max reductions are left untagged: max is associative, so width-independent.)
//
// This mirrors gpu/metal_vit.go's ViTBlock, which pinned the same hazard first. Two
// backends carry one coupling; both should say so.
const ViTBlock = 256

// ViT holds the compiled encoder kernel pipelines.
type ViT struct {
	QuantRows Pipeline
	GEMMW8A8  Pipeline
	GEMMF32   Pipeline
	AddBias   Pipeline
	AddVec    Pipeline
	LayerNorm Pipeline
	GELUTanh  Pipeline
	Attention Pipeline

	// Qwen2.5-VL additions.
	RMSNorm      Pipeline
	RopeQK       Pipeline
	AttentionSeg Pipeline
	SiLUMul      Pipeline
	GELUErf      Pipeline

	// Tiled GEMMs — same math, staged through shared memory.
	GEMMW8A8Tiled Pipeline
	GEMMF32Tiled  Pipeline
	// Register-blocked f32 GEMM (aligned fast path). Reach it via GEMMF32Plan.
	GEMMF32Reg Pipeline
	// Register-blocked int8 GEMM (aligned fast path). Reach it via GEMMW8A8Plan.
	GEMMW8A8Reg Pipeline
}

// NewViT loads ViTPTX on this device and builds every encoder pipeline. The module is
// tracked by the Device, so ReleaseObjects frees it.
func (d *Device) NewViT() (ViT, error) {
	lib, err := d.CompileLibrary(ViTPTX)
	if err != nil {
		return ViT{}, fmt.Errorf("cuda: load ViT module: %w", err)
	}
	var v ViT
	for _, bind := range []struct {
		name string
		dst  *Pipeline
	}{
		{KernelQuantRows, &v.QuantRows},
		{KernelGEMMW8A8, &v.GEMMW8A8},
		{KernelGEMMF32, &v.GEMMF32},
		{KernelAddBias, &v.AddBias},
		{KernelAddVec, &v.AddVec},
		{KernelLayerNorm, &v.LayerNorm},
		{KernelGELUTanh, &v.GELUTanh},
		{KernelAttention, &v.Attention},
		{KernelRMSNorm, &v.RMSNorm},
		{KernelRopeQK, &v.RopeQK},
		{KernelAttentionSeg, &v.AttentionSeg},
		{KernelSiLUMul, &v.SiLUMul},
		{KernelGELUErf, &v.GELUErf},
		{KernelGEMMW8A8Tiled, &v.GEMMW8A8Tiled},
		{KernelGEMMF32Tiled, &v.GEMMF32Tiled},
		{KernelGEMMF32Reg, &v.GEMMF32Reg},
		{KernelGEMMW8A8Reg, &v.GEMMW8A8Reg},
	} {
		p, err := d.NewComputePipeline(lib, bind.name)
		if err != nil {
			return ViT{}, err
		}
		*bind.dst = p
	}
	return v, nil
}

// RowGrid is the geometry the per-row kernels (quant_rows, layernorm) require: one
// block per row, ViTBlock threads cooperating on that row's reduction.
func RowGrid(rows int) LaunchConfig {
	if rows <= 0 {
		return LaunchConfig{}
	}
	return LaunchConfig{
		GridX: uint32(rows), GridY: 1, GridZ: 1,
		BlockX: ViTBlock, BlockY: 1, BlockZ: 1,
	}
}

// SegAttentionGrid is attention_seg's geometry: one block per (head, query), with the
// segment's score row in dynamic shared memory sized to the LARGEST segment (a window,
// or a whole image on the fullatt blocks).
func SegAttentionGrid(seq, heads, maxSeg int) LaunchConfig {
	if seq <= 0 || heads <= 0 || maxSeg <= 0 {
		return LaunchConfig{}
	}
	return LaunchConfig{
		GridX: uint32(seq * heads), GridY: 1, GridZ: 1,
		BlockX: ViTBlock, BlockY: 1, BlockZ: 1,
		SharedMemBytes: uint32(maxSeg * 4),
	}
}

// AttentionGrid is attention's geometry: one block per (head, query), with the score
// row staged in dynamic shared memory. np*4 bytes must fit the device's per-block
// shared limit (48 KB by default ⇒ np ≤ 12288), which every SigLIP/Qwen grid does.
func AttentionGrid(np, heads int) LaunchConfig {
	if np <= 0 || heads <= 0 {
		return LaunchConfig{}
	}
	return LaunchConfig{
		GridX: uint32(np * heads), GridY: 1, GridZ: 1,
		BlockX: ViTBlock, BlockY: 1, BlockZ: 1,
		SharedMemBytes: uint32(np * 4),
	}
}

// TileGrid is the tiled GEMMs' 2-D geometry: one GEMMTile×GEMMTile output tile per
// block. Both kernels bounds-check, so a grid that overhangs M or N is safe.
func TileGrid(M, N int) LaunchConfig {
	if M <= 0 || N <= 0 {
		return LaunchConfig{}
	}
	return LaunchConfig{
		GridX:  uint32((N + GEMMTile - 1) / GEMMTile),
		GridY:  uint32((M + GEMMTile - 1) / GEMMTile),
		GridZ:  1,
		BlockX: GEMMTile, BlockY: GEMMTile, BlockZ: 1,
	}
}

// RegBlock is gemm_f32_reg's output-tile width; it must match vit.cu's RBM/RBN. RegK is
// its K step (RBK). Both are alignment requirements for the fast path.
const (
	RegBlock = 64
	RegK     = 16
)

// GEMMF32Plan picks the f32 GEMM kernel and its launch geometry for a shape M×N×K —
// the CUDA analogue of Metal's ViT.GEMMF32Plan.
//
// The register-blocked kernel is chosen when M and N are multiples of 64 and K of 16,
// which every large encoder / ViT projection satisfies; anything else (odd tails, tiny
// shapes) falls back to gemm_f32_tiled, which bounds-checks. Both bind identically
// (A, B, C, M, N, K), so a caller only swaps the pipeline and config.
func (v ViT) GEMMF32Plan(M, N, K int) (Pipeline, LaunchConfig) {
	if M%RegBlock == 0 && N%RegBlock == 0 && K%RegK == 0 && M > 0 && N > 0 && K > 0 {
		return v.GEMMF32Reg, LaunchConfig{
			GridX: uint32(N / RegBlock), GridY: uint32(M / RegBlock), GridZ: 1,
			BlockX: RegBlock / RTN, BlockY: RegBlock / RTM, BlockZ: 1,
		}
	}
	return v.GEMMF32Tiled, TileGrid(M, N)
}

// IntRegK is gemm_w8a8_reg's K-step in ELEMENTS (IBK words of four int8); with
// IntRegBlock it forms the kernel's alignment requirement.
//
// 16 rather than gemm_f32_reg's 64-element step is deliberate: SigLIP-so400m's
// intermediate width is 4304, a multiple of 16 but NOT of 64, so a K%64 kernel
// would have missed the MLP projections that are most of a ViT's work.
const (
	IntRegBlock = 64
	IntRegK     = 16
)

// GEMMW8A8Plan picks the int8 GEMM kernel and its launch geometry for M×N×K —
// the int8 analogue of GEMMF32Plan (audit M-12).
//
// It needs only K%16==0: M and N edge tiles stage zeros and skip their stores,
// so the inner loop stays unguarded while any M and N are accepted. That
// matters because SigLIP-so400m's intermediate width is 4304 — a multiple of 16
// but not 64 — and an M%64/N%64 requirement would have excluded the MLP
// projections, which are most of a ViT's work. A K%16!=0 shape falls back to
// gemm_w8a8_tiled. Both bind identically (A, aScale, B, bScale, C, M, N, K) and
// produce identical bits, so a caller only swaps the pipeline and config.
func (v ViT) GEMMW8A8Plan(M, N, K int) (Pipeline, LaunchConfig) {
	if K%IntRegK == 0 && M > 0 && N > 0 && K > 0 {
		return v.GEMMW8A8Reg, LaunchConfig{
			GridX: uint32((N + IntRegBlock - 1) / IntRegBlock), GridY: uint32((M + IntRegBlock - 1) / IntRegBlock), GridZ: 1,
			BlockX: IntRegBlock / RTN, BlockY: IntRegBlock / RTM, BlockZ: 1,
		}
	}
	return v.GEMMW8A8Tiled, TileGrid(M, N)
}

// RTM/RTN are gemm_f32_reg's per-thread micro-tile dims; they must match vit.cu.
const (
	RTM = 4
	RTN = 4
)
