// Package linalg holds aikit's shared CPU compute kernels: the dot products and matmuls, the quantized weight formats
// and the matmuls over them, and the assembly behind them, so aikit's encoder and ann, goinfer's decoder and ken use one
// copy. It is Experimental as a whole (aikit's README, § "Stability tiers").
//
// # The door: WeightMat and Workspace
//
// A weight matrix is a [WeightMat], built once and multiplied many times:
//
//   - Build: [WrapF32], [WrapF16], [WrapBF16], [WrapInt8], [WrapInt4], [WrapQ4K], [WrapQ5K] and [WrapQ6K] wrap bytes
//     the caller already holds (a file mapping, a loader's buffer) without copying; [QuantizeInt8], [QuantizeInt4],
//     [QuantizeF16] and [QuantizeBF16] quantize f32 weights. [RepackInt4InPlace] permutes canonical int4 bytes into
//     the layout this CPU's kernel runs, and [PreferredInt4Layout] reports which one that is.
//   - Multiply: [WeightMat.MatmulBTInto] computes dst = a · Wᵀ for any kind; [MatmulBTBatchInto] runs several weights
//     that share one activation (q‖k‖v, gate‖up) under one quantization and one fork/join.
//     [WeightMat.QuantizeActW4A8] with [WeightMat.MatmulBTW4A8PreInto] reuses one quantized activation across int4
//     matmuls.
//   - Read back: [WeightMat.Kind] (a [Kind] whose String is the stable name: "f32", "int4", "q4k", ...), and the raw
//     accessors a serializer or a GPU upload reads ([WeightMat.Int4F16], [WeightMat.Int8], [WeightMat.Q4K], ...).
//
// A [Workspace] is one goroutine's scratch: matmuls through it allocate nothing once warm. It carries the policy a
// call runs under: the parallel threshold and width ([Workspace.SetThreshold], [Workspace.SetWorkers]) and the
// activation-quantization group ([Workspace.SetActQuantGroup]). [SetParallelThreshold], [SetParallelWidth] and
// [SetActQuantGroup] set the process default a Workspace overrides, and [WeightMat.SetActQuantGroup] stamps a group
// on one weight.
//
// # Storage kinds
//
//	kind        bytes per weight          matmul
//	----------- ------------------------- ------------------------------------------------------------
//	f32         4                         f32 GEMM
//	f16, bf16   2                         a tile widened to f32, then the f32 GEMM
//	int8        1 + a scale per row       W8A8 (int8 activations) or weight-only Q8 (f32 activations)
//	int4        ½ + a binary16 scale per  W4A8; in the canonical, row4 (arm64 DotProd) or split-half
//	            group of 32               (amd64 AVX2) layout, all three bit-identical
//	q4k, q5k,   the GGUF K-quant blocks   an integer dot over the blocks (native on amd64 AVX2; see
//	q6k         as the file stores them   [KQuantNative])
//
// # What is bit-identical to what
//
// Parallelization and re-blocking split output columns or rows, and each output is one worker's full K reduction, so
// a fanned-out call equals the serial call of the same kernel bit for bit, and so does every M. The int4 layouts
// equal each other; a batch equals its ops run one by one through MatmulBTInto; a pre-quantized W4A8 call equals the
// call that quantizes. Against its scalar reference, an integer dot (int8 by int8) is exact; a kernel that accumulates
// in floating point (the f32 GEMM, the K-quant block scaling) differs by float reassociation, which its test bounds.
//
// # Dispatch
//
// Every kernel has a portable Go implementation; a faster one is chosen by GOARCH at build time and by the CPU's
// features at init (aikit's internal/cpufeat): on arm64 NEON, with SDOT where the CPU has DotProd; on amd64 AVX2 and
// FMA, F16C, AVX-512 VNNI (+VL) for the int8 and int4 dots, and POPCNT. A binary never executes an instruction its
// CPU lacks. [ActiveKernels] reports the tiers in use, [SelfCheck] compares each fast kernel with its reference at
// startup (and can turn a disagreeing tier off), and the aikit_no* build tags force a tier off ([ForcedFallbacks]).
//
// Every assembly entry point, its file, the function that dispatches to it and the test that holds it to its
// reference are listed in aikit's docs/internal/kernel-inventory.md, generated from the kernel registry each package
// keeps beside its tests (go run -C tools ./kernelinventory --write; a test fails when the two differ).
//
// # Raw-slice exceptions
//
// Five quantized matmuls still take raw slices instead of a WeightMat, each because its caller needs something the door
// does not offer yet. Each goes when that WeightMat form exists:
//
//   - [MatmulBTQ8Fused], [MatmulBTQ8FusedInto]: aikit's encoder, for its int8 weights at large M. A blocked GEMM that
//     widens the int8 rows inside the panel pack, bit-identical to dequantize-then-f32 MatmulBT; WeightMat's
//     weight-only Q8 path is a per-row dequant-dot with another accumulation order, so routing the encoder through it
//     would change its numerics. Waiting for: a blocked-GEMM option on a weight-only Q8 WeightMat.
//   - [MatmulBTW8A8Pre]: aikit's ann, for a paged int8 index's scan, which quantizes the query once and scores every
//     block against it (re-quantizing per block cost about 52 ms a query at 10M vectors). Waiting for: an int8
//     pre-quantized path on WeightMat, as [WeightMat.QuantizeActW4A8] and [WeightMat.MatmulBTW4A8PreInto] are for int4.
//   - [MatmulBTW4A8F16Into], [MatmulBTW4A8F16Pre]: goinfer's fused gate‖up, which hands each worker a range of one
//     weight's rows and applies SwiGLU in the same pass. Waiting for: a row-range view of a WeightMat.
//
// # Related packages
//
// The attention kernels (the f64-accumulating QK and AV products and the fused tile) are in linalg/attn, the
// element-wise ones (exp, tanh, erf, SiLU, GELU, softmax) in linalg/elem, and Hamming distance and sign-bit packing
// in linalg/bitvec. None of them imports linalg.
package linalg
