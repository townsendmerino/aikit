# Milestone M9 — WebGPU backend (resident weights)

Parent plan: [`docs/gemma-decoder-plan.md`](../gemma-decoder-plan.md) §5, §6 (M9),
and cpu-acceleration §E. Touches: `encoder/gpu/gpu.go` (resident-buffer API),
`decoder/backend_gpu.go` / `backend_nogpu.go`, `demo/gemma`.

Status: **DONE & validated on Linux 2026-06-02** on a real NVIDIA RTX 2070
SUPER via **Vulkan**. `--backend webgpu` runs the whole decoder on the GPU and
produces correct output; weights are uploaded once and kept resident.

## What was implemented

- **Decoder WebGPU backend** behind `-tags gpu` (`backend_gpu.go`), reusing the
  shared `encoder/gpu` foundation. The default build (`backend_nogpu.go`) keeps
  the package pure-Go / no-cgo — `CGO_ENABLED=0 go build ./...` still works, and
  `--backend webgpu` without the tag falls back to CPU with a note.
- **Resident weights** (the per-call-transfer fix). `encoder/gpu` gained
  `UploadMatrix` + `ResidentMatrix` + `MatmulBTResident`: a weight matrix is
  uploaded to a GPU storage buffer **once** and reused. The decoder backend
  caches one resident buffer per weight (keyed by the slice's backing pointer —
  the same weight recurs every token), so each token only uploads the tiny M=1
  activation and reads the result back. On any per-call GPU error it falls back
  to the CPU matmul, so results are always correct.

## Why this mattered

The naive foundation re-uploads *both* operands every call. For the decoder
that meant re-sending the constant weights every token — the LM head alone is
the ~671 MB embedding. Before resident weights, 6 tokens took ~3 minutes;
after, 32 tokens take ~4 s. Without this fix the GPU backend was a correctness
demo only; with it, it's a usable (if not yet winning) execution path.

## Validation

- `TestWebGPUBackend_matchesCPU` (`-tags gpu`): the decoder's projection + LM-head
  shapes match the CPU matmul to ≤ 7e-5, including a **resident-reuse** call
  (same weight, fresh activation) — skips cleanly with no adapter.
- `TestWebGPU_forwardParity`: a full forward on the GPU backend reproduces the
  f32 oracle's argmax (`' Paris'`) — end-to-end proof that resident weights for
  every projection + the tied LM head give correct logits.
- `--backend webgpu` demo: `"…Paris. It is the most visited city in the world."`

## Performance (270M, RTX 2070 via Vulkan, 32-token generation)

| backend | wall (incl ~1 s load) | tok/s |
|---|---|---|
| CPU (AVX2, M7) | 3.2 s | ~14–18 |
| WebGPU (resident) | 5.3 s | ~8 |

The GPU **trails the CPU for single-token (M=1) decode** — exactly the plan's
prediction. With weights resident the bottleneck is no longer bandwidth but
**latency**: ~127 separate synchronous dispatch+readback round-trips per token.
The GPU wins where there's enough parallel work to amortize that — large-batch
prefill and the 1B/4B/27B checkpoints CPU can't serve — not 270M token-by-token.

## Known follow-ups (the remaining GPU work)

- **Resident activations / fused passes** (cpu-acceleration §E/§F): keep the
  hidden state on-device across a layer's matmuls and port norms/rope/softmax to
  WGSL, so a token isn't 127 host↔device round-trips. This is what would make
  the GPU competitive for decode.
- **Tiled WGSL matmul** (§F): the kernel is still naive one-invocation-per-output.
- **int8/int4 on GPU**: the resident buffers are f32; quantized GPU weights would
  cut upload + VRAM for the big checkpoints.
- **Batched prefill**: process the prompt as one M=L matmul to amortize latency.
