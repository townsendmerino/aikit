# Milestone G6 — Mixture of Experts (Mixtral)

Parent plan: [`docs/multi-model-plan.md`](../multi-model-plan.md) §6, §7 (G6).
Touches: `decoder/arch.go` (`MoEConfig`), `mlp.go` (`moeMLP` + router/top-k),
`weights.go` (`expertWeights`, `Router`/`Experts`, MoE loader branch + quant),
`config.go` (`validateMixtral` + MoE keys), `registry.go` (mixtral adapter).

Status: **DONE & validated on Linux 2026-06-03.** Mixtral-tiny (8 experts, top-2)
runs through the generic forward at **cosine 1−1e-13** vs the HF float32 oracle
(maxSampleΔ 0.00000). This is the one *structural* FFN addition in the plan — a
router + sparse experts replacing the dense MLP — and the only G-milestone that
adds a genuinely new compute path rather than a knob.

## What it proves

MoE slots into the existing `mlp()` dispatch as a third FFN variant alongside
gated (Ge/SwiGLU) and non-gated (GPT-2). Everything else about Mixtral is the
llama descriptor (RMSNorm no-offset, Pre2, SwiGLU experts, single-base RoPE, no
QK-norm, no bias, untied head), so the attention/norm/embedding paths are
untouched. The MoE forward is exactly HF's MixtralSparseMoeBlock:

```
probs  = softmax(Router·h)          // over all NumExperts
(w, e) = topk(probs, TopK)          // top-k experts + their probs
if NormTopKProb { w /= sum(w) }     // Mixtral renormalizes the chosen weights
out    = Σ_j w[j] · down(silu(gate(h)) ⊙ up(h))   // only the chosen experts run
```

The subtle ordering — softmax over ALL experts, THEN top-k, THEN renormalize the
selected probabilities — is what makes the logits match to 1e-13.

## What changed

- **`MoEConfig`** (`arch.go`): `{NumExperts, TopK, NormTopKProb}` on the
  descriptor; `arch.MoE != nil` selects the sparse path.
- **`moeMLP`** (`mlp.go`): router matmul → `softmaxF32` → `topK` → optional
  renormalization → weighted sum of the chosen SwiGLU experts. Only top-k of E
  experts are evaluated (the point of MoE). Added `softmaxF32` and `topK`
  helpers (O(k·E), tiny).
- **Weights** (`weights.go`): `expertWeights{Gate,Up,Down}` + `LayerWeights.Router`
  / `.Experts`; the loader branches on `arch.MoE` to load the router and E experts
  (Mixtral's `block_sparse_moe.gate` + `experts.N.w1/w2/w3`). `matmulWeights`
  includes the router + every expert, so int8 quant (M8) covers them — important
  since experts dominate an MoE checkpoint's size.
- **mixtral adapter + `validateMixtral`** + Config `num_local_experts` /
  `num_experts_per_tok` / `norm_topk_prob` (a `*bool` so absent ⇒ HF's default
  true). Uses full attention (recent HF Mixtral ignores the config's vestigial
  `sliding_window`).

## Validation

- `decoder/mixtral_test.go` (`-short`-gated): loads real Mixtral-tiny, asserts
  the MoE config (8x top-2), that 8 experts + an 8-row router loaded per layer,
  then matches the float32 oracle — argmax identical, sample/top-k ≤ 5e-3, cosine
  1−1e-13.
- `decoder/moe_test.go`: `TestSoftmaxF32`, `TestTopK` (descending, index
  tracking, k==n), `TestResolveArchitecture_mixtral` (descriptor + default
  norm_topk_prob + k>E rejection).
- Oracle: `scripts/pin_llama_forward.py testdata/mixtral-tiny mixtral`.

## Next

- **Qwen2-MoE / Qwen3-MoE / DeepSeek**: add a *shared expert* (always-on, summed
  with the routed output) and a sigmoid/normalized router variant — a couple more
  `MoEConfig` knobs on this base.
- **Real large MoE** (Mixtral-8x7B ≈ 90 GB): runnable once G7 quant lands; the
  code is identical, only the loader memory changes.
- **Perf**: experts are independent matmuls — a natural parallelism target for
  the M7 backend.
