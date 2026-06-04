# aikit todo — what to do next (post-v0.2.0)

State of the decoder follow-ups, grounded in
[`multi-model-plan.md`](multi-model-plan.md) §7–§8 and
[`milestones/G7-gguf.md`](milestones/G7-gguf.md). Effort key: **S** = a few days,
**M** = a week or two, **L** = a month+. Everything under "Shipped" is in the
[Unreleased] CHANGELOG, not yet tagged.

---

## Shipped since v0.2.0

### GGUF tokenizer — both families ✅
`tokenizer.LoadGGUF` chats from a bare `.gguf` with no sidecar: SPM byte-fallback
(`model == "llama"`, HF-parity-gated on TinyLlama) and byte-level (`gpt2`:
Llama-3/Qwen/GPT-2, knobs from `tokenizer.ggml.pre`, parity-gated vs a real
Llama-3.2-1B GGUF id-for-id).

### Resident quantization — the full ladder ✅
"The way to actually run big quantized models in small memory." All streamed at
load (no whole-model f32 spike), off mmap'd GGUFs, all SIMD, all parity-gated:

- **Streaming int8/int4** (`Quant:"int8"|"int4"`): quantize each tensor as it
  loads, free the f32 before the next — int8 ~¼ f32, int4 ~⅛ f32 (group-wise,
  `MatmulBTQ4`; embedding/head stay int8). TinyLlama argmax preserved, cosine
  0.994 (int4).
- **mmap GGUF** (`embed.OpenGGUFMmap`): raw quantized bytes are reclaimable page
  cache; the tokenizer never pages in the weights.
- **SIMD quant matmuls**: int4 **6.7×** / int8 **6.9×** over the scalar loops
  (widen-to-scratch + `dotF32`).
- **int8×int8 (W8A8)** (`Quant:"int8int8"`): activations also int8, true integer
  kernel (`dotI8` — AVX2 `VPMOVSXBW`+`VPMADDWD` / arm64 NEON `SMULL`+`SADALP`,
  bit-exact, QEMU-validated) — **3.4×** over the f32-widen int8; lossier (0.9979),
  so opt-in.
- **Parallel load**: the per-layer dequant/re-quant fans out across cores
  (`parallelLayers`, GGUF + safetensors) — Mellum2-12B load ~2 min → **~20 s**,
  race-clean.

Ladder: `f32 (1.0) → int8 (0.9996) → int8int8 (3.4× faster, 0.9979) → int4 (⅛ f32)`.

### GPTQ + AWQ — safetensors-resident int4 ✅
HF int4 checkpoints load (`quantization_config` → `gptqReconstruct` / `awqReconstruct`
un-pack `qweight`/`qzeros`/`scales` ± `g_idx` to f32, then stream through the
resident int8/int4 path). GPTQ = AutoGPTQ packing (`[in/8,out]`, act-order via
`g_idx`); AWQ = AutoAWQ GEMM (`[in,out/8]`, `[0,4,1,5,2,6,3,7]` de-interleave).
Validated vs the committed f32 oracle on TinyLlama-1.1B (cosine 0.991 GPTQ / 0.996
AWQ, argmax preserved).

### Constrained / structured generation ✅
New `constrain` package: a logit mask (via the new
`decoder.SamplingParams.LogitProcessor` hook) forces output to satisfy a
byte-level grammar; ships a streaming JSON grammar — a model that *cannot* emit
malformed JSON. Proven by a random-logits hard-invariant test vs `encoding/json`
and `demo/gemma --json`.

### Mellum2 + YaRN ✅
JetBrains Mellum2 (`model_type "mellum"`, 12B-A2.5B MoE code model) **runs
end-to-end from a bare GGUF** — generates coherent code under `--quant int4` in
pure Go. Combines MoE-every-layer + 3:1 sliding/full interleave + QK-norm + the
one new piece, **YaRN** RoPE (HF-exact: NTK-by-parts + `attention_factor` mscale;
`TestYarn_matchesHF`, rel ≤ 1e-12). GGUF path handles stacked-expert MoE + Q5_0
dequant; `ggufConfig` dispatches `llama` + `mellum`. (Also unlocks YaRN for any
long-context Qwen/Llama.)

---

## Still open, ranked

### 1. Tier-1: `rag` — the product · M–L
[`ideas.md`](ideas.md)'s top idea, and now the biggest remaining win: compose
chunk → embed → ann/bm25 → fuse → encoder-rerank → decoder into one
`Answer(query) → (text, []Citation)` pipeline — making the library more than the
sum of its packages. Bigger integration, harder to validate offline (needs real
models). (Constrained generation, the other Tier-1 item, is shipped; its
follow-ups are a general GBNF/regex engine + a JSON-Schema → grammar compiler on
the same `Grammar` interface.)

### 2. More GGUF quant types (Q5_K/Q3_K/IQ*) — incremental · S
Each is "a `dequant*` func + a size entry" on the existing GGUF seam, but needs a
fixture or the Python `gguf` reference to parity-gate (Q4_K_M/Q5_0/Q6_K already
cover the common laptop mixes, so low marginal value).

### 3. Incremental perf — residual only · S
The two big wins shipped (parallel load + arm64 NEON `dotI8`; see Shipped above).
What's left is small:
- **Quantize direct from the source quant to int4**, skipping the f32 round-trip
  — the remaining load-time headroom is memory-bandwidth-bound on that
  intermediate.
- A faster **SDOT** NEON `dotI8` (vs the base-ISA SMULL/SADALP path) — needs
  ARMv8.2 DotProd + runtime feature detection.
- ~~mmap safetensors on the fs.FS path~~ — N/A: real directories already mmap
  (`openCheckpointMmap`); `fs.FS` is heap by necessity (no fd) and only serves
  small embedded test models.

### 4. Mellum2 polish · S
- **Exact `mellum2` tokenizer parity.** The GGUF byte-level tokenizer falls to
  GPT-2-style defaults for `tokenizer.ggml.pre == "mellum2"` (good enough for
  coherent output, not byte-exact). Pin a golden from the model's `tokenizer.json`
  and map its pretokenizer regex.
- **More GGUF architectures.** `ggufConfig` dispatches `llama` + `mellum`;
  qwen2/qwen3/gemma GGUFs are the same pattern (map `<arch>.*` metadata onto the
  existing descriptors) once a fixture is on hand.

### 5. Shared-expert MoE + longrope/dynamic RoPE — lowest urgency · S–M
A couple more `MoEConfig` knobs for shared-expert MoE (Qwen-MoE/DeepSeek), and the
remaining RoPE scalings (longrope/su, dynamic). Cleanly scoped, only pays off for
those families. (YaRN is done.)

---

## Models supported (decoder)

Gemma 3 · Qwen2.5/3 · Llama-2/3 · Mistral · GPT-2 · Mixtral · **Mellum2**.
Checkpoint formats: f32/bf16/f16 safetensors (single + sharded), **GPTQ + AWQ**
int4 safetensors, and **GGUF** (`llama` + `mellum` archs; F32/F16/Q8_0/Q4_0/Q5_0/
Q4_K/Q6_K). Any of these re-quantizes to resident int8/W8A8/int4.

---

## Recommendation

The decoder / quant / GGUF / structured-output arc is complete and broad (incl.
the perf items — parallel load + arm64 NEON W8A8). The single highest-leverage
next step is the **`rag` pipeline** (#1) — the "makes the library more than its
packages" feature. Everything else (#2–#5) is incremental and self-contained.
