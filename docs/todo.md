# aikit todo — what to do next (post-v0.2.0)

State of the decoder follow-ups, grounded in
[`multi-model-plan.md`](multi-model-plan.md) §7–§8 and
[`milestones/G7-gguf.md`](milestones/G7-gguf.md). Effort key: **S** = a few days,
**M** = a week or two, **L** = a month+. Everything under "Shipped" is in the
[Unreleased] CHANGELOG, not yet tagged.

---

## Shipped since v0.2.0

### GGUF tokenizer — both families ✅

`tokenizer.LoadGGUF` chats from a bare `.gguf` with no sidecar:

- **SPM byte-fallback** (`tokenizer.ggml.model == "llama"`: Llama-2/Mistral/
  TinyLlama) — HF-parity-gated on TinyLlama.
- **Byte-level** (`gpt2`: Llama-3/Qwen/GPT-2) — `modeByteLevel`, pretokenizer
  knobs from `tokenizer.ggml.pre`; parity-gated against a real Llama-3.2-1B GGUF
  (`LoadGGUF` == `Load`(tokenizer.json) id-for-id, json itself HF-golden-validated).

### Resident quantization — the full ladder ✅

"The way to actually run big quantized models in small memory." All streamed at
load (no whole-model f32 spike), off mmap'd GGUFs, all SIMD, all parity-gated:

- **Streaming int8** (`Quant:"int8"`): quantize each tensor as it loads, free the
  f32 before the next — loads in ~¼ the RAM; a bare `.gguf` lands resident as
  int8 (`TestGGUF_int8_resident`).
- **mmap GGUF** (`embed.OpenGGUFMmap`): raw quantized bytes are reclaimable page
  cache, not a heap copy; the tokenizer no longer pages in the weights.
- **int4 group-quant** (`Quant:"int4"`): projections → group-wise 4-bit
  (`MatmulBTQ4`), ~⅛ f32; embedding + LM head stay int8 (logit-critical).
  TinyLlama argmax preserved, cosine 0.994.
- **int4 + int8 SIMD matmuls**: widen-to-scratch + `dotF32` — int4 **6.7×**,
  int8 **6.9×** over the scalar loops.
- **int8×int8 (W8A8)** (`Quant:"int8int8"`): activations also int8, true integer
  kernel (`dotI8` — AVX2 `VPMOVSXBW`+`VPMADDWD`, bit-exact, scalar fallback off
  amd64) — **3.4×** over the f32-widen int8. Lossier (cosine 0.9979), so opt-in.

The accuracy/speed ladder is now explicit: `f32 → int8 (0.9996) → int8int8
(3.4× faster, 0.9979) → int4 (⅛ f32)`.

---

## Still open, ranked

### 1. Tier-1: define what aikit *is* — highest value · M / M–L

[`ideas.md`](ideas.md) argues the biggest wins are at a higher altitude than
deepening the decoder. With the decoder + quant + GGUF depth essentially
complete, these are now the most impactful next step:

- **`rag`** — compose chunk → embed → ann/bm25 → fuse → encoder-rerank → decoder
  into one `Answer(query) → (text, []Citation)` pipeline. The product; makes the
  library more than the sum of its packages. Bigger integration, harder to
  validate offline (needs real models).
- **Constrained / structured generation** — logit-masking on the existing
  `Sampler` so a small model *cannot* emit malformed JSON/invalid grammar.
  Self-contained, builds on the `Sampler`, fully offline-validatable (the
  constraint is a hard invariant).

### 2. More GGUF quant types (Q5_K/Q3_K/IQ*) — incremental · S

Each is "a `dequant*` func + a size entry" on the existing GGUF seam, but needs a
Q5_K fixture or the Python `gguf` reference to parity-gate the dequant (Q4_K_M
already covers the dominant laptop quant, so this is low marginal value).

### 3. Incremental perf — incremental · S–M

- A NEON `dotI8` (SDOT) so the W8A8 path is SIMD off amd64 (it's scalar there
  today; mirror the f32 `dotNEON` path).
- mmap safetensors on the `fs.FS` path GGUF-style.

### 4. GPTQ / AWQ (safetensors-resident int4) — broadens coverage · M

"The other half of G7." Different packing (`qweight`/`qzeros`/`scales`/`g_idx`),
same dequant idea; the safetensors loader already handles the container. Adds the
HF-hosted int4 ecosystem.

### 5. Shared-expert MoE + YaRN/longrope — lowest urgency · S–M

"A couple more `MoEConfig` knobs" on the G6 MoE base for Qwen-MoE/DeepSeek, plus
YaRN's mscale for long context. Cleanly scoped, but only pays off for those
specific families.

---

## Recommendation

The decoder/quant/GGUF arc is complete. The highest-leverage next step is the
**Tier-1** work (#1) — `rag` for "the product," or constrained generation for a
self-contained, rigorously-testable new capability. The rest (#2–#5) are
incremental and can be picked up opportunistically.
