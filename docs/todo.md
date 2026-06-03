# aikit todo — what to do next (post-v0.2.0)

Ranked recommendation for the decoder follow-ups, grounded in
[`multi-model-plan.md`](multi-model-plan.md) §7–§8 and
[`milestones/G7-gguf.md`](milestones/G7-gguf.md) "Scope / next". Effort key:
**S** = a few days, **M** = a week or two, **L** = a month+.

---

## The four decoder follow-ups, ranked (value × effort × risk)

### 1. GGUF tokenizer + more K-quants (Q5_K/Q3_K) — **SPM tokenizer DONE** ⭐ · S

Closes the loop v0.2.0 opened. ✅ **`tokenizer.LoadGGUF` shipped** for the
SentencePiece byte-fallback family (`tokenizer.ggml.model == "llama"`:
Llama-2/Mistral/TinyLlama), HF-parity-gated on TinyLlama, and `demo/gemma` now
chats from a bare `.gguf` end-to-end. Still open, both deferred for lack of a
parity fixture (the repo's bar is no unvalidated dequant/tokenizer code):

- **Byte-level GGUF tokenizer** (`gpt2` family: Llama-3/Qwen/GPT-2) — same
  `modeByteLevel` machinery, knobs from `tokenizer.ggml.pre`; needs a committed
  byte-level GGUF to gate (testdata has only the SPM/llama TinyLlama GGUF).
- **More K-quants (Q5_K/Q3_K/IQ\*)** — each "a `dequant*` func + a size entry",
  but needs a Q5_K fixture or the Python `gguf` reference to validate.

### 2. Resident int8/int4 matmul (dequant-per-tile) — highest *real* value · M–L

Both planning docs flag this as "the way to actually run big quantized models in
small memory." Today we dequant-to-f32 on load, so a Q4 model eats *f32 RAM* —
format coverage without the RAM win that is the whole point of quantization.
M8 (int8 weight quant) is the foundation. Touches the matmul hot path and the
`Backend` seam, so it wants care. Do deliberately, second.

### 3. GPTQ / AWQ (safetensors-resident int4) — broadens coverage · M

"The other half of G7." Different packing (`qweight`/`qzeros`/`scales`/`g_idx`),
same dequant-to-f32 idea; the safetensors loader already handles the container.
Adds the HF-hosted int4 ecosystem. Coverage-breadth, not loop-closing — inherits
the same RAM caveat as #2 until it lands.

### 4. Shared-expert MoE + YaRN/longrope — most contained, lowest urgency · S–M

"A couple more `MoEConfig` knobs" on the G6 MoE base for Qwen-MoE/DeepSeek, plus
YaRN's mscale for long context. Cleanly scoped, but only pays off for those
specific families.

---

## Meta-note: the highest-value next thing may be none of the four

[`ideas.md`](ideas.md) Tier 1 argues the biggest wins *define what aikit is*
rather than deepen the decoder:

- **`rag`** — compose all packages into one cited-answer pipeline (the product).
- **Constrained / structured generation** — logit-masking on the existing
  `Sampler` so a small model *cannot* emit malformed JSON.

If the real question is "what's most impactful," `rag` ranks above all four
above. The four deepen the decoder; the Tier-1 ideas define the library.

---

## Decision

**Now:** #1, the **GGUF tokenizer** (+ a couple more K-quants if cheap) — small,
closes the v0.2.0 story, unblocks a clean "point at a `.gguf`, get a chat" demo.
**Next substantial push:** #2, resident int4.
